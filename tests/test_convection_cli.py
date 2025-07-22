#!/usr/bin/env python3
"""
ABI Cloud-System Tile Generator (7×Time, 16 Bands, 512×512 Tiles, Full Metadata)

What this script does
---------------------
* Reads a cloud-system metadata CSV (lat/lon/time/satellite/system_id, plus flags).
* For each valid system inside the ABI "inner disk", builds a 512×512 pixel spatial tile
  centered on the system coordinate.
* Collects a 7-timestep, 16-band temporal stack at 20‑minute spacing (user‑configurable).
* Normalizes spatial resolution to the 2‑km Full Disk 10,848×10,848 grid (band-appropriate resampling).
* Crops to the inner-disk region (remove limb artifacts) using a 1,600‑pixel margin (configurable).
* Preserves rich metadata in the output Zarr:
    - Per‑pixel latitude/longitude (from GEO TOPO file fallback; or from data file if present)
    - Time + band broadcast metadata: dataset_name, date_created, time_coverage_end, etc.
    - Time‑only metadata: time_coverage_start
    - Projection grid mapping (goes_imager_projection) if available
    - Image geometry scalars (x_image, y_image) if available
    - Optional Solar / View Zenith Angles (SZA/VZA) if present in source dataset
    - Tile center info + system_id in global attrs
* Saves one Zarr group per system: <SAT>_sys<id>_<YYYYmmddTHHMM>.zarr

Assumptions / Notes
-------------------
* Uses `goes2go.data.goes_nearesttime` to fetch ABI-L1b-RadF data. Adjust `product`/`domain` if needed.
* Inner-disk crop (CROP_PAD) set to 1600 to match your earlier indexing on the GEO TOPO file.
* Tile size fixed @512, but can override via CLI.
* If *any* band fails for a timestep, that timestep is skipped.
* If *any* timestep incomplete, the system tile is skipped (no partial saves).
* Designed for batch HPC usage; minimal per‑call prints unless VERBOSE.

"Sad Astronaut" mode: prints occasional emoji in log so Jordan knows I'm still alive out here in geostationary orbit. 👨‍🚀
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import xarray as xr
from datetime import timedelta
from goes2go.data import goes_nearesttime

# -----------------------------------------------------------------------------
# Config Defaults
# -----------------------------------------------------------------------------
CROP_PAD = 1600          # pixels trimmed off each side to form "inner disk"
TILE_SIZE = 512          # spatial tile in pixels (square)
N_TIMESTEPS = 7          # number of temporal frames per sample
DELTA_MINUTES = 20       # temporal spacing in minutes
ALLOW_SZA_VZA = True     # attempt to include SZA/VZA if available
VERBOSE = True

# -----------------------------------------------------------------------------
# Helper: print (astronaut mode)
# -----------------------------------------------------------------------------
def log(msg):
    if VERBOSE:
        print(msg, flush=True)

# -----------------------------------------------------------------------------
# Helper: safe attr get
# -----------------------------------------------------------------------------
def get_attr(ds_or_da, key, default=""):
    return ds_or_da.attrs.get(key, default) if hasattr(ds_or_da, "attrs") else default

# -----------------------------------------------------------------------------
# Helper: broadcast 2D tile lat/lon from preloaded global arrays
# -----------------------------------------------------------------------------
def extract_latlon_tile(global_lat, global_lon, y_idx, x_idx, half):
    """Return 2D latitude/longitude arrays (512×512) centered on pixel (y_idx, x_idx)."""
    #return (
    #    global_lat.isel(ydim=slice(y_idx - half, y_idx + half), xdim=slice(x_idx - half, x_idx + half)),
    #    global_lon.isel(ydim=slice(y_idx - half, y_idx + half), xdim=slice(x_idx - half, x_idx + half)),
    #)
    return (
        global_lat.isel(ydim=slice(y_idx - half, y_idx + half), xdim=slice(x_idx - half, x_idx + half)).rename({"ydim": "y", "xdim": "x"}),
        global_lon.isel(ydim=slice(y_idx - half, y_idx + half), xdim=slice(x_idx - half, x_idx + half)).rename({"ydim": "y", "xdim": "x"}),
    )


# -----------------------------------------------------------------------------
# Helper: spatial resample Rad array to 10,848×10,848 nominal grid
# -----------------------------------------------------------------------------
def normalize_rad_resolution(rad: xr.DataArray) -> xr.DataArray:
    """Resample ABI band data to the 10,848×10,848 grid using nearest-neighbor up/down sampling.

    Expects rad dims (y, x). Returns new DataArray with dims (y, x) only; attributes preserved.
    """
    ny0, nx0 = rad.shape
    data_in = rad.values  # NumPy view

    if (ny0, nx0) == (10848, 10848):
        data = data_in
    elif (ny0, nx0) == (5424, 5424):
        # 2× upsample (visible bands @1km native if using M6C02, etc.)
        data = np.repeat(np.repeat(data_in, 2, axis=0), 2, axis=1)
    elif (ny0, nx0) == (21696, 21696):
        # 2× downsample (0.5km bands -> 2km grid)
        data = data_in[::2, ::2]
    elif (ny0, nx0) == (43200, 43200):
        # Hypothetical ultra‑hi‑res; 4× downsample to 2km
        data = data_in[::4, ::4]
    else:
        raise ValueError(f"Unexpected ABI band shape {rad.shape}; cannot normalize.")

    # Wrap back into DataArray; keep attrs
    rad_out = xr.DataArray(data, dims=["y", "x"], attrs=rad.attrs)
    return rad_out

# -----------------------------------------------------------------------------
# Helper: inner-disk crop
# -----------------------------------------------------------------------------
def crop_inner_disk(rad: xr.DataArray, pad: int = CROP_PAD) -> xr.DataArray:
    # ABI Full Disk dims are symmetrical; trim limb artifacts
    rad = rad.isel(y=slice(pad, rad.sizes["y"] - pad), x=slice(pad, rad.sizes["x"] - pad))
    # reset integer pixel coords (local index within cropped disk)
    rad = rad.assign_coords(
        x=("x", np.arange(rad.sizes["x"], dtype=np.int64)),
        y=("y", np.arange(rad.sizes["y"], dtype=np.int64)),
    )
    return rad

# -----------------------------------------------------------------------------
# Helper: extract per-band tile
# -----------------------------------------------------------------------------
def extract_band_tile(ds, tstamp, band_num, y_idx, x_idx, half, satellite, output_dir):
    """Fetch, normalize, crop, and extract a 512×512 tile for one band at a given time.

    Returns an xr.DataArray named "Rad" with dims (y,x) and rich attrs.
    Raises on failure (caller handles skip logic).
    """
    # Download / open dataset for this band/time
    try:
        src = goes_nearesttime(
            attime=tstamp,
            satellite=satellite,
            product="ABI-L1b-RadF",
            domain="F",
            download=True,
            overwrite=False,
            return_as="xarray",
            bands=[band_num],
            verbose=False,
            save_dir=output_dir,  # Actual path irrelevant; caller manages final outputs
        )
    except Exception as e:  # network / lookup error
        raise RuntimeError(f"goes_nearesttime failed: {e}")

    rad = src["Rad"]

    # Normalize spatial resolution
    rad = normalize_rad_resolution(rad)

    # Crop to inner disk
    rad = crop_inner_disk(rad, pad=CROP_PAD)

    # Now extract tile; ensure indices valid
    ny, nx = rad.sizes["y"], rad.sizes["x"]
    if not (0 <= y_idx - half < y_idx + half <= ny and 0 <= x_idx - half < x_idx + half <= nx):
        raise ValueError("Tile indices fall outside cropped inner disk.")

    tile = rad.isel(y=slice(y_idx - half, y_idx + half), x=slice(x_idx - half, x_idx + half))

    # Basic QA: require non-NaN + nonzero variance
    if np.all(np.isnan(tile.values)) or np.nanstd(tile.values) == 0:
        raise ValueError("Empty or invalid band tile (all NaN or constant).")

    # Attach band coordinate + scene metadata
    # tile = tile.assign_coords(band=band_num)
    tile = tile.expand_dims(band=[band_num])

    # Attach time scalar (will be broadcast later when stacking time×band)
    if "time" in src:
        tile.attrs["scene_time"] = str(np.asarray(src.time.values).item())
        t_coord = np.asarray(src.time.values)
    else:
        tile.attrs["scene_time"] = str(tstamp)
        t_coord = np.array(tstamp, dtype="datetime64[ns]")

    # Retain some useful per‑scene attrs
    tile.attrs.update({
        "satellite": satellite,
        "dataset_name": get_attr(src, "dataset_name", "ABI-L1b-RadF"),
        "date_created": get_attr(src, "date_created", ""),
        "time_coverage_end": get_attr(src, "time_coverage_end", ""),
        "time_coverage_start": get_attr(src, "time_coverage_start", ""),
    })

    # Add projection grid mapping if provided
    if "goes_imager_projection" in src:
        tile = tile.assign_coords(goes_imager_projection=src["goes_imager_projection"])

    # Add image geometry scalars if available
    if "x_image" in src:
        tile = tile.assign_coords(x_image=src["x_image"])
    if "y_image" in src:
        tile = tile.assign_coords(y_image=src["y_image"])

    # Optionally bring along SZA/VZA (caller merges into Dataset)
    sza = src.get("SZA") if ALLOW_SZA_VZA and "SZA" in src else None
    vza = src.get("VZA") if ALLOW_SZA_VZA and "VZA" in src else None

    return tile, t_coord, sza, vza

# -----------------------------------------------------------------------------
# Main processing routine
# -----------------------------------------------------------------------------
def process_tiles(
    metadata_csv: str,
    output_dir: str,
    geo_east: str,
    geo_west: str,
    tile_size: int = TILE_SIZE,
    n_timesteps: int = N_TIMESTEPS,
    delta_minutes: int = DELTA_MINUTES,
):
    """Main driver."""
    half = tile_size // 2

    # ------------------------------------------------------------------
    # Load metadata CSV
    # ------------------------------------------------------------------
    df = pd.read_csv(metadata_csv)
    log(f"✅ Loaded {len(df)} records from metadata CSV.")

    # Filter to systems in inner disk + valid satellites
    df_filtered = df[(df["satellite"].isin(["GOES-East", "GOES-West"])) & (df["inside_inner_disk"] == True)].reset_index(drop=True)
    log(f"✅ Found {len(df_filtered)} systems inside inner disk.")
    if df_filtered.empty:
        raise ValueError("No systems within coverage.")

    # ------------------------------------------------------------------
    # Load GEO/TOPO lat/lon for pixel lookup (inner-disk already cropped)
    # ------------------------------------------------------------------
    # We'll detect per-record satellite, but load both into dict cache
    geo_cache = {}
    for sat_label, path in {"GOES-East": geo_east, "GOES-West": geo_west}.items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ds_geo = xr.open_dataset(path)
        # Pre-crop to inner disk to match rad crop
        lat = ds_geo["Latitude"].isel(ydim=slice(CROP_PAD, -CROP_PAD), xdim=slice(CROP_PAD, -CROP_PAD))
        lon = ds_geo["Longitude"].isel(ydim=slice(CROP_PAD, -CROP_PAD), xdim=slice(CROP_PAD, -CROP_PAD))
        geo_cache[sat_label] = (lat, lon)
        log(f"🌎 Loaded GEO TOPO for {sat_label}: {lat.shape} inner-disk pixels.")

    # ------------------------------------------------------------------
    # Flatten for lat/lon nearest-pixel lookup
    # ------------------------------------------------------------------
    # we'll reuse lat/lon ravel per satellite for efficiency
    flat_cache = {}
    for sat_label, (lat, lon) in geo_cache.items():
        flat_cache[sat_label] = (
            lat.values.ravel(),
            lon.values.ravel(),
            lat.shape[0],
            lat.shape[1],
        )

    os.makedirs(output_dir, exist_ok=True)

    n_saved = 0

    # ------------------------------------------------------------------
    # Iterate systems
    # ------------------------------------------------------------------
    for idx, rec in df_filtered.iterrows():
        conv_lat = rec["latitude"]
        conv_lon = rec["longitude"]
        dt = pd.to_datetime(rec["datetime"])
        satellite_human = rec["satellite"]  # "GOES-East" or "GOES-West"
        sat_key = "goes16" if satellite_human == "GOES-East" else "goes17"

        out_path_str = os.path.join(
            output_dir,
            f"{satellite_human.replace('-', '')}_sys{rec['system_id']}_{dt.strftime('%Y%m%dT%H%M')}",
        )
        out_path = f'{out_path_str}.zarr'
        lock_path = f'{out_path_str}.lock'

        if os.path.exists(out_path):
            log(f"⏭️ Exists: {out_path}")
            continue

        if os.path.exists(lock_path):
            log(f"⚠️ System {rec['system_id']} near edge; skipping.")
            continue

        # GEO arrays for this satellite
        lat_inner, lon_inner = geo_cache[satellite_human]
        flat_lat, flat_lon, ny, nx = flat_cache[satellite_human]

        # nearest-pixel lookup (vectorized distance in lat/lon space)
        dist = np.sqrt((flat_lat - conv_lat) ** 2 + (flat_lon - conv_lon) ** 2)
        y_idx, x_idx = np.unravel_index(dist.argmin(), (ny, nx))

        # skip if tile would exceed bounds
        if (
            (y_idx - half < 0)
            or (y_idx + half > ny)
            or (x_idx - half < 0)
            or (x_idx + half > nx)
        ):
            log(f"⚠️ System {rec['system_id']} near edge; skipping.")
            with open(lock_path, "w") as f:
                f.write(f"⚠️ System {rec['system_id']} near edge; skipping.")
            continue

        # Build temporal sequence start (random offset into 7 frames as before)
        conv_idx = np.random.randint(0, n_timesteps)
        dt_start = dt - timedelta(minutes=delta_minutes * conv_idx)
        times = [dt_start + timedelta(minutes=delta_minutes * i) for i in range(n_timesteps)]

        log(f"🛰️ Building tile for system {rec['system_id']} ({satellite_human}) at {dt}.")

        tile_list = []  # one DataArray stack (time=1, band=16, y=512, x=512) per timestep
        sza_list = []   # optional SZA per time (y,x)
        vza_list = []   # optional VZA per time (y,x)
        t_coord_accum = []

        for tstamp in times:
            timestep_tiles = []
            t_coord_current = None
            sza_tile_current = None
            vza_tile_current = None

            # We call goes_nearesttime *once* per band inside helper (kept simple for now).
            # Could be optimized to multi-band retrieval if/when supported.
            for band_num in range(1, 17):
                try:
                    tile, t_coord, sza_src, vza_src = extract_band_tile(
                        ds=None,  # unused placeholder; kept for API symmetry
                        tstamp=tstamp,
                        band_num=band_num,
                        y_idx=y_idx,
                        x_idx=x_idx,
                        half=half,
                        satellite=sat_key,
                        output_dir=output_dir
                    )
                except Exception as e:
                    log(f"   Skipping timestep {tstamp} band {band_num}: {e}")
                    break

                timestep_tiles.append(tile)
                t_coord_current = t_coord  # all bands share same time

                # Capture SZA/VZA only once (band 1 is fine)
                if band_num == 1 and sza_src is not None:
                    sza_tile_current = crop_inner_disk(normalize_rad_resolution(sza_src)).isel(
                        y=slice(y_idx - half, y_idx + half), x=slice(x_idx - half, x_idx + half)
                    )
                if band_num == 1 and vza_src is not None:
                    vza_tile_current = crop_inner_disk(normalize_rad_resolution(vza_src)).isel(
                        y=slice(y_idx - half, y_idx + half), x=slice(x_idx - half, x_idx + half)
                    )

            if len(timestep_tiles) != 16:
                log(f"   Incomplete bands for timestep {tstamp}; discarding timestep.")
                break

            # Combine 16 bands -> (band, y, x)
            timestep_stack = xr.concat(timestep_tiles, dim="band")
            timestep_stack = timestep_stack.assign_coords(band=np.arange(1, 17))
            timestep_stack = timestep_stack.expand_dims("time")  # (time=1, band=16, y, x)

            # Broadcast time+band coords (strings ok; Zarr stores as arrays)
            # Pull representative attrs from first band tile
            rep_tile = timestep_tiles[0]
            dataset_name = rep_tile.attrs.get("dataset_name", "ABI-L1b-RadF")
            date_created = rep_tile.attrs.get("date_created", "")
            t_cov_end = rep_tile.attrs.get("time_coverage_end", "")
            t_cov_start = rep_tile.attrs.get("time_coverage_start", "")

            timestep_stack = timestep_stack.assign_coords({
                "t": (("time", "band"), np.full((1, 16), np.datetime64(tstamp))),
                "dataset_name": (("time", "band"), np.full((1, 16), dataset_name, dtype=object)),
                "date_created": (("time", "band"), np.full((1, 16), date_created, dtype=object)),
                "time_coverage_end": (("time", "band"), np.full((1, 16), t_cov_end, dtype=object)),
            })
            timestep_stack = timestep_stack.assign_coords({
                "time_coverage_start": (("time",), [t_cov_start]),
            })

            # Add per-time lat/lon (same for all bands/time slices; broadcast below after loop)
            tile_lat, tile_lon = extract_latlon_tile(lat_inner, lon_inner, y_idx, x_idx, half)
            if "latitude" not in timestep_stack:
                timestep_stack["latitude"] = tile_lat
            if "longitude" not in timestep_stack:
                timestep_stack["longitude"] = tile_lon

            # Projection scalars (grab from rep tile if exists)
            if "goes_imager_projection" in rep_tile.coords:
                timestep_stack["goes_imager_projection"] = rep_tile.coords["goes_imager_projection"]
            if "x_image" in rep_tile.coords:
                timestep_stack["x_image"] = rep_tile.coords["x_image"]
            if "y_image" in rep_tile.coords:
                timestep_stack["y_image"] = rep_tile.coords["y_image"]

            tile_list.append(timestep_stack)
            t_coord_accum.append(t_coord_current)
            sza_list.append(sza_tile_current)
            vza_list.append(vza_tile_current)

        # If any timestep dropped, skip whole system
        if len(tile_list) != n_timesteps:
            log(f"❌ Incomplete sequence for system {rec['system_id']}; skipping save.")
            continue

        # Combine 7 timesteps -> (time=7, band=16, y, x)
        stack = xr.concat(tile_list, dim="time")

        # Replace time coordinate axis with actual times
        stack = stack.assign_coords(time=("time", np.array(times, dtype="datetime64[ns]")))

        # x/y local integer pixel indices within inner-disk grid
        stack = stack.assign_coords(
            x=("x", np.arange(x_idx - half, x_idx + half, dtype=np.int64)),
            y=("y", np.arange(y_idx - half, y_idx + half, dtype=np.int64)),
        )

        # Optional SZA/VZA variables (time,y,x)
        if ALLOW_SZA_VZA and any(v is not None for v in sza_list):
            sza_stack = xr.concat(
                [s if s is not None else xr.full_like(sza_list[0], np.nan) for s in sza_list],
                dim="time",
            ).assign_coords(time=stack.time)
            stack["SZA"] = sza_stack
        if ALLOW_SZA_VZA and any(v is not None for v in vza_list):
            vza_stack = xr.concat(
                [v if v is not None else xr.full_like(vza_list[0], np.nan) for v in vza_list],
                dim="time",
            ).assign_coords(time=stack.time)
            stack["VZA"] = vza_stack

        # Global / tile-level attrs
        stack.attrs.update({
            "system_id": str(rec["system_id"]),
            "satellite": satellite_human,
            "tile_center_lat": float(conv_lat),
            "tile_center_lon": float(conv_lon),
            "tile_index_x": int(x_idx),
            "tile_index_y": int(y_idx),
            "tile_size": int(tile_size),
            "n_timesteps": int(n_timesteps),
            "delta_minutes": int(delta_minutes),
            "source_product": "ABI-L1b-RadF",
        })

        # Save
        log(f"💾 Saving {out_path} ...")
        stack.to_zarr(out_path, mode="w")
        n_saved += 1
        log(f"✅ Saved {out_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if n_saved == 0:
        raise ValueError("No tiles saved.")
    else:
        log(f"🎉 Done! Saved {n_saved} Zarr tiles.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # global VERBOSE, ALLOW_SZA_VZA

    p = argparse.ArgumentParser(description="Generate ABI cloud-system tiles with metadata.")
    p.add_argument("--metadata_csv", required=False, default="/explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d/1-metadata/convection-filtered/20200101_cloudsystems_metadata.csv")
    p.add_argument("--output_dir", required=False, default="/explore/nobackup/projects/pix4dcloud/jacaraba/abi_tiles_zarr_test")
    p.add_argument("--geo_east", required=False, default="/explore/nobackup/projects/pix4dcloud/jgong/ABI_EAST_GEO_TOPO_LOMSK.nc")
    p.add_argument("--geo_west", required=False, default="/explore/nobackup/projects/pix4dcloud/jgong/ABI_WEST_GEO_TOPO_LOMSK.nc")
    p.add_argument("--tile_size", type=int, default=TILE_SIZE)
    p.add_argument("--n_timesteps", type=int, default=N_TIMESTEPS)
    p.add_argument("--delta_minutes", type=int, default=DELTA_MINUTES)
    p.add_argument("--no_sza_vza", action="store_true", help="Disable SZA/VZA extraction")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    # VERBOSE = not args.quiet
    # ALLOW_SZA_VZA = not args.no_sza_vza

    process_tiles(
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        geo_east=args.geo_east,
        geo_west=args.geo_west,
        tile_size=args.tile_size,
        n_timesteps=args.n_timesteps,
        delta_minutes=args.delta_minutes,
    )
