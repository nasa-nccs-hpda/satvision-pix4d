import os
import pandas as pd
import xarray as xr
import numpy as np
from datetime import timedelta
from goes2go.data import goes_nearesttime
import sys

# 1. Load metadata
# metadata_csv = sys.argv[1] 
metadata_csv = "/explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d/1-metadata/convection-filtered/20200101_cloudsystems_metadata.csv"
df = pd.read_csv(metadata_csv)
print(f"✅ Loaded {len(df)} records.")

# 2. Filter valid systems
df_filtered = df[
    (df["satellite"].isin(["GOES-East", "GOES-West"])) &
    (df["inside_inner_disk"] == True)
].reset_index(drop=True)
print(f"✅ Found {len(df_filtered)} systems inside inner disk.")

if df_filtered.empty:
    raise ValueError("No systems within coverage.")

# 3. Load TOPO lat/lon
first_sat = df_filtered.iloc[0]["satellite"]
geo_file = "/explore/nobackup/projects/pix4dcloud/jgong/ABI_EAST_GEO_TOPO_LOMSK.nc" if first_sat == "GOES-East" else "/explore/nobackup/projects/pix4dcloud/jgong/ABI_WEST_GEO_TOPO_LOMSK.nc"
ds_geo = xr.open_dataset(geo_file)
lat = ds_geo["Latitude"].isel(ydim=slice(1600, -1600), xdim=slice(1600, -1600))
lon = ds_geo["Longitude"].isel(ydim=slice(1600, -1600), xdim=slice(1600, -1600))

flat_lat = lat.values.ravel()
flat_lon = lon.values.ravel()
ny, nx = lat.shape
tile_size = 512
half = tile_size // 2

# 4. Output directory
output_dir = "/explore/nobackup/projects/pix4dcloud/jacaraba/abi_tiles_zarr_test"
os.makedirs(output_dir, exist_ok=True)

# 5. Main loop
n_saved = 0
for idx, rec in df_filtered.iterrows():

    conv_lat = rec["latitude"]
    conv_lon = rec["longitude"]
    dt = pd.to_datetime(rec["datetime"])
    satellite = rec["satellite"]
    sat_key = "goes16" if satellite == "GOES-East" else "goes17"

    # Nearest pixel
    dist = np.sqrt((flat_lat - conv_lat)**2 + (flat_lon - conv_lon)**2)
    y_idx, x_idx = np.unravel_index(dist.argmin(), lat.shape)

    if (y_idx - half < 0 or y_idx + half > ny or
        x_idx - half < 0 or x_idx + half > nx):
        continue

    # Build 7 timesteps
    conv_idx = np.random.randint(0, 7)
    dt_start = dt - timedelta(minutes=20 * conv_idx)
    times = [dt_start + timedelta(minutes=20*i) for i in range(7)]

    out_path = os.path.join(
        output_dir,
        f"{satellite.replace('-', '')}_sys{rec['system_id']}_{dt.strftime('%Y%m%dT%H%M')}.zarr"
    )

    if not os.path.exists(out_path):

        tile_list = []

        for t in times:

            timestep_tiles = []

            for band_num in range(1, 17):

                try:

                    ds = goes_nearesttime(
                        attime=t,
                        satellite=sat_key,
                        product="ABI-L1b-RadF",
                        domain="F",
                        download=True,
                        overwrite=False,
                        return_as="xarray",
                        bands=[band_num],
                        verbose=False,
                        save_dir=output_dir
                    )

                    rad = ds["Rad"]
                    print(rad.shape)

                    # Resample to 10848 × 10848 only if needed
                    ny0, nx0 = rad.shape
                    if (ny0, nx0) == (10848, 10848):
                        data = rad.values  # Already correct resolution
                    elif (ny0, nx0) == (5424, 5424):
                        data = np.repeat(np.repeat(rad.values, 2, axis=0), 2, axis=1)  # 2x upsample
                    elif (ny0, nx0) == (21696, 21696):
                        data = rad.values[::2, ::2]  # 2x downsample
                    elif (ny0, nx0) == (43200, 43200):  # unlikely but hypothetical 0.125 km?
                        data = rad.values[::4, ::4]
                    else:
                        raise ValueError(f"Unexpected shape: {rad.shape}")

                    rad = xr.DataArray(data, dims=["y", "x"], attrs=rad.attrs)
                    print(rad)

                    # Crop inner disk
                    rad = rad.isel(
                        y=slice(1600, rad.sizes["y"] - 1600),
                        x=slice(1600, rad.sizes["x"] - 1600)
                    )
                    rad = rad.assign_coords(
                        x=np.arange(rad.sizes["x"]),
                        y=np.arange(rad.sizes["y"])
                    )

                    tile = rad.isel(
                        y=slice(y_idx - half, y_idx + half),
                        x=slice(x_idx - half, x_idx + half)
                    )
                    print(rad.shape)

                    if np.all(np.isnan(tile.values)) or np.all(tile.values == 0):
                        raise ValueError("Empty or invalid tile")

                    timestep_tiles.append(tile)

                except Exception as e:
                    print(f"Skipping {t} band {band_num}: {e}")
                    break

            if len(timestep_tiles) != 16:
                print(f"Incomplete bands for timestep {t}")
                break

            timestep_stack = xr.concat(timestep_tiles, dim="band")
            tile_list.append(timestep_stack)

        if len(tile_list) != 7:
            print(f"Incomplete sequence for system {rec['system_id']}")
            continue

        stack = xr.concat(tile_list, dim="time")

        stack.to_zarr(out_path, mode="w")
        print(f"✅ Saved {out_path}")
        n_saved += 1

# Summary
if n_saved == 0:
    raise ValueError("No tiles saved.")
else:
    print(f"🎉 Done! Saved {n_saved} Zarr tiles.")
