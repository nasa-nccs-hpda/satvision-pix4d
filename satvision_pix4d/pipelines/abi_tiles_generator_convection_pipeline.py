import os
import logging
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from glob import glob
from tqdm import tqdm
from goes2go.data import goes_nearesttime

import pandas as pd
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from scipy.spatial import cKDTree

class ABIConvectionTileExtractor:
    def __init__(
        self,
        output_dir="./abi_tiles",
        tile_size=512,
        channels=list(range(1, 17)),
        local_data_dir=None,
        product="ABI-L1b-RadF",
        domain="F",
        download=True,
        overwrite=False,
        num_workers=None
    ):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s; %(levelname)s; %(message)s"
        )

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f'Created output dir: {self.output_dir}')

        self.data_output_dir = os.path.join(self.output_dir, "2-data")
        os.makedirs(self.data_output_dir, exist_ok=True)
        logging.info(f'Created data output dir: {self.data_output_dir}')

        self.convection_tiles_output_dir = os.path.join(
            self.output_dir, "3-tiles", "convection")
        os.makedirs(self.convection_tiles_output_dir, exist_ok=True)
        logging.info(f'Created convection tiles output dir: {self.convection_tiles_output_dir}')

        self.tile_size = tile_size
        self.channels = channels
        self.local_data_dir = local_data_dir
        self.product = product
        self.domain = domain
        self.download = download
        self.overwrite = overwrite
        self.num_workers = num_workers or os.cpu_count()

        self.projection_files = {
            "GOES-East": "/explore/nobackup/projects/pix4dcloud/jgong/ABI_EAST_GEO_TOPO_LOMSK.nc",
            "GOES-West": "/explore/nobackup/projects/pix4dcloud/jgong/ABI_WEST_GEO_TOPO_LOMSK.nc",
        }
        self._latlon_cache = {}
        self._abi_band_cache = {}

    def _load_latlon(self, satellite):
        if satellite in self._latlon_cache:
            return self._latlon_cache[satellite]

        ds_geo = xr.open_dataset(self.projection_files[satellite])
        lat = ds_geo["Latitude"].values
        lon = ds_geo["Longitude"].values
        self._latlon_cache[satellite] = (lat, lon)
        return lat, lon

    def _prepare_rad(self, rad):
        res = rad.shape[0] // 5424
        if res == 1:
            upsampled = np.repeat(
                np.repeat(rad.values, 2, axis=0), 2, axis=1)
            rad = xr.DataArray(
                upsampled,
                dims=["y", "x"],
                attrs=rad.attrs
            )
        elif res == 4:
            rad = rad.isel(x=slice(0, None, 2), y=slice(0, None, 2))

        crop = 1600
        rad = rad.isel(
            y=slice(crop, rad.sizes["y"] - crop),
            x=slice(crop, rad.sizes["x"] - crop)
        )
        rad = rad.assign_coords(
            x=np.arange(rad.sizes["x"]),
            y=np.arange(rad.sizes["y"])
        )
        return rad

    def _download_abi_stack(self, dt, satellite):
        band_stack = []
        for ch in self.channels:
            key = (dt, satellite, ch)
            if key in self._abi_band_cache:
                rad = self._abi_band_cache[key]
            else:
                ds = goes_nearesttime(
                    attime=dt,
                    satellite="goes16" if satellite == "GOES-East" else "goes17",
                    product=self.product,
                    domain=self.domain,
                    download=self.download,
                    overwrite=self.overwrite,
                    return_as="xarray",
                    bands=ch,
                    verbose=False,
                    save_dir=self.data_output_dir
                )
                rad = self._prepare_rad(ds["Rad"])
                self._abi_band_cache[key] = rad
                logging.info(
                    f"Downloaded {satellite} band {ch} for {dt}")
            band_stack.append(rad)
        return xr.concat(band_stack, dim="band")

    def _extract_tile(self, arr, center_y, center_x):
        half = self.tile_size // 2
        return arr.isel(
            y=slice(center_y - half, center_y + half),
            x=slice(center_x - half, center_x + half)
        )

    def gen_tiles(self, convection_metadata_df):
        # Keep only inside disk
        convection_metadata_df = convection_metadata_df[
            convection_metadata_df["inside_inner_disk"] == True
        ].reset_index(drop=True)

        records = []

        # Process each satellite separately so the KDTree matches
        for satellite in ["GOES-East", "GOES-West"]:
            sat_df = convection_metadata_df[convection_metadata_df["satellite"] == satellite]
            if sat_df.empty:
                continue

            lat_grid, lon_grid = self._load_latlon(satellite)
            lat_flat = lat_grid.ravel()
            lon_flat = lon_grid.ravel()

            # Build a KDTree of (lat, lon) grid
            tree = cKDTree(np.c_[lat_flat, lon_flat])

            # Query all points at once
            query_points = np.c_[sat_df["latitude"].values, sat_df["longitude"].values]
            dists, indices = tree.query(query_points, k=1)

            # Convert indices back to 2D pixel indices
            y_idx, x_idx = np.unravel_index(indices, lat_grid.shape)
            y_idx -= 1600
            x_idx -= 1600

            # Compute tile centers
            y_tile = (y_idx // self.tile_size) * self.tile_size + self.tile_size // 2
            x_tile = (x_idx // self.tile_size) * self.tile_size + self.tile_size // 2

            # Store records
            for i, (_, rec) in enumerate(sat_df.iterrows()):
                records.append({
                    "satellite": satellite,
                    "y_tile": y_tile[i],
                    "x_tile": x_tile[i],
                    "datetime": pd.to_datetime(rec["datetime"]),
                    "system_id": rec["system_id"]
                })

        df_tiles = pd.DataFrame(records)
        logging.info(
            f'Original systems, {df_tiles_unique.shape[0]} tiles to process.')

        # 🟢 Remove duplicates per satellite+tile center
        df_tiles_unique = df_tiles.drop_duplicates(
            subset=["satellite", "y_tile", "x_tile"])
        logging.info(
            f'Removed duplicates, {df_tiles_unique.shape[0]} tiles left.')

        # 🟢 Extract tiles
        for _, row in tqdm(df_tiles_unique.iterrows(), total=len(df_tiles_unique), desc="Extracting ABI tiles"):

            satellite = row["satellite"]
            y_tile = row["y_tile"]
            x_tile = row["x_tile"]
            dt = row["datetime"]

            times = [dt + timedelta(minutes=20 * i) for i in range(-7, 7)]
            offsets = {
                "center": (0, 0),
                "top": (-self.tile_size // 2, 0),
                "bottom": (self.tile_size // 2, 0),
                "left": (0, -self.tile_size // 2),
                "right": (0, self.tile_size // 2)
            }

            for pos, (dy, dx) in offsets.items():
                tile_list = []
                for t in times:
                    abi_stack = self._download_abi_stack(t, satellite)
                    tile = self._extract_tile(abi_stack, y_tile + dy, x_tile + dx)
                    if np.all(np.isnan(tile.values)):
                        logging.warning(f"Skipping {t} empty tile")
                        continue
                    tile_list.append(tile)

                if not tile_list:
                    continue

                time_stack = xr.concat(tile_list, dim="time")
                filename = os.path.join(
                    self.convection_tiles_output_dir,
                    f"{satellite.replace('-','')}_abi_{dt.strftime('%Y%m%dT%H%M')}_ytile{y_tile}_xtile{x_tile}_{pos}.zarr"
                )
                time_stack.chunk({"time": 1, "band": 1, "y": 512, "x": 512}).to_zarr(filename, mode="w")
                logging.info(f"Saved {filename}")

    def clear_cache(self):
        self._abi_band_cache.clear()

def process_csv(csv_path):
    df = pd.read_csv(csv_path)
    logging.info(f'Processing {csv_path}')
    extractor = ABIConvectionTileExtractor(
        output_dir=os.path.join(
            "/explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d/3-tiles/convection",
            f"abi_tiles_{Path(csv_path).stem}"
        ),
        tile_size=512,
        channels=list(range(1, 17)),
        download=True,
        overwrite=False
    )
    extractor.gen_tiles(df)
    extractor.clear_cache()
    return


if __name__ == "__main__":

    csv_files = glob(
        "/explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d/1-metadata/convection-filtered/2020*_cloudsystems_metadata.csv"
    )

    N_WORKERS = 40
    with Pool(processes=N_WORKERS) as pool:
        pool.map(process_csv, csv_files)
