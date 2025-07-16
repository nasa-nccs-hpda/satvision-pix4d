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
        missing_bands = []

        for ch in self.channels:
            key = (dt, satellite, ch)
            if key in self._abi_band_cache:
                rad = self._abi_band_cache[key]
            else:
                try:
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
                        save_dir=self.data_output_dir,
                    )
                #except FileNotFoundError:
                except:
                    missing_bands.append(ch)
                    continue

                rad = self._prepare_rad(ds["Rad"])
                self._abi_band_cache[key] = rad

            band_stack.append(rad)

        if missing_bands:
            logging.warning(
                f"⚠️ Skipping timestep {dt} ({satellite}): missing bands {missing_bands}"
            )
            raise RuntimeError(
                f"Missing bands {missing_bands} for timestep {dt}. Skipping."
            )

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

        # Process each record independently (no deduplication)
        for _, rec in tqdm(
            convection_metadata_df.iterrows(),
            total=len(convection_metadata_df),
            desc="Extracting ABI tiles"
        ):
            dt = pd.to_datetime(rec["datetime"])
            satellite = rec["satellite"]

            if pd.isna(satellite):
                logging.warning(f"Skipping system {rec['system_id']} with NaN satellite.")
                continue

            lat_grid, lon_grid = self._load_latlon(satellite)
            lat_flat = lat_grid.ravel()
            lon_flat = lon_grid.ravel()

            # Find nearest grid point
            dist = np.sqrt(
                (lat_flat - rec["latitude"]) ** 2 +
                (lon_flat - rec["longitude"]) ** 2
            )
            idx_min = dist.argmin()
            y_idx, x_idx = np.unravel_index(idx_min, lat_grid.shape)

            y_idx -= 1600
            x_idx -= 1600

            # Compute center tile indices
            y_tile = (y_idx // self.tile_size) * self.tile_size + self.tile_size // 2
            x_tile = (x_idx // self.tile_size) * self.tile_size + self.tile_size // 2

            # Random position index [0-6]
            conv_idx = np.random.randint(0, 7)

            # Compute the first timestep so that dt lands at conv_idx
            dt_start = dt - timedelta(minutes=20 * conv_idx)

            # 7 timesteps
            times = [dt_start + timedelta(minutes=20 * i) for i in range(7)]

            tile_list = []
            for t in times:
                abi_stack = self._download_abi_stack(t, satellite)
                tile = self._extract_tile(abi_stack, y_tile, x_tile)

                if np.all(np.isnan(tile.values)):
                    logging.warning(f"Skipping {t}: empty tile")
                    continue

                tile_list.append(tile)

            if not tile_list:
                logging.warning(f"No valid tiles for system {rec['system_id']}")
                continue

            time_stack = xr.concat(tile_list, dim="time")
            filename = os.path.join(
                self.convection_tiles_output_dir,
                f"{satellite.replace('-','')}_abi_{dt.strftime('%Y%m%dT%H%M')}_sys{rec['system_id']}.zarr"
            )
            time_stack.chunk({"time": 1, "band": 1, "y": 512, "x": 512}).to_zarr(filename, mode="w")
            logging.info(
                f"Saved {filename} (convective timestep index: {conv_idx})"
            )

    def clear_cache(self):
        self._abi_band_cache.clear()

def process_csv(csv_path):
    df = pd.read_csv(csv_path)
    logging.info(f'Processing {csv_path}')
    extractor = ABIConvectionTileExtractor(
        output_dir="/explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d",
        tile_size=512,
        channels=list(range(1, 17)),
        download=True,
        overwrite=False
    )
    extractor.gen_tiles(df)
    extractor.clear_cache()
    return


if __name__ == "__main__":

    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    csv_files = glob(
        "/explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d/1-metadata/convection-filtered/2020*_cloudsystems_metadata.csv"
    )

    N_WORKERS = 40
    with Pool(processes=N_WORKERS) as pool:
        pool.map(process_csv, csv_files)
