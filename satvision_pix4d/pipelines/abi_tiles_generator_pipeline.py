import os
import random
import logging
import numpy as np
import xarray as xr
import pandas as pd
import rioxarray as rxr
import geopandas as gpd

from glob import glob
from tqdm import tqdm
from pyproj import CRS
from pathlib import Path
from shapely.geometry import box
from rasterio.enums import Resampling
from datetime import datetime, timedelta
from goes2go import goes_latest, GOES
from goes2go.data import goes_nearesttime
from concurrent.futures import ProcessPoolExecutor

from satvision_pix4d.readers.convection_reader \
    import ConvectionMetadataParser


class ABITileExtractor:

    def __init__(
                self,
                stratification_strategy: str,
                convection_regex: str = None,
                output_dir: str = "./abi_tiles",
                local_abi_data_dir: str = '/css/geostationary/BackStage/GOES-16-ABI-L1B-FULLD',
                download: bool = False,
                overwrite: bool = False,
                satellite: str = "goes16",
                domain: str = 'F',
                year: int = 2020,
                num_tiles: int = 100,
                tile_size: int = 512,
                duration_minutes: int = 120,
                step_minutes: int = 20,
                product: str = "ABI-L1b-RadF",
                channels: list = list(range(1, 17)),
                seed: int = 42,
                output_extension: str = 'npy',
                nodata: int = -9999,
                num_workers: int = None
            ):

        assert stratification_strategy in \
            ["random", "convection", "clouds", "landcover"]
        self.stratification_strategy: str = stratification_strategy
        self.output_dir: str = output_dir
        self.download: bool = download
        self.overwrite: bool = overwrite

        # set local data dir
        self.local_abi_data_dir = local_abi_data_dir

        # for convection values, TODO: assert
        # cannot be None if stratification_strategy is
        # convection
        self.convection_regex = convection_regex

        assert satellite in ["goes16", "goes17"]
        self.satellite: str = satellite
        self.domain: str = domain  # ({'C', 'F', 'M'})

        if self.satellite == "goes16":
            self.default_projection = "+proj=geos +h=35786023 +a=6378137 +b=6356752.31414 +lon_0=-75 +sweep=x +no_defs"
            self.projection_file = "/explore/nobackup/projects/pix4dcloud/jgong/ABI_EAST_GEO_TOPO_LOMSK.nc"
        elif self.satellite == "goes17":
            self.default_projection = "+proj=geos +h=35786023 +a=6378137 +b=6356752.31414 +lon_0=-137.0 +sweep=x +no_defs"
            self.projection_file = "/explore/nobackup/projects/pix4dcloud/jgong/ABI_WEST_GEO_TOPO_LOMSK.nc"

        # setup geoference information
        # self.source_coords = rxr.open_rasterio(self.projection_file)

        # Ensure lat/lon are 2D and match target's spatial dims
        # self.source_lat_2d = self.source_coords["Latitude"].squeeze().values
        # self.source_lon_2d = self.source_coords["Longitude"].squeeze().values

        self.year: int = year
        self.num_tiles: int = num_tiles
        self.tile_size: int = tile_size
        self.duration_minutes: int = duration_minutes
        self.step_minutes: int = step_minutes
        self.product: str = product
        self.channels: list = channels
        self.seed: int = seed
        self.output_extension: str = output_extension
        self.nodata: int = nodata
        self.num_workers: int = num_workers or os.cpu_count()

        # ABI coverage bounds
        self.bounds = {
            "goes16": {"lon": (-147.40, -2.6022), "lat": (-45.64, 45.64)},
            "goes17": {"lon": (150.425, 295.5748), "lat": (-43.6826, 43.6826)},
        }
        random.seed(self.seed)

        # TODO: This should be moved somewhere else
        # because this is only valid for random times
        # Get valid locations for tiles
        # self.valid_centers = self._get_valid_tile_centers(
        #    self.source_lat_2d,
        #    self.source_lon_2d,
        #    self.tile_size,
        #    self.bounds[self.satellite]
        # )

        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f'Working dir to output data to: {self.output_dir}')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s; %(levelname)s; %(message)s'
        )

    def gen_tiles(self):
        """
        Generate tiles based on the selected stratification strategy.
        """
        logging.info(f"Proceeding to generate tiles via '{self.stratification_strategy}' stratification.")

        strategy_funcs = {
            "random": self._gen_tiles_random,
            "convection": self._gen_tiles_convection,
            "clouds": self._gen_tiles_clouds,
            "landcover": self._gen_tiles_landcover
        }

        func = strategy_funcs.get(self.stratification_strategy)
        if func is None:
            raise ValueError(f"❌ Unsupported stratification strategy: {self.stratification_strategy}")

        func()
        return

    def _gen_tiles_random(self):
        raise NotImplementedError

    def _gen_tiles_clouds(self):
        raise NotImplementedError

    def _gen_tiles_landcover(self):
        raise NotImplementedError

    def _gen_tiles_convection(self):

        # get convection filenames
        convection_filenames = glob(self.convection_regex)
        logging.info(
            f'Found {len(convection_filenames)} convection filenames.')

        # output dir for convection metadata
        convection_output_dir = os.path.join(
            self.output_dir,
            '1-metadata',
            'convection'
        )

        # iterate over each convection file
        for filename in convection_filenames[:1]:

            # get filename metadata
            convection_parser = ConvectionMetadataParser(
                filename, convection_output_dir)
            convection_metadata_df = convection_parser.generate_metadata()
            print(convection_metadata_df)

            # generate tiles after metadata is taken
            # option #1: we will extract the middle timestamp
            # per convective system, making sure we have one tile
            # per convective system
            # option #2: get one tile per timestamp from the convective
            # system, which would give us multiple tiles per convective
            # system, no unique convection systems

            # Sort to ensure chronological order
            convection_metadata_df = convection_metadata_df.sort_values(
                ["system_id", "datetime"]).reset_index(drop=True)

            # Pick the middle timestep for each system
            def get_middle_row(group):
                return group.iloc[len(group)//2]

            # Get the middle timestep for each system
            middle_df = convection_metadata_df.groupby(
                "system_id", group_keys=False).apply(
                    get_middle_row).reset_index(drop=True)

            logging.info("✅ Middle timestep per system extracted.")

            # extract the tile
            self._extract_from_convection_metadata(middle_df)

        return

    def _extract_from_convection_metadata(self, metadata_df):

        logging.info(
            f"Extracting tiles for {len(metadata_df)} unique systems...")

        for _, rec in tqdm(
            metadata_df.iterrows(), total=len(metadata_df), desc="Extracting ABI tiles"):

            dt = pd.to_datetime(rec["datetime"])
            sys_id = rec["system_id"]
            y_center = rec["center_y"]
            x_center = rec["center_x"]

            # Create the 14 timesteps
            times = [dt + timedelta(minutes=20 * i) for i in range(-7, 7)]
            print(times)

            offsets = {
                "center": (0, 0),
                "top": (-self.tile_size // 2, 0),
                "bottom": (self.tile_size // 2, 0),
                "left": (0, -self.tile_size // 2),
                "right": (0, self.tile_size // 2)
            }

            # For each offset position
            for pos, (dy, dx) in offsets.items():

                tile_list = []

                # For each timestep
                for t in times:

                    # Download single timestep
                    abi_stack = self._download_abi_stack(t)

                    # Extract tile at offset position
                    tile = self._extract_tile(
                        abi_stack,
                        y_center + dy,
                        x_center + dx
                    )
                    tile_list.append(tile)

                # Concatenate tiles over time dimension
                time_stack = xr.concat(tile_list, dim="time")

                # Save compressed chunked Zarr
                fname = self.output_dir / f"abi_{dt.strftime('%Y%m%dT%H%M')}_sys{sys_id}_{pos}.zarr"
                time_stack.chunk({"time": 1, "band": 1, "y": 512, "x": 512}).to_zarr(fname, mode="w")

                logging.info(f"Saved tile: {fname}")

        return

    def _download_abi_stack(self, dt):
        """
        Load ABI bands for a given datetime, preferring local files if configured.
        """
        band_stack = []
        year = dt.strftime("%Y")
        doy = dt.strftime("%j")   # day of year
        hour = dt.strftime("%H")

        for ch in self.channels:

            local_file_found = False

            if self.local_abi_data_dir:

                # Try to load locally
                pattern = os.path.join(
                    self.local_abi_data_dir,
                    year,
                    doy,
                    hour,
                    f"OR_ABI-L1b-RadF-M6C{ch:02d}_G16_s{year}{doy}{hour}*.nc"
                )
                files = sorted(glob(pattern))
                if files:
                    f = files[0]
                    ds = xr.open_dataset(f)
                    rad = ds["Rad"]
                    rad = rad.assign_coords(
                        x=np.arange(rad.sizes["x"]),
                        y=np.arange(rad.sizes["y"])
                    )
                    rad.rio.write_crs(self.default_projection, inplace=True)
                    band_stack.append(rad)
                    local_file_found = True
                else:
                    logging.warning(f"No local file for band {ch} at {dt}.")

            if not local_file_found:
                # Use goes_nearesttime as fallback
                logging.info(f"Downloading band {ch} from AWS for {dt}.")
                ds = goes_nearesttime(
                    attime=dt,
                    satellite=self.satellite,
                    product=self.product,
                    domain=self.domain,
                    download=self.download,
                    overwrite=self.overwrite,
                    return_as="xarray",
                    bands=ch,
                    verbose=False
                )
                rad = ds["Rad"]
                rad = rad.assign_coords(
                    x=np.arange(rad.sizes["x"]),
                    y=np.arange(rad.sizes["y"])
                )
                rad.rio.write_crs(self.default_projection, inplace=True)
                band_stack.append(rad)

        if not band_stack:
            raise RuntimeError(f"No bands loaded for datetime {dt}.")

        return xr.concat(band_stack, dim="band")

    def _extract_tile(self, arr, center_y, center_x):
        half = self.tile_size // 2
        return arr[..., center_y - half:center_y + half, center_x - half:center_x + half]
