import os
import logging
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from goes2go.data import goes_nearesttime

class ABIRandomTileExtractor:
    """
    Fully random ABI tile generator:
      - random timestamps (within a user time window)
      - random spatial tile centers (in-bounds)
      - 7 timesteps spaced 20 minutes
      - saves Zarr: [time, band, y, x]
    """

    def __init__(
        self,
        output_dir: str = "./abi_tiles",
        tile_size: int = 512,
        channels = list(range(1, 17)),
        satellite: str = "goes16",
        product: str = "ABI-L1b-RadF",
        domain: str = "F",
        download: bool = False,
        overwrite: bool = False,
        num_tiles: int = 1000,
        cadence_minutes: int = 20,
        n_timesteps: int = 2,
        # time window: pick something explicit; user can override in CLI later
        start_dt: str = "2020-01-01T00:00:00",
        end_dt: str = "2020-12-31T23:59:59",
        seed: int = 0,
    ):

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.data_output_dir = os.path.join(self.output_dir, "2-data")
        os.makedirs(self.data_output_dir, exist_ok=True)

        self.random_tiles_output_dir = os.path.join(self.output_dir, "3-tiles", "random")
        os.makedirs(self.random_tiles_output_dir, exist_ok=True)

        self.tile_size = int(tile_size)
        self.channels = list(channels)
        self.product = product
        self.domain = domain
        self.download = download
        self.overwrite = overwrite

        self.num_tiles = int(num_tiles)
        self.cadence_minutes = int(cadence_minutes)
        self.n_timesteps = int(n_timesteps)

        self.start_dt = pd.to_datetime(start_dt)
        self.end_dt = pd.to_datetime(end_dt)
        if self.end_dt <= self.start_dt:
            raise ValueError("end_dt must be after start_dt")

        self.rng = np.random.default_rng(seed)

        # unify satellite naming with your convection extractor
        # convection extractor uses "GOES-East"/"GOES-West" but goes2go expects goes16/goes17 etc.
        self.satellite = satellite  # goes16/goes17/goes18

        # caching (optional)
        self._abi_band_cache = {}

        logging.info('Initialized ABIRandomTileExtractor')

    def _prepare_rad(self, rad: xr.DataArray) -> xr.DataArray:
        """
        Keeps your existing approach: res handling + inner crop.
        """
        res = rad.shape[0] // 5424
        if res == 1:
            upsampled = np.repeat(np.repeat(rad.values, 2, axis=0), 2, axis=1)
            rad = xr.DataArray(upsampled, dims=["y", "x"], attrs=rad.attrs)
        elif res == 4:
            rad = rad.isel(x=slice(0, None, 2), y=slice(0, None, 2))

        crop = 1600
        rad = rad.isel(y=slice(crop, rad.sizes["y"] - crop),
                       x=slice(crop, rad.sizes["x"] - crop))
        rad = rad.assign_coords(x=np.arange(rad.sizes["x"]), y=np.arange(rad.sizes["y"]))
        return rad

    def _download_abi_stack(self, dt: datetime) -> xr.DataArray:
        """
        Returns [band, y, x]
        """
        band_stack = []
        missing = []

        for ch in self.channels:
            # print("DOWNLOADING CHANNELS", self.channels)
            key = (pd.Timestamp(dt).to_datetime64(), self.satellite, ch)
            if key in self._abi_band_cache:
                rad = self._abi_band_cache[key]
            else:
                try:
                    
                    ds = goes_nearesttime(
                        attime=dt,
                        satellite=self.satellite,
                        product=self.product,
                        domain=self.domain,
                        download=self.download,
                        overwrite=self.overwrite,
                        return_as="xarray",
                        bands=ch,
                        verbose=True,
                        save_dir=self.data_output_dir,
                    )
                    # print(ds)
                    rad = self._prepare_rad(ds["Rad"])
                    # print(rad)
                    self._abi_band_cache[key] = rad
                except Exception as e:
                    # print("Exception", e)
                    missing.append(ch)
                    continue

            band_stack.append(rad)

            # print("LEN BANDSTACK", len(band_stack))

        if missing:
            raise RuntimeError(f"Missing bands {missing} for {dt}")

        return xr.concat(band_stack, dim="band").assign_coords(band=np.array(self.channels, dtype=np.int16))

    def _extract_tile(self, arr: xr.DataArray, center_y: int, center_x: int) -> xr.DataArray:
        half = self.tile_size // 2
        return arr.isel(
            y=slice(center_y - half, center_y + half),
            x=slice(center_x - half, center_x + half)
        )

    def _sample_random_time(self) -> datetime:
        span_s = int((self.end_dt - self.start_dt).total_seconds())
        offset = int(self.rng.integers(0, span_s))
        # You can optionally snap to ABI cadence; leaving continuous is fine because goes_nearesttime snaps anyway
        return (self.start_dt + pd.to_timedelta(offset, unit="s")).to_pydatetime()

    def _sample_random_center(self, ny: int, nx: int) -> tuple[int, int]:
        half = self.tile_size // 2

        # valid center range
        min_y, max_y = half, ny - half
        min_x, max_x = half, nx - half

        # align to tile grid like your convection method (optional, but keeps tiling consistent)
        y = int(self.rng.integers(min_y, max_y))
        x = int(self.rng.integers(min_x, max_x))

        y = (y // self.tile_size) * self.tile_size + half
        x = (x // self.tile_size) * self.tile_size + half
        return y, x

    def gen_tiles(self):
        """
        Generate self.num_tiles random tiles and write each to zarr.
        """
        saved = 0
        attempts = 0

        logging.info('Starting to generate tiles.')
        while saved < self.num_tiles:
            attempts += 1
            dt = self._sample_random_time()
            logging.info(f'Sampled random time: {dt}')

            # choose a random conv_idx so dt lands somewhere inside the 7-step window
            conv_idx = int(self.rng.integers(0, self.n_timesteps))
            dt_start = dt - timedelta(minutes=self.cadence_minutes * conv_idx)
            times = [dt_start + timedelta(minutes=self.cadence_minutes * i) for i in range(self.n_timesteps)]

            logging.info(f'Testing a single timestep: {times[0]}')
            try:
                # print("BEFRE DOWNLOADING")
                stack0 = self._download_abi_stack(times[0])  # [band,y,x]
                # print("DOWNLOADED THE STACK")
                # print(stack0)
            except Exception as e:
                continue
            logging.info(f'Continuing with full timseries')

            ny, nx = stack0.sizes["y"], stack0.sizes["x"]
            y_center, x_center = self._sample_random_center(ny, nx)
            logging.info(f'y_center, x_center samples: {y_center}, {x_center}')

            tile_list = []
            ok = True
            for t in times:
                try:
                    stack = self._download_abi_stack(t)
                except Exception:
                    ok = False
                    break

                tile = self._extract_tile(stack, y_center, x_center)

                # enforce shape
                if tile.sizes["y"] != self.tile_size or tile.sizes["x"] != self.tile_size:
                    ok = False
                    break

                if np.all(np.isnan(tile.values)):
                    ok = False
                    break

                # print(tile)
                tile_list.append(tile)

            if not ok or len(tile_list) != self.n_timesteps:
                continue

            ts = xr.concat(tile_list, dim="time").assign_coords(
                time=np.array([np.datetime64(t) for t in times])
            )

            out_name = f"{self.satellite}_abi_{pd.Timestamp(dt).strftime('%Y%m%dT%H%M%S')}_rand{saved:07d}.zarr"
            out_path = os.path.join(self.random_tiles_output_dir, out_name)

            ts = ts.chunk({"time": 1, "band": 1, "y": self.tile_size, "x": self.tile_size})
            ts.to_zarr(out_path, mode="w")

            saved += 1
            if saved % 50 == 0:
                logging.info(f"Saved {saved}/{self.num_tiles} random tiles (attempts={attempts})")

        logging.info(f"Done. Saved {saved} random tiles.")
