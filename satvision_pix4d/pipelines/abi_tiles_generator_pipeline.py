import os
import random
import logging
import numpy as np
import xarray as xr
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


class ABITileExtractor:

    def __init__(
                self,
                stratification_strategy: str,
                output_dir: str = "./abi_tiles",
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
        """
        Initialize the AVIRIS-NG tile extractor.

        Parameters:
        - input_regex (str): Regex pattern to find AVIRIS-NG .img files.
        - output_folder (str): Path to save extracted GeoTIFF tiles.
        - tile_size (int): Size of square tile (default: 64x64).
        - num_tiles_per_image (int): Number of random tiles
             to extract per image.
        - stride (int): Step size for moving tiles. Default:
            tile_size (non-overlapping).
        - output_extension (str): Extension to output (str, npy)
        - num_workers (int): Number of parallel processes to use.
        """

        assert stratification_strategy in \
            ["random", "convection", "clouds", "landcover"]
        self.stratification_strategy: str = stratification_strategy
        self.output_dir: str = output_dir
        self.download: bool = download
        self.overwrite: bool = overwrite

        assert satellite in ["goes16", "goes17"]
        self.satellite: str = satellite
        self.domain: str = domain # ({'C', 'F', 'M'})

        if self.satellite == "goes16":
            self.default_projection = "+proj=geos +h=35786023 +a=6378137 +b=6356752.31414 +lon_0=-75 +sweep=x +no_defs"
            self.projection_file = "/explore/nobackup/projects/pix4dcloud/jgong/ABI_EAST_GEO_TOPO_LOMSK.nc"
        elif self.satellite == "goes17":
            self.default_projection = "+proj=geos +h=35786023 +a=6378137 +b=6356752.31414 +lon_0=-137.0 +sweep=x +no_defs"
            self.projection_file = "/explore/nobackup/projects/pix4dcloud/jgong/ABI_WEST_GEO_TOPO_LOMSK.nc"

        # setup geoference information
        self.source_coords = rxr.open_rasterio(self.projection_file)

        # Ensure lat/lon are 2D and match target's spatial dims
        self.source_lat_2d = self.source_coords["Latitude"].squeeze().values
        self.source_lon_2d = self.source_coords["Longitude"].squeeze().values

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

        # Get valid locations for tiles
        self.valid_centers = self._get_valid_tile_centers(
            self.source_lat_2d,
            self.source_lon_2d,
            self.tile_size,
            self.bounds[self.satellite]
        )

        os.makedirs(self.output_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s; %(levelname)s; %(message)s'
        )

    def _calculate_degrees(self, data_array):

        # Calculate latitude and longitude from GOES ABI fixed grid projection data
        # GOES ABI fixed grid projection is a map projection relative to the GOES satellite
        # Units: latitude in °N (°S < 0), longitude in °E (°W < 0)
        # See GOES-R Product User Guide (PUG) Volume 5 (L2 products) Section 4.2.8 for details & example of calculations
        # "file_id" is an ABI L1b or L2 .nc file opened using the netCDF4 library

        # Read in GOES ABI fixed grid projection variables and constants
        x_coordinate_1d = data_array['x'][:]  # E/W scanning angle in radians
        y_coordinate_1d = data_array['y'][:]  # N/S elevation angle in radians

        projection_info = data_array['goes_imager_projection']
        lon_origin = projection_info.longitude_of_projection_origin
        H = projection_info.perspective_point_height+projection_info.semi_major_axis
        r_eq = projection_info.semi_major_axis
        r_pol = projection_info.semi_minor_axis
        
        # Create 2D coordinate matrices from 1D coordinate vectors
        x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)
        
        # Equations to calculate latitude and longitude
        lambda_0 = (lon_origin*np.pi)/180.0  
        a_var = np.power(np.sin(x_coordinate_2d),2.0) + (np.power(np.cos(x_coordinate_2d),2.0)*(np.power(np.cos(y_coordinate_2d),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(y_coordinate_2d),2.0))))
        b_var = -2.0*H*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
        c_var = (H**2.0)-(r_eq**2.0)
        r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
        s_x = r_s*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
        s_y = - r_s*np.sin(x_coordinate_2d)
        s_z = r_s*np.cos(x_coordinate_2d)*np.sin(y_coordinate_2d)
        
        # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
        np.seterr(all='ignore')
        
        abi_lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
        abi_lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
        
        return abi_lat, abi_lon

    def _get_valid_tile_centers(self, lat, lon, tile_size, bounds):
        half = tile_size // 2
        height, width = lat.shape

        # Build base mask: valid lat/lon and no fill values
        valid_mask = (
            (lat > -90) & (lat < 90) &
            (lon > -180) & (lon < 360) &  # lon might be 0–360 in GOES-17
            (lat != -999) & (lon != -999)
        )

        # Exclude edges where a full tile wouldn't fit
        valid_mask[:half, :] = False
        valid_mask[-half:, :] = False
        valid_mask[:, :half] = False
        valid_mask[:, -half:] = False

        # Apply user-defined lon/lat bounds
        lon_min, lon_max = bounds['lon']
        lat_min, lat_max = bounds['lat']

        within_bounds = (
            (lat >= lat_min) & (lat <= lat_max) &
            (lon >= lon_min) & (lon <= lon_max)
        )

        combined_mask = valid_mask & within_bounds

        # Get valid (y, x) indices
        return np.argwhere(combined_mask)

    def _extract_tile(self, abi_var, center, tile_size):
        y_idx, x_idx = center
        half = tile_size // 2
        return abi_var[..., y_idx - half:y_idx + half, x_idx - half:x_idx + half]

    def _random_datetimes(self):
        """Generate N random datetime start points for tile extraction."""
        start = datetime(self.year, 1, 1)
        end = datetime(self.year + 1, 1, 1) - timedelta(minutes=self.duration_minutes)
        return [start + (end - start) * random.random() for _ in range(self.num_tiles)]

    def _random_point(self):
        """Return a random (lon, lat) point within satellite bounds."""
        lon_min, lon_max = self.bounds[self.satellite]["lon"]
        lat_min, lat_max = self.bounds[self.satellite]["lat"]
        lon = random.uniform(lon_min, lon_max)
        lat = random.uniform(lat_min, lat_max)
        return lon, lat

    def _download_and_stack_array_method(self, time: datetime):
        """Download and stack ABI bands across timesteps and channels."""

        # get the times for download
        times = []
        for i in range(self.duration_minutes // self.step_minutes):
            times.append(time + timedelta(minutes=i * self.step_minutes))

        # get the stack of tiles
        tiles_stack = {}

        # get a random number of tiles
        # maybe later, for now, lets extract 1000 per image
        local_num_tiles = 1000

        # Sample from precomputed valid centers
        centers = [
            self.valid_centers[np.random.randint(len(self.valid_centers))] for _ in range(local_num_tiles)]

        # Iterate over the range of times
        for t in times:
            band_stack = []
            for ch in self.channels:

                #try:

                # get xarray dataset
                dataset = goes_nearesttime(
                    attime=t,
                    satellite=self.satellite,
                    product=self.product,
                    domain=self.domain,
                    download=self.download,
                    overwrite=self.overwrite,
                    return_as="xarray",
                    save_dir=self.output_dir,
                    bands=ch,
                    verbose=False
                )

                # interpolate and sample
                resolution_check = dataset['Rad'].shape[0] // 5424
                C = dataset['Rad']
                if resolution_check == 1:
                    # Upsample by factor of 2 using interpolation
                    C = C.interp(
                        x=np.linspace(0, C.sizes['x'] - 1, C.sizes['x'] * 2),
                        y=np.linspace(0, C.sizes['y'] - 1, C.sizes['y'] * 2),
                        method='linear'
                    )
                elif resolution_check == 4:
                    # Downsample by selecting every 2nd pixel
                    C = C.isel(x=slice(0, None, 2), y=slice(0, None, 2))

                # Fix x and y coordinates
                C = C.assign_coords(
                    x=np.arange(C.sizes['x']),
                    y=np.arange(C.sizes['y'])
                )

                # Attach coordinates
                C = C.assign_coords(
                    latitude=(("y", "x"), self.source_lat_2d),
                    longitude=(("y", "x"), self.source_lon_2d))

                C.rio.write_crs(self.default_projection, inplace=True)
                band_stack.append(C)

                #except Exception:
                #    continue
                #break

            # concatenate bands
            abi_stack = xr.concat(band_stack, dim="band")

            # extract the tiles
            for tile_id in range(local_num_tiles):

                # if the key does not exist, create it
                if f'tile_{tile_id}' not in tiles_stack:
                    tiles_stack[f'tile_{tile_id}'] = []
                
                # append the tile for the time in the list
                tiles_stack[f'tile_{tile_id}'].append(
                    self._extract_tile(
                        abi_stack, centers[tile_id], self.tile_size
                    )
                )

            print(tiles_stack)
        
        # extract the tiles
        for tile_id in range(local_num_tiles):


            abi_temporal_stack = xr.concat(tiles_stack[f'tile_{tile_id}'], dim="time")
            print(abi_temporal_stack)
            abi_temporal_stack.chunk(
                {"time": 1, "band": 1, "y": 512, "x": 512}).to_zarr(f"{tile_id}.zarr", mode="w")
        

    def _download_and_stack_raster_method(self, time: datetime, lon: float, lat: float):
        """Download and stack ABI bands across timesteps and channels."""

        # get the times for download
        times = []
        for i in range(self.duration_minutes // self.step_minutes):
            times.append(time + timedelta(minutes=i * self.step_minutes))

        # get the stack of tiles
        tile_stack = []

        my_first_band = None

        for t in times:
            band_stack = []
            for ch in self.channels:

                print(ch, type(ch))
                #try:

                # get xarray dataset
                dataset = goes_nearesttime(
                    attime=t,
                    satellite=self.satellite,
                    product=self.product,
                    domain=self.domain,
                    download=self.download,
                    overwrite=self.overwrite,
                    return_as="xarray",
                    save_dir=self.output_dir,
                    bands=ch,
                    verbose=False
                )
                print(dataset.time_coverage_start.values, dataset.time_coverage_end.values)

                print("Shape of the radiance before", dataset['Rad'].shape)

                print(dataset.goes_imager_projection)
                print(dataset.goes_imager_projection.attrs)

                sat_height = dataset["goes_imager_projection"].attrs["perspective_point_height"]
                dataset = dataset.assign_coords({
                    "x": dataset["x"].values * sat_height,
                    "y": dataset["y"].values * sat_height,
                })
                
                crs = CRS.from_cf(dataset["goes_imager_projection"].attrs)
                dataset.rio.write_crs(crs.to_string(), inplace=True)
            
                da_web = dataset["Rad"].rio.reproject("epsg:6933")
                print("THE CRS", da_web.rio.crs)
                print("✅ Reprojected shape (1km):", da_web.shape)

                da_web.rio.to_raster(f'{ch}_6933_1km_proj.tif')


                """
                # read raster
                rad_array = dataset['Rad'].rio.write_crs(self.default_projection)
                print("THE CRS", rad_array.rio.crs)

                print("Shape of the radiance after CRS", rad_array.shape)

                from rasterio.warp import calculate_default_transform

                # Get transform and shape manually
                transform, width, height = calculate_default_transform(
                    rad_array.rio.crs,
                    "EPSG:6933",
                    rad_array.rio.width,
                    rad_array.rio.height,
                    *rad_array.rio.bounds(),
                    resolution=1000  # 1km resolution
                )

                # Now reproject using that transform and shape
                rad_array_1km = rad_array.rio.reproject(
                    dst_crs="EPSG:6933",
                    transform=transform,
                    shape=(height, width),
                    resampling=Resampling.cubic
                )

                print("✅ Reprojected shape (1km):", rad_array_1km.shape)
                print("Transform:", rad_array_1km.rio.transform())
                print("Bounds:", rad_array_1km.rio.bounds())


                #dataset = dataset.rio.reproject("EPSG:4326")
                #dataset['Rad'].rio.to_raster(f'{ch}_4326_1km.tif')
                rad_array = rad_array.rio.reproject(
                    dst_crs="EPSG:6933",  # Equal-Area Cylindrical (or use EPSG:3857)
                    #resolution=1000,      # 1km = 1000m
                    resampling=Resampling.cubic
                )
                print("Shape of the Radiance", rad_array.shape)
                print(rad_array.dims)
                print("TRANFORM", rad_array.rio.transform())
                # dataset['Rad'].rio.to_raster(f'{ch}_6933_1km_bilinear.tif')
                # print(dataset)
                """
                """
                # Step 2: Convert center point (lon, lat) to EPSG:6933
                from pyproj import Transformer
                transformer = Transformer.from_crs(self.default_projection, "EPSG:6933", always_xy=True)
                x_center, y_center = transformer.transform(lon, lat)

                # Step 3: Create 512 km × 512 km bounding box
                half_tile = 512_000 / 2
                geom = box(x_center - half_tile, y_center - half_tile, x_center + half_tile, y_center + half_tile)
                gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:6933")
                gdf.to_file('tile_shape.gpkg')

                # Step 4: Clip tile
                tile = dataset.rio.clip(gdf.geometry)
                print(tile.shape)
                """
                #except Exception:
                #    continue
                break
            break
            
        """
                    data = rxr.open_rasterio(file, masked=True)
                    data = data.rio.reproject("EPSG:4326")
                    geom = box(
                        lon - 0.25, lat - 0.25,
        #                lon + 0.25, lat + 0.25
        #            )
        #            clipped = data.rio.clip([geom], crs="EPSG:4326")
        #            if clipped.rio.shape[1] >= self.tile_size and clipped.rio.shape[2] >= self.tile_size:
        #                band_stack.append(clipped.isel(
        #                    x=slice(0, self.tile_size),
        #                    y=slice(0, self.tile_size)
        #                ))
        #        except Exception:
        #            continue

        #    if band_stack:
        #        time_stack = xr.concat(band_stack, dim="band")
        #        tile_stack.append(time_stack)

        #if tile_stack:
        #    return xr.concat(tile_stack, dim="time")
        #return None
        """

    def gen_tiles(self):
        datetimes = self._random_datetimes()
        logging.info(f'Generated {len(datetimes)} random times.')
        for i, dt in enumerate(tqdm(datetimes, desc="Generating ABI tiles")):
            stacked = self._download_and_stack_array_method(dt)
            # print(stacked)
            break
        #    if stacked is not None:
        #        filename = self.output_dir / f"{self.satellite}_tile_{i+1}_{dt.strftime('%Y%m%dT%H%M')}.tif"
        #        stacked.rio.to_raster(filename)

        return


    """
    def _find_valid_tile_positions(self, data, height, width):
        
        Scans the image for valid tile positions while ensuring tiles
        are the correct size.

        Returns:
        - valid_positions (list of tuples): List of valid (x, y) tile coordinates.
        
        tqdm.write(
            f"🔎 Scanning image for valid {self.tile_size}x{self.tile_size} tiles (stride={self.stride})...")

        valid_positions = []
        total_tiles = (
            (height - self.tile_size) // self.stride) * \
                ((width - self.tile_size) // self.stride)

        # Single progress bar for valid tile scanning
        with tqdm(total=total_tiles, desc="Scanning tiles", unit="tile", leave=True) as pbar:
            for x in range(0, height - self.tile_size, self.stride):
                for y in range(0, width - self.tile_size, self.stride):
                    tile = data.isel(x=slice(x, x + self.tile_size), y=slice(y, y + self.tile_size))

                    # Ensure tile is exactly `tile_size x tile_size` and has minimal no-data pixels
                    if tile.shape[1] == self.tile_size and tile.shape[2] == self.tile_size:
                        if not np.any(tile.values <= self.nodata):  # Adjust threshold if needed
                            valid_positions.append((x, y))

                    pbar.update(1)  # Update progress bar

        tqdm.write(
            f"✅ Found {len(valid_positions)} valid tile positions (stride={self.stride}).")
        return valid_positions

    def _extract_valid_tiles(self, data, image_path, valid_positions):
        
        Extracts valid tiles from the shuffled list.

        Parameters:
        - data: Raster dataset.
        - image_path: File path of the original image.
        - valid_positions: List of valid (x, y) coordinates.
        
        # Shuffle valid positions for randomness
        random.shuffle(valid_positions)

        # Ensure we don't request more tiles than available
        # If the number of tiles is set to negative, extract
        # all tiles from the dataset available
        if self.num_tiles_per_image < 0:
            num_tiles_to_extract = len(valid_positions)
        else:
            num_tiles_to_extract = min(
                self.num_tiles_per_image, len(valid_positions))
        if num_tiles_to_extract < self.num_tiles_per_image:
            tqdm.write(
                f'⚠️ Only {num_tiles_to_extract} valid tiles' +
                ' found, adjusting extraction count.')

        results = []
        for tile_index in range(num_tiles_to_extract):
            x, y = valid_positions[tile_index]

            tile = data.isel(
                x=slice(x, x + self.tile_size),
                y=slice(y, y + self.tile_size)
            )
            tile = tile.rio.reproject('ESRI:102001')

            # Double-check tile size
            if tile.shape[1] != self.tile_size \
                    or tile.shape[2] != self.tile_size:
                continue

            tile_filename = os.path.join(
                self.output_folder,
                f"{Path(image_path).stem}_tile_{tile_index+1}.{self.output_extension}"
            )

            if self.output_extension == 'tif':
                tile.rio.to_raster(
                    tile_filename,
                    BIGTIFF="IF_SAFER",
                    compress='LZW',
                    driver='GTiff',
                    dtype='float32'
                )
            else:
                np.save(
                    tile_filename, {"data": tile.values, "position": (x, y)}
                )

            results.append(f"✅ Saved: {tile_filename}")

        return results

    def extract_random_tiles(self, image_path):
        
        Extracts exactly `num_tiles_per_image` valid tiles per image.
        
        try:
            tqdm.write(f"📏 Processing: {image_path}")
            data = rxr.open_rasterio(image_path)
            height, width = data.shape[1], data.shape[2]
            tqdm.write(f"📏 Image size: {height}x{width}")

            valid_positions = self._find_valid_tile_positions(
                data, height, width)

            if not valid_positions:
                tqdm.write(
                    f"❌ No valid tiles found for {image_path}, skipping...")
                return

            results = self._extract_valid_tiles(
                data, image_path, valid_positions)

            for res in results:
                tqdm.write(res)

        except Exception as e:
            tqdm.write(f"❌ Error processing {image_path}: {e}")

    def process_all_images(self):
        
        Process all AVIRIS-NG .img files in parallel.
        
        image_files = glob(self.input_regex)
        tqdm.write(f"📂 Found {len(image_files)} files to process.")

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            executor.map(self.extract_random_tiles, image_files)

        tqdm.write("✅ All images processed successfully!")
        return
    """




class ABITileGenerator:
    def __init__(
        self,
        satellite: str = "goes16",
        date: datetime = datetime(2020, 6, 1, 12),
        duration_minutes: int = 120,
        step_minutes: int = 20,
        tile_size: int = 512,
        output_dir: str = "./abi_tiles",
        bands: list = None,
        num_tiles: int = 10,
        seed: int = 42,
    ):
        """
        Initialize the ABI tile generator.

        Args:
            satellite (str): 'goes16' or 'goes17'
            date (datetime): starting datetime for sampling
            duration_minutes (int): total time range to sample from
            step_minutes (int): sampling step in minutes
            tile_size (int): size of each tile in pixels
            output_dir (str): where to save the tiles
            bands (list): list of band numbers to use
            num_tiles (int): number of random tiles to extract
            seed (int): seed for reproducibility
        """
        assert satellite in ["goes16", "goes17"]
        self.satellite = satellite
        self.date = date
        self.duration_minutes = duration_minutes
        self.step_minutes = step_minutes
        self.tile_size = tile_size
        self.output_dir = output_dir
        self.bands = bands or list(range(1, 17))  # ABI has 16 bands
        self.num_tiles = num_tiles
        self.seed = seed

        # ABI coverage bounds
        self.bounds = {
            "goes16": {"lon": (-147.40, -2.6022), "lat": (-45.64, 45.64)},
            "goes17": {"lon": (150.425, 295.5748), "lat": (-43.6826, 43.6826)},
        }

        os.makedirs(self.output_dir, exist_ok=True)
        random.seed(self.seed)

    def _generate_timestamps(self):
        """Generate timestamps spaced by step_minutes for duration."""
        steps = self.duration_minutes // self.step_minutes
        return [self.date + timedelta(minutes=i * self.step_minutes) for i in range(steps + 1)]

    def _download_data(self, time: datetime, band: int):
        """Download a single band for a given time using GOES2go."""
        goes = GOES(self.satellite)
        ds = goes.nearesttime(
            product="ABI-L1b-RadC",
            start=time,
            scan="M3",
            channel=band,
            return_as="xr"
        )
        return ds

    def _get_random_tile_coords(self, width, height):
        """Generate random top-left corner (x, y) coordinates."""
        x = random.randint(0, height - self.tile_size)
        y = random.randint(0, width - self.tile_size)
        return x, y

    def _save_tile(self, data, path, timestamp):
        """Save tile as GeoTIFF and attach timestamp metadata."""
        data.rio.to_raster(
            path,
            driver="GTiff",
            compress="LZW",
            BIGTIFF="IF_SAFER"
        )

    def run(self):
        """Main function to run the tile generator."""
        timestamps = self._generate_timestamps()
        tqdm.write(f"📅 Using timestamps: {timestamps}")

        for i in range(self.num_tiles):
            time_idx = random.randint(0, len(timestamps) - 1)
            t = timestamps[time_idx]

            band_data = []
            for band in self.bands:
                try:
                    ds = self._download_data(t, band)
                    x, y = self._get_random_tile_coords(ds.dims["x"], ds.dims["y"])
                    tile = ds.isel(x=slice(x, x + self.tile_size), y=slice(y, y + self.tile_size))
                    band_data.append(tile.values)
                except Exception as e:
                    tqdm.write(f"⚠️ Error downloading band {band} at {t}: {e}")
                    continue

            if not band_data:
                tqdm.write("❌ No bands downloaded, skipping tile.")
                continue

            # Stack bands into shape (bands, tile_size, tile_size)
            tile_stack = np.stack(band_data, axis=0)
            tile_path = os.path.join(self.output_dir, f"{self.satellite}_tile_{i}_{t.strftime('%Y%m%dT%H%M')}.tif")

            # Create xarray object with spatial coords
            tile_xr = rxr.open_rasterio(ds.encoding["source"]).isel(x=slice(x, x + self.tile_size), y=slice(y, y + self.tile_size))
            tile_xr.data = tile_stack
            self._save_tile(tile_xr, tile_path, t)
            tqdm.write(f"✅ Saved {tile_path}")




class ABITileGeneratorV2:
    def __init__(
        self,
        satellite: str,
        year: int,
        num_tiles: int,
        output_dir: str,
        tile_size: int = 512,
        duration_minutes: int = 120,
        step_minutes: int = 20,
        product: str = "ABI-L1b-RadF",
        channels: list = None
    ):
        self.satellite = satellite
        self.year = year
        self.num_tiles = num_tiles
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.duration_minutes = duration_minutes
        self.step_minutes = step_minutes
        self.product = product
        self.channels = channels or [f"C{str(i).zfill(2)}" for i in range(1, 17)]
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define lat/lon bounds for inner circle of GOES satellites
        self.bounds = {
            "goes16": {"lon": (-147.40, -2.6022), "lat": (-45.64, 45.64)},
            "goes17": {"lon": (150.425, 295.5748), "lat": (-43.6826, 43.6826)}
        }

    def _random_datetimes(self):
        """Generate N random datetime start points for tile extraction."""
        start = datetime(self.year, 1, 1)
        end = datetime(self.year + 1, 1, 1) - timedelta(minutes=self.duration_minutes)
        return [start + (end - start) * random.random() for _ in range(self.num_tiles)]

    def _random_point(self):
        """Return a random (lon, lat) point within satellite bounds."""
        lon_min, lon_max = self.bounds[self.satellite]["lon"]
        lat_min, lat_max = self.bounds[self.satellite]["lat"]
        lon = random.uniform(lon_min, lon_max)
        lat = random.uniform(lat_min, lat_max)
        return lon, lat

    def _download_and_stack(self, time: datetime, lon: float, lat: float):
        """Download and stack ABI bands across timesteps and channels."""
        times = [time + timedelta(minutes=i * self.step_minutes) for i in range(self.duration_minutes // self.step_minutes)]
        tile_stack = []

        for t in times:
            band_stack = []
            for ch in self.channels:
                try:
                    file = GOES(
                        satellite=self.satellite,
                        product=self.product,
                        date=t,
                        channel=ch
                    ).file
                    data = rxr.open_rasterio(file, masked=True)
                    data = data.rio.reproject("EPSG:4326")
                    geom = box(
                        lon - 0.25, lat - 0.25,
                        lon + 0.25, lat + 0.25
                    )
                    clipped = data.rio.clip([geom], crs="EPSG:4326")
                    if clipped.rio.shape[1] >= self.tile_size and clipped.rio.shape[2] >= self.tile_size:
                        band_stack.append(clipped.isel(
                            x=slice(0, self.tile_size),
                            y=slice(0, self.tile_size)
                        ))
                except Exception:
                    continue

            if band_stack:
                time_stack = xr.concat(band_stack, dim="band")
                tile_stack.append(time_stack)

        if tile_stack:
            return xr.concat(tile_stack, dim="time")
        return None

    def run(self):
        datetimes = self._random_datetimes()
        for i, dt in enumerate(tqdm(datetimes, desc="Generating ABI tiles")):
            lon, lat = self._random_point()
            stacked = self._download_and_stack(dt, lon, lat)
            if stacked is not None:
                filename = self.output_dir / f"{self.satellite}_tile_{i+1}_{dt.strftime('%Y%m%dT%H%M')}.tif"
                stacked.rio.to_raster(filename)

