import os
import random
import logging
import rioxarray as rxr
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


class BaseTileExtractor:

    def __init__(
                self,
                input_regex: str,
                output_folder: str,
                tile_size: int = 64,
                num_tiles_per_image: int = 10,
                stride: int = None,
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
        self.input_regex = input_regex
        self.output_folder = output_folder
        self.tile_size = tile_size
        self.num_tiles_per_image = num_tiles_per_image
        self.stride = stride if stride else tile_size
        self.output_extension = output_extension
        self.nodata = nodata
        self.num_workers = num_workers or os.cpu_count()

        os.makedirs(self.output_folder, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s; %(levelname)s; %(message)s'
        )

    def _find_valid_tile_positions(self, data, height, width):
        """
        Scans the image for valid tile positions while ensuring tiles
        are the correct size.

        Returns:
        - valid_positions (list of tuples): List of valid (x, y) tile coordinates.
        """
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
        """
        Extracts valid tiles from the shuffled list.

        Parameters:
        - data: Raster dataset.
        - image_path: File path of the original image.
        - valid_positions: List of valid (x, y) coordinates.
        """
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
        """
        Extracts exactly `num_tiles_per_image` valid tiles per image.
        """
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
        """
        Process all AVIRIS-NG .img files in parallel.
        """
        image_files = glob(self.input_regex)
        tqdm.write(f"📂 Found {len(image_files)} files to process.")

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            executor.map(self.extract_random_tiles, image_files)

        tqdm.write("✅ All images processed successfully!")
        return
    