import os
import logging
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta


class ConvectionMetadataParser:
    """
    Parses convection mask NetCDF files to produce metadata records
    containing datetime, system ID, and system centroid pixel locations.
    Also assigns satellite coverage and whether inside inner disk.
    """

    def __init__(self, convection_filename: str, output_dir: str):
        self.convection_filename = convection_filename
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_filename = os.path.join(
            output_dir,
            f'{Path(self.convection_filename).stem}_metadata.csv'
        )

    def classify_satellite_and_region(self, lat, lon_raw):
        """
        Hybrid logic:
        - For GOES-East, use normalized longitude (-180/+180)
        - For GOES-West, use raw 0–360 longitude
        """
        # GOES-East bounds (-180/+180)
        east_lon_min, east_lon_max = -147.4, -2.6022
        east_lat_min, east_lat_max = -45.64, 45.64
        # GOES-West bounds (0–360)
        west_lon_min, west_lon_max = 150.425, 295.5748
        west_lat_min, west_lat_max = -43.6826, 43.6826

        # Normalize for East
        lon_norm = lon_raw - 360 if lon_raw > 180 else lon_raw

        if east_lon_min <= lon_norm <= east_lon_max and east_lat_min <= lat <= east_lat_max:
            return "GOES-East", True, lon_norm
        elif west_lon_min <= lon_raw <= west_lon_max and west_lat_min <= lat <= west_lat_max:
            return "GOES-West", True, lon_norm
        elif east_lon_min <= lon_norm <= east_lon_max:
            return "GOES-East", False, lon_norm
        elif west_lon_min <= lon_raw <= west_lon_max:
            return "GOES-West", False, lon_norm
        else:
            return "None", False, lon_norm

    def generate_metadata(self):
        if os.path.exists(self.output_filename):
            logging.info(f"Skipping processing since file exists. Loading: {self.output_filename}")
            return pd.read_csv(self.output_filename), None

        metadata = []

        ds = xr.open_dataset(self.convection_filename)
        time_hours = ds["time"].values
        latitudes = ds["lat"].values
        longitudes = ds["lon"].values

        date_str = (Path(self.convection_filename).stem).split('_')[0]
        date_obj = datetime.strptime(date_str, "%Y%m%d")

        system_time_index = {}

        for t_idx, hour_since_midnight in tqdm(
            enumerate(time_hours),
            desc="Processing timesteps",
            total=len(time_hours)
        ):
            dcs = ds["DCS_number"].isel(nt=t_idx)
            unique_ids = np.unique(dcs.values)
            unique_ids = unique_ids[unique_ids > 0]
            dt = date_obj + timedelta(hours=float(hour_since_midnight))
            logging.info(f"Found {len(unique_ids)} unique systems at {dt}")

            for sys_id in unique_ids:
                if sys_id not in system_time_index:
                    system_time_index[sys_id] = []
                system_time_index[sys_id].append(dt)

                mask = dcs.values == sys_id
                yx = np.argwhere(mask)
                if yx.size > 0:
                    y, x = yx.mean(axis=0)

                    lat = float(latitudes[int(y)])
                    lon_raw = float(longitudes[int(x)])
                    satellite, in_inner_disk, lon_norm = self.classify_satellite_and_region(lat, lon_raw)

                    metadata.append({
                        "datetime": dt,
                        "system_id": int(sys_id),
                        "center_y": int(y),
                        "center_x": int(x),
                        "latitude": lat,
                        "longitude": lon_raw,
                        "longitude_norm": lon_norm,
                        "satellite": satellite,
                        "inside_inner_disk": in_inner_disk
                    })

        metadata_df = pd.DataFrame(metadata)

        lifetime_records = []
        for sys_id, timestamps in tqdm(system_time_index.items(), desc="Computing lifetimes"):
            timestamps_sorted = sorted(timestamps)
            start_time = timestamps_sorted[0]
            end_time = timestamps_sorted[-1]
            n_steps = len(timestamps)
            duration_min = n_steps * 20

            lifetime_records.append({
                "system_id": int(sys_id),
                "start_time": start_time,
                "end_time": end_time,
                "num_time_steps": n_steps,
                "duration_min": duration_min
            })

        lifetime_df = pd.DataFrame(lifetime_records)

        metadata_df.to_csv(self.output_filename, index=False)
        lifetime_csv = self.output_filename.replace("_metadata.csv", "_lifetimes.csv")
        lifetime_df.to_csv(lifetime_csv, index=False)

        logging.info(f"✅ Saved metadata to {self.output_filename}")
        logging.info(f"✅ Saved lifetimes to {lifetime_csv}")

        return metadata_df, lifetime_df


if __name__ == "__main__":

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    def process_convection_file(filename, output_dir):
        parser = ConvectionMetadataParser(filename, output_dir)
        return parser.generate_metadata()

    convection_regex = "/explore/nobackup/projects/pix4dcloud/Jingbo/cloudsystem_mask_2019-2020/2020*.nc"
    convection_metadata_output_dir = "/explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d/1-metadata/convection-filtered"

    convection_filenames = sorted(glob(convection_regex))
    logging.info(f"Found {len(convection_filenames)} convection filenames.")

    num_workers = multiprocessing.cpu_count()
    logging.info(f"Using {num_workers} workers for parallel processing.")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_convection_file,
                filename,
                convection_metadata_output_dir
            ): filename
            for filename in convection_filenames
        }

        for future in as_completed(futures):
            filename = futures[future]
            try:
                metadata_df, lifetime_df = future.result()
                logging.info(f"Finished processing: {filename}")
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
