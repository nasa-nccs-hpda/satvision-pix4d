import os
import xarray as xr

from pathlib import Path
from goes2go import GOES
from datetime import datetime, timedelta


class ABI_L1B_Reader:
    def __init__(
                self,
                product: str = "ABI-L1b-RadF",
                satellite: str = "goes16",
                domain: str = "F",
                bands: list = list(range(1, 17)),
                save_dir: str = "abi_data",
                memory_only: bool = False
            ):
        """
        Initialize the ABI L1b reader.

        Args:
            product (str): GOES product name (e.g., 'ABI-L1b-RadF')
            satellite (str): 'goes16' or 'goes17'
            domain (str): 'C' or 'F'
            channels (list): List of channel strings like ['C01', 'C02', ...]
            save_dir (str): Directory to store downloaded
                files (if not memory_only)
            memory_only (bool): If True, load files into
                memory as xarray.Dataset
        """
        self.product = product
        self.satellite = satellite
        self.domain = domain
        self.bands = bands
        self.memory_only = memory_only
        self.save_dir = Path(save_dir)

        # if we need to store the data in memory
        if not self.memory_only:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def download(
                self,
                start_time: datetime = None,
                end_time: datetime = None,
                interval_minutes: int = 10
            ):
        """
        Download or load GOES ABI L1b data for a given time range and channels.

        Args:
            start_time (datetime): Start datetime
            end_time (datetime): End datetime
            interval_minutes (int): Interval in minutes for granule selection

        Returns:
            List of file paths or xarray.Dataset objects,
                depending on memory_only
        """
        goes = GOES(
            product=self.product,
            satellite=self.satellite,
            domain=self.domain,
            bands=self.bands
        )
        print(goes)

        # Download and read the latest data as an xarray Dataset
        # ds = goes.latest()
        ds = goes.nearesttime('2022-01-01')
        print(ds)

        """
        current_time = start_time
        data_outputs = []

        while current_time <= end_time:
            for channel in self.channels:
                try:
                    file_path = goes.download(
                        date=current_time,
                        channel=channel,
                        save_dir=None if self.memory_only else self.save_dir,
                        overwrite=False,
                        memory=self.memory_only,
                    )
                    if self.memory_only:
                        ds = xr.open_dataset(file_path)
                        data_outputs.append(ds)
                        print(f"[✓] Loaded {channel} for {current_time} into memory.")
                    else:
                        data_outputs.append(file_path)
                        print(
                            f"[✓] Downloaded {channel} for {current_time} → {file_path}")
                except Exception as e:
                    print(f"[!] Failed to retrieve {channel} for {current_time}: {e}")
            current_time += timedelta(minutes=interval_minutes)

        return data_outputs
        """

    def list_downloaded_files(self):
        """
        List all downloaded files in the save directory (if not memory_only).
        """
        if self.memory_only:
            print("[!] Cannot list files when memory_only=True.")
            return []
        return list(self.save_dir.rglob("*.nc"))


if __name__ == '__main__':

    # testing the ABI reader
    abi_l1_reader = ABI_L1B_Reader()

    # read data
    abi_l1_reader.download()
