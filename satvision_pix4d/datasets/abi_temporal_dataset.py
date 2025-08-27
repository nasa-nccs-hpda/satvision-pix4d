
import os
import glob
import torch
import logging
import numpy as np
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from satvision_pix4d.transforms.min_max import PerTileMinMaxNormalize
from satvision_pix4d.transforms.z_score import PerChannelStandardize

class ABITemporalDataset(Dataset):
    """
    ABITemporalDataset for .zarr files in specified directories.
    Each file must have shape [time, band, y, x].
    """

    def __init__(
        self,
        data_paths: list,
        img_size: int = 512,
        in_chans: int = 16,
        temporal_embeddings: list = None,
        transform=None,
    ):
        self.min_year = 2000
        self.img_size = img_size
        self.in_chans = in_chans
        
        # self.transform = transform
        # self.transform = PerTileMinMaxNormalize()
        # /home/jacaraba/satvision-pix4d/satvision_pix4d/transforms/min_max.py
        self.transform = PerChannelStandardize(
            mean=[127.34686819, 74.7938111, 44.78416806, 0.92638064, 8.63405386,
                2.11682601, 0.6701766, 3.00366557, 8.60859377, 14.6878577,
                46.23647303, 36.75061689, 80.34350524, 92.54769503,
                101.56140718, 85.76285184],
            std=[57.3661886, 53.84979581, 37.15785529, 2.66545547, 8.2306831,
                2.00586419, 0.24013674, 0.57227082, 1.53546711, 2.40714125,
                6.63860341, 5.02559105, 9.92560728, 10.89802816, 11.07162115,
                8.08676479]
        )

        if temporal_embeddings is None:
            self.temporal_embeddings = ["year", "month", "hour"]
        else:
            self.temporal_embeddings = temporal_embeddings

        # Find all .zarr files in given directories
        self.files = []
        for p in data_paths:
            self.files.extend(sorted(glob.glob(os.path.join(p, "*.zarr"))))

        if not self.files:
            raise RuntimeError("No .zarr files found in provided paths!")

        logging.info(f"Loaded {len(self.files)} files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        # Load dataset
        ds = xr.open_zarr(path)

        # Get Rad: (time, band, y, x)
        rad = ds["__xarray_dataarray_variable__"].values.astype(np.float32)
        rad = rad[:, :self.in_chans, :, :]
        rad = np.nan_to_num(rad, nan=0.0)
        # print(idx, np.isnan(rad).any())

        # Convert to torch tensor (T, C, H, W)
        tensor = torch.from_numpy(rad)

        # Extract timestamps: (time, band)
        t = ds["t"].values
        # t_per_time = t[:, 0]  # use first band timestamp per time

        # Vectorized datetime parsing
        pd_timestamps = pd.to_datetime(t)#_per_time)

        emb_map = {
            "year": pd_timestamps.year - self.min_year,
            "month": pd_timestamps.month - 1,
            "day": pd_timestamps.day - 1,
            "hour": pd_timestamps.hour,
            "minute": pd_timestamps.minute
        }

        timestamps = np.stack(
            [emb_map[k] for k in self.temporal_embeddings],
            axis=1
        ).astype(np.int32)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, timestamps


if __name__ == "__main__":
    # Example directory list
    train_dirs = ["/home/jacaraba/tiles_pix4d"]

    # Train dataset
    train_ds = ABITemporalDataset(
        data_paths=train_dirs,
        img_size=512,
        in_chans=14
    )

    # DataLoader example
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # First batch
    for imgs, ts in train_loader:
        print("Images:", imgs.shape)   # (B, T, C, H, W)
        print("Timestamps:", ts.shape) # (B, T, n_components)
        print(ts)
        break
