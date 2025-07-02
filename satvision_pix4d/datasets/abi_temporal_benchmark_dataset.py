import os
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def generate_random_date_str(idx):
    if idx == 0:
        year = random.randint(2001, 2006)
    elif idx == 1:
        year = random.randint(2007, 2015)
    elif idx == 2:
        year = random.randint(2016, 2022)
    else:
        # To support more indices, just loop over periods
        periods = [
            (2001, 2006),
            (2007, 2015),
            (2016, 2022),
        ]
        selected = periods[idx % len(periods)]
        year = random.randint(*selected)

    month = random.randint(1, 12)
    hour = random.randint(1, 24)
    date_str = f"{year:04d}-{month:02d}-28T{hour:02d}:43:59Z"
    return date_str


class ABITemporalBenchmarkDataset(Dataset):
    """
    ABITemporalToyDataset designed for Pix4D-CloudMAE and SatMAE variants
    """

    def __init__(
        self,
        data_paths: list,
        split: str,
        img_size: int = 224,
        in_chans: int = 14,
        num_timesteps: int = 7,
        transform=None,
    ):
        self.min_year = 2001
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_timesteps = num_timesteps
        self.transform = transform
        self.split = split
        self.data_paths = data_paths

        self.img_list = sorted(list(range(0, 250000)))
        self.mask_list = sorted(list(range(0, 250000)))

        random_inst = random.Random(12345)
        n_items = len(self.img_list)
        print(f"Found {n_items} possible patches to use")

        range_n_items = range(n_items)
        idxs = set(random_inst.sample(range_n_items, len(range_n_items) // 5))
        total_idxs = set(range_n_items)
        if split == "train":
            idxs = total_idxs - idxs

        print(f"> Using {len(idxs)} patches for this dataset ({split})")
        self.img_list = [self.img_list[i] for i in idxs]
        self.mask_list = [self.mask_list[i] for i in idxs]
        print(f">> {split}: {len(self.img_list)}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx, transpose=True):
        imgs = []
        timestamps = []

        for t in range(self.num_timesteps):
            # Create random image
            img = np.random.rand(
                self.img_size,
                self.img_size,
                self.in_chans,
            ).astype(np.float32)
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.from_numpy(img).permute(2,0,1)

            imgs.append(img)

            # Generate timestamp
            ts = self.parse_ts(generate_random_date_str(t))
            timestamps.append(ts)

        imgs = torch.stack(imgs, dim=0)  # (T, C, H, W)
        ts = np.stack(timestamps, axis=0)  # (T, 3)

        return imgs, ts

    def get_filenames(self, path):
        files_list = []
        for filename in sorted(os.listdir(path)):
            files_list.append(os.path.join(path, filename))
        return files_list

    def parse_ts(self, timestamp):
        year = int(timestamp[:4])
        month = int(timestamp[5:7])
        hour = int(timestamp[11:13])
        return np.array(
            [
                year - self.min_year,
                month - 1,
                hour,
            ]
        )


if __name__ == "__main__":
    # Define simple transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(512),
        ]
    )

    # Instantiate toy dataset with 7 timesteps
    train_ds = ABITemporalBenchmarkDataset(
        data_paths=[],
        split="train",
        transform=transform,
        img_size=512,
        in_chans=3,
        num_timesteps=7
    )

    # Get sample
    imgs, ts = train_ds.__getitem__(idx=12)
    print("Timestamps:\n", ts)
    print("Image tensor shape:", imgs.shape)
