"""
ABI Data Module for loading and transforming ABI chips.
"""

from transforms import MinMaxEmissiveScaleReflectance
from transforms import ConvertABIToReflectanceBT
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import lightning as L

translation = [1, 2, 0, 4, 5, 6, 3, 8, 9, 10, 11, 13, 14, 15]


class AbiToaTransform:
    """
    torchvision transform which transforms the input imagery into
    addition to generating a MiM mask
    """

    def __init__(self, img_size):

        self.transform_img = \
            T.Compose([
                T.Lambda(lambda img: img[:, :, translation]),
                ConvertABIToReflectanceBT(),
                MinMaxEmissiveScaleReflectance(),
                T.ToTensor(),
                T.Resize((img_size, img_size), antialias=True),
            ])

    def __call__(self, img):

        img = self.transform_img(img)

        return img

# ABI Chip Loader


class AbiChipDataset(Dataset):
    def __init__(self, chip_paths, img_size=128):
        self.chip_paths = chip_paths
        self.transform = AbiToaTransform(img_size=img_size)

    def __len__(self):
        return len(self.chip_paths)

    def __getitem__(self, idx):
        chip = np.load(self.chip_paths[idx], allow_pickle=True)
        image = chip['chip']
        if self.transform is not None:
            image = self.transform(image)
        mask = torch.from_numpy(
            chip['data'].item()['Cloud_mask_binary']).unsqueeze(0).float()

        return {"chip": image, "mask": mask}


"""
Pass the paths to the training, validation, and test datasets and the splits for each dataset.
"""


class AbiDataModule(L.LightningDataModule):
    def __init__(self, train_path, val_path, test_path, train_split=(0, 0.8), val_split=(0.8, 0.9), test_split=(0.9, 1.0), batch_size=32, training_split=0.8, num_workers=1):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.chip_dir = train_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.training_split = training_split
        self.num_workers = num_workers

    def setup(self, stage):
        train_chips = glob.glob(self.train_path + "*.npz")
        train_start = int(len(train_chips) * self.train_split[0])
        train_end = int(len(train_chips) * self.train_split[1])

        self.train_dataset = AbiChipDataset(train_chips[train_start:train_end])

        val_chips = glob.glob(self.val_path + "*.npz")
        val_start = int(len(val_chips) * self.val_split[0])
        val_end = int(len(val_chips) * self.val_split[1])

        self.val_dataset = AbiChipDataset(val_chips[val_start:val_end])

        test_chips = glob.glob(self.test_path + "*.npz")
        test_start = int(len(test_chips) * self.test_split[0])
        test_end = int(len(test_chips) * self.test_split[1])

        self.test_dataset = AbiChipDataset(test_chips[test_start:test_end])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
