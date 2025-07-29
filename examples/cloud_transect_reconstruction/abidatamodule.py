from transforms import MinMaxEmissiveScaleReflectance, ConvertABIToReflectanceBT
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
        mask = torch.from_numpy(chip['data'].item()['Cloud_mask_binary']).unsqueeze(0).float()

        return {"chip": image, "mask": mask}

class AbiDataModule(L.LightningDataModule):
    def __init__(self, chip_dir = "", batch_size = 32, training_split=0.8, num_workers=1):
        super().__init__()
        self.chip_dir = chip_dir
        self.batch_size = batch_size
        self.training_split = training_split
        self.num_workers = num_workers

    def setup(self, stage):
        total_chips = glob.glob(self.chip_dir + "*.npz")
        train_idx_cut = int(len(total_chips) * self.training_split)
        train_idx = int(len(total_chips) * 0.8)
        val_idx = int(len(total_chips) * 0.9)
        self.train_dataset = AbiChipDataset(total_chips[:train_idx_cut])
        self.val_dataset = AbiChipDataset(total_chips[train_idx:val_idx])
        self.test_dataset = AbiChipDataset(total_chips[val_idx:])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=self.num_workers)