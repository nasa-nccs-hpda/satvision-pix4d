import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from model import LightningUNET
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import random

datapath = '/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16/'
BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 16
LEARNING_RATE = 0.001
EPOCHS = 1000
DATALOADER_WORKERS = 10

channel_mins = torch.tensor([26.8503,  0.1685, -8.2685, -1.4102, -1.1436, -0.1486,  0.0375,  0.2940,
         0.7091,  1.0280,  2.7341, 10.9818,  7.9130, 10.6544, 14.0237, 17.8245])
channel_maxs = torch.tensor([ 804.0361, 4095.0000,  373.1670,   79.2232,   94.8480,   29.7895,
          19.2759,    7.1074,   17.2302,   27.8833,  103.7081,   78.4874,
         159.9617,  175.4634,  178.8900,  132.4386])

# ABI Chip Loader
class AbiChipDataset(Dataset):
    def __init__(self, chip_paths):
        self.chip_paths = chip_paths

    def __len__(self):
        return len(self.chip_paths)

    def __getitem__(self, idx):
        chip = np.load(self.chip_paths[idx], allow_pickle=True)
        image = torch.from_numpy(chip['chip'])
        image = (image - channel_mins) / (channel_maxs - channel_mins)
        image = image.permute(2, 0, 1).float()
        mask = torch.from_numpy(chip['data'].item()['Cloud_mask_binary']).unsqueeze(0).float()

        return (image, mask)

class AbiDataModule(pl.LightningDataModule):
    def __init__(self, chip_dir = datapath, batch_size = BATCH_SIZE):
        super().__init__()
        self.chip_dir = chip_dir
        self.batch_size = batch_size

    def setup(self, stage):
        total_chips = glob.glob(self.chip_dir + "*.npz")
        train_idx = int(len(total_chips) * 0.8)
        val_idx = int(len(total_chips) * 0.9)
        self.train_dataset = AbiChipDataset(total_chips[:train_idx])
        self.val_dataset = AbiChipDataset(total_chips[train_idx:val_idx])
        self.test_dataset = AbiChipDataset(total_chips[val_idx:])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=DATALOADER_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=DATALOADER_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=DATALOADER_WORKERS)

if __name__ == '__main__':
    pl.seed_everything(27, workers=True)

    unet = LightningUNET.load_from_checkpoint("/explore/nobackup/projects/pix4dcloud/szhang16/checkpoints/lightning_logs/version_39084251/checkpoints/epoch=25-step=3900.ckpt", in_channels=IMG_CHANNELS, classes=1)

    lr_scheduler = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min")

    datamodule = AbiDataModule(datapath, BATCH_SIZE)

    trainer = pl.Trainer(
        default_root_dir="/explore/nobackup/projects/pix4dcloud/szhang16/checkpoints",
        deterministic=True,
        accelerator="auto",
        max_epochs=EPOCHS,
        callbacks=[lr_scheduler, early_stopping]
    )

    trainer.test(model=unet, datamodule=datamodule)
