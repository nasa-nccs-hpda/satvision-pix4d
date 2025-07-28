import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np

import torchvision.transforms as T

from torchmetrics import JaccardIndex
import torchvision.transforms.functional as TF

from transforms import MinMaxEmissiveScaleReflectance, ConvertABIToReflectanceBT

datapath = '/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16/'

BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 14
LEARNING_RATE = 1e-4
EPOCHS = 100
DATALOADER_WORKERS = 10

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
    def __init__(self, chip_paths):
        self.chip_paths = chip_paths
        self.transform = AbiToaTransform(img_size=IMG_HEIGHT)

    def __len__(self):
        return len(self.chip_paths)

    def __getitem__(self, idx):
        chip = np.load(self.chip_paths[idx], allow_pickle=True)
        image = chip['chip'] 
        if self.transform is not None:
            image = self.transform(image)
        mask = torch.from_numpy(chip['data'].item()['Cloud_mask_binary']).unsqueeze(0).float()

        return (image, mask)

class AbiDataModule(pl.LightningDataModule):
    def __init__(self, chip_dir = datapath, batch_size = BATCH_SIZE):
        super().__init__()
        self.chip_dir = chip_dir
        self.batch_size = batch_size

    def setup(self, stage):
        total_chips = glob.glob(self.chip_dir + "*.npz")
        train_idx = int(len(total_chips) * 0.2)
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

class UNET(nn.Module):
    def __init__(self, in_channels=16, decoder_classes=64, num_classes = 1, head_dropout = 0.2, output_shape=(91, 40)):
        super().__init__()
        self.layers = [in_channels, 64, 128, 256, 512, 1024]
        
        self.double_conv_downs = nn.ModuleList(
            [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])
        
        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
             for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])
            
        self.double_conv_ups = nn.ModuleList(
        [self.__double_conv(layer, layer//2) for layer in self.layers[::-1][:-2]])
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(64, decoder_classes, kernel_size=1)

        self.prediction_head = nn.Sequential(
            nn.Conv2d(decoder_classes, num_classes,
                      kernel_size=3, stride=1, padding=1),
            nn.Dropout(head_dropout),
            nn.Upsample(size=output_shape,
                        mode='bilinear',
                        align_corners=False)
        )

        
    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv
        
    def forward(self, x):
        # down layers
        concat_layers = []
        
        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)
        
        concat_layers = concat_layers[::-1]
        
        # up layers
        for up_trans, double_conv_up, concat_layer  in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:])
            
            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)
            
        x = self.final_conv(x)

        x = self.prediction_head(x)
        
        return x

    def predict(self, x):
        x = self.forward(x)
        x = torch.sigmoid(x)
        x = (x > 0.5).float()
        return x

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
 
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = torch.sum(probs * targets)
        cardinality = torch.sum(probs + targets)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score

class LightningUNET(pl.LightningModule):
    def __init__(self, in_channels=3, classes=1, dice_weight=0.5, learning_rate=0.001):
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.learning_rate = learning_rate
        self.model = UNET(in_channels=self.in_channels, num_classes=self.classes)
        self.iou = JaccardIndex(task="binary", num_classes=2)
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.forward(inputs)

        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        loss = self.dice_weight * dice + (1 - self.dice_weight) * ce
 
        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.forward(inputs)

        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        loss = self.dice_weight * dice + (1 - self.dice_weight) * ce
 
        self.log("val_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        iou = self.iou(torch.sigmoid(logits) > 0.5, targets)
        self.log("val_iou", iou, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.forward(inputs)
        preds = torch.sigmoid(logits) > 0.5
        loss = self.iou(preds, targets)

        self.log("test_iou", loss, prog_bar=True)        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == '__main__':
    datamodule = AbiDataModule(chip_dir=datapath, batch_size=BATCH_SIZE)
    datamodule.setup(stage=None)

    model = LightningUNET(in_channels=IMG_CHANNELS, classes=1, learning_rate=LEARNING_RATE)

    datamodule = AbiDataModule(chip_dir=datapath, batch_size=BATCH_SIZE)

    checkpoint_callback = ModelCheckpoint(dirpath="/explore/nobackup/projects/pix4dcloud/szhang16/checkpoints/unetquarter", save_top_k=1, every_n_epochs=5)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model=model, datamodule=datamodule)

    trainer.test(model=model, datamodule=datamodule)
