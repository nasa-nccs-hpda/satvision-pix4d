import os
import sys
import torch
import subprocess
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

from transforms import MinMaxEmissiveScaleReflectance, ConvertABIToReflectanceBT

sys.path.append('/explore/nobackup/people/jacaraba/development/satvision-toa')
from satvision_toa.utils import load_config
from satvision_toa.models.mim import build_mim_model

datapath = '/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16/'

BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 16
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

        return {"chip": image, "mask": mask}

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

class SatVisionUNet(nn.Module):
    def __init__(self, out_channels=1, freeze_encoder=True, input_channels=14, final_size=(96, 40)):
        super().__init__()
        self.final_size = final_size

        # Load Swin encoder
        config = load_config()
        backbone = build_mim_model(config)
        self.encoder = backbone.encoder

        # Adjust input channels if needed
        if input_channels != 14:
            self.encoder.patch_embed.proj = nn.Conv2d(
                input_channels,
                self.encoder.patch_embed.proj.out_channels,
                kernel_size=self.encoder.patch_embed.proj.kernel_size,
                stride=self.encoder.patch_embed.proj.stride,
                padding=self.encoder.patch_embed.proj.padding,
                bias=False
            )

        # Load pretrained weights
        checkpoint = torch.load(config.MODEL.RESUME, weights_only=False)
        checkpoint = checkpoint['module']
        checkpoint = {k.replace('model.encoder.', ''): v for k, v in checkpoint.items() if k.startswith('model.encoder')}
        self.encoder.load_state_dict(checkpoint, strict=False)

        # Freeze encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.fusion_conv3 = nn.Conv2d(4096, 1024, 1)
        self.fusion_conv2 = nn.Conv2d(2048, 512, 1)
        self.fusion_conv1 = nn.Conv2d(1024, 256, 1)

        self.upconv4 = nn.Sequential(nn.Conv2d(4096, 2048, 3, padding=1), nn.ReLU(), nn.Conv2d(2048, 1024, 3, padding=1))
        self.upconv3 = nn.Sequential(nn.Conv2d(1024 + 1024, 1024, 3, padding=1), nn.ReLU(), nn.Conv2d(1024, 512, 3, padding=1))
        self.upconv2 = nn.Sequential(nn.Conv2d(512 + 512, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 256, 3, padding=1))
        self.upconv1 = nn.Sequential(nn.Conv2d(256 + 256, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 128, 3, padding=1))
        self.upconv0 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 64, 3, padding=1))

        self.final = nn.Conv2d(64, 64, kernel_size=1)

        self.prediction_head = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            nn.Upsample(size=self.final_size, mode='bilinear', align_corners=False)
        )

    def forward(self, x, mask=None):
        # Multi-scale encoder features
        enc_features = self.encoder.extra_features(x)  # returns [stage1, stage2, stage3, stage4]
        x = enc_features[-1]  # [B, 4096, 4, 4]

                # Upconv4: 4x4 -> 8x8
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.upconv4(x)

        # Upconv3: 8x8 -> 16x16 with fusion
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        enc_feat = F.interpolate(enc_features[2], size=x.shape[2:], mode="bilinear", align_corners=False)
        enc_feat = self.fusion_conv3(enc_feat)
        x = torch.cat([x, enc_feat], dim=1)
        x = self.upconv3(x)

        # Upconv2: 16x16 -> 32x32 with fusion
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        enc_feat = F.interpolate(enc_features[1], size=x.shape[2:], mode="bilinear", align_corners=False)
        enc_feat = self.fusion_conv2(enc_feat)
        x = torch.cat([x, enc_feat], dim=1)
        x = self.upconv2(x)

        # Upconv1: 32x32 -> 64x64 with fusion
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        enc_feat = F.interpolate(enc_features[0], size=x.shape[2:], mode="bilinear", align_corners=False)
        enc_feat = self.fusion_conv1(enc_feat)
        x = torch.cat([x, enc_feat], dim=1)
        x = self.upconv1(x)

        # Final upconv: 64x64 -> 128x128
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.upconv0(x)
        x = self.final(x)  # [B, 64, 128, 128]

        x = self.prediction_head(x)  # [B, out_channels, 91, 40]

        return x


class SatVisionUNetLightning(pl.LightningModule):
    def __init__(self, lr=1e-4, dice_weight=0.5, freeze_encoder=True, num_classes=1):
        super().__init__()
        self.save_hyperparameters()
        self.model = SatVisionUNet(out_channels=num_classes, freeze_encoder=freeze_encoder, input_channels=14, final_size=(91, 40))
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight
        self.iou = JaccardIndex(task="binary", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, stage):
        chips, masks = batch["chip"], batch["mask"]
        logits = self.forward(chips)  # (B, C, H, W)
        ce = self.ce_loss(logits, masks)
        dice = self.dice_loss(logits, masks)
        loss = self.dice_weight * dice + (1 - self.dice_weight) * ce
        self.log(f'{stage}_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        if stage == 'val' or stage == 'test':
            iou = self.iou(logits.sigmoid() > 0.5, masks)
            self.log(f'{stage}_iou', iou, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, b, i): return self._common_step(b, 'train')
    def validation_step(self, b, i): return self._common_step(b, 'val')
    def test_step(self, b, i): return self._common_step(b, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

if __name__ == '__main__':
    datamodule = AbiDataModule(chip_dir=datapath, batch_size=BATCH_SIZE)
    datamodule.setup(stage=None)

    model = SatVisionUNetLightning(
        lr=LEARNING_RATE,
        dice_weight=0.5,
        freeze_encoder=True
    )

    datamodule = AbiDataModule(chip_dir=datapath, batch_size=BATCH_SIZE)
    checkpoint_callback = ModelCheckpoint(dirpath="/explore/nobackup/projects/pix4dcloud/szhang16/checkpoints/satfull", save_top_k=1, every_n_epochs=5)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model=model, datamodule=datamodule)

    trainer.test(model=model, datamodule=datamodule)
