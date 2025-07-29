import os
import sys
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning.pytorch.loggers import TensorBoardLogger

from abidatamodule import AbiDataModule
from models import LightningModel

BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 14
LEARNING_RATE = 1e-4
TRAINING_SPLIT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
EPOCHS = 100
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else None
DATALOADER_WORKERS = 71

datapath = '/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16/'

if __name__ == '__main__':
    checkpointpath = '/explore/nobackup/people/szhang16/checkpoints/' + MODEL_NAME

    if MODEL_NAME.startswith("sat"):
        model = LightningModel(model_name="SatVisionUNet", in_channels=IMG_CHANNELS, num_classes=1, lr=LEARNING_RATE, freeze_encoder=True, final_size=(91, 40))
    elif MODEL_NAME.startswith("unet"):
        model = LightningModel(model_name="UNet", in_channels=IMG_CHANNELS, num_classes=1, lr=LEARNING_RATE, freeze_encoder=False)
    else:
        raise ValueError(f"Unknown model name: {MODEL_NAME}")

    datamodule = AbiDataModule(chip_dir=datapath, batch_size=BATCH_SIZE, num_workers=DATALOADER_WORKERS)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpointpath, save_top_k=-1, every_n_epochs=2)
    best_checkpoint_callback = ModelCheckpoint(dirpath=checkpointpath, save_top_k=1, monitor="val_loss", mode="min", filename="best-{epoch:02d}-{val_loss:.2f}")

    logger = TensorBoardLogger("/explore/nobackup/people/szhang16/checkpoints", name=MODEL_NAME)

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, best_checkpoint_callback],
        logger=logger,
        default_root_dir="/explore/nobackup/people/szhang16/checkpoints"
    )

    trainer.fit(model=model, datamodule=datamodule)

    trainer.test(model=model, datamodule=datamodule)
