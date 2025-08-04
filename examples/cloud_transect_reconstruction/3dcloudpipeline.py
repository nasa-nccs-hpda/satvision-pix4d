"""
3D Cloud Pipeline for Training Models
Run with python3 3dcloudpipeline.py <model_name> <training_split>
e.g. python3 3dcloudpipeline.py satfull 0.8
model_name must begin with "sat" or "unet" to use SatVision or UNet models respectively

training_split is a float between 0 and 0.8 indicating the fraction of data to use for training
the last 20% of the data will be used for validation and testing

All other configurable parameters are set in the code below
"""

import sys
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import CSVLogger

from abidatamodule import AbiDataModule
from models import LightningModel

"""
CONFIGURABLE PARAMATERS
"""

BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 14
LEARNING_RATE = 1e-4
EPOCHS = 100
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else None
SAVE_EVERY_N_EPOCHS = 5
DATALOADER_WORKERS = 71

traindatapath = '/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16/'
TRAINING_SPLIT = (0, float(sys.argv[2]) if len(sys.argv) > 2 else 0.8)
valdatapath = '/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16/'
VAL_SPLIT = (0.8, 0.9)
testdatapath = '/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16/'
TEST_SPLIT = (0.9, 1.0)

checkpointpath = '/explore/nobackup/people/szhang16/checkpoints/'

"""
MAIN EXECUTION
"""

if __name__ == '__main__':
    if MODEL_NAME.startswith("sat"):
        model = LightningModel(model_name="SatVisionUNet", in_channels=IMG_CHANNELS,
                               num_classes=1, lr=LEARNING_RATE, freeze_encoder=True, final_size=(91, 40))
    elif MODEL_NAME.startswith("unet"):
        model = LightningModel(model_name="UNet", in_channels=IMG_CHANNELS,
                               num_classes=1, lr=LEARNING_RATE, freeze_encoder=False)
    else:
        raise ValueError(f"Unknown model name: {MODEL_NAME}")

    datamodule = AbiDataModule(
        train_path=traindatapath,
        val_path=valdatapath,
        test_path=testdatapath,
        train_split=TRAINING_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=DATALOADER_WORKERS
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpointpath + MODEL_NAME, save_top_k=-1, every_n_epochs=SAVE_EVERY_N_EPOCHS)
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpointpath + MODEL_NAME, save_top_k=1, monitor="val_loss", mode="min", filename="best-{epoch:02d}-{val_loss:.2f}")

    logger_tb = TensorBoardLogger(checkpointpath, name=MODEL_NAME)
    logger_csv = CSVLogger(checkpointpath, name=MODEL_NAME)

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, best_checkpoint_callback],
        logger=[logger_tb, logger_csv],
        default_root_dir=checkpointpath,
    )

    trainer.fit(model=model, datamodule=datamodule)
