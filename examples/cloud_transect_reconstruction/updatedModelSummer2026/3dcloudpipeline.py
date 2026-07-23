import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from abidatamodule import AbiDataModule
from models import LightningModel

"""
CONFIGURABLE PARAMETERS
"""
# Note: Batch size should be much smaller for 3D CNNs to avoid Out of Memory errors.
BATCH_SIZE = 4 
LEARNING_RATE = 1e-4
EPOCHS = 100
SAVE_EVERY_N_EPOCHS = 5
DATALOADER_WORKERS = 8

# Update this path to where your new 2026 chips are located
datapath = '/explore/nobackup/projects/pix4dcloud/aliewehr/chipTests/chips/allChips'
TRAINING_SPLIT = (0, 0.8)
VAL_SPLIT = (0.8, 0.9)
TEST_SPLIT = (0.9, 1.0)

checkpointpath = './checkpoints/'

"""
MAIN EXECUTION
"""
if __name__ == '__main__':
    # Initialize the new 3D model
    model = LightningModel(lr=LEARNING_RATE)

    # Initialize the updated Data Module
    datamodule = AbiDataModule(
        data_path=datapath,
        train_split=TRAINING_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=DATALOADER_WORKERS
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpointpath + "unet3d_baseline", save_top_k=-1, every_n_epochs=SAVE_EVERY_N_EPOCHS)
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpointpath + "unet3d_baseline", save_top_k=1, monitor="val_loss", mode="min", filename="best-{epoch:02d}-{val_loss:.2f}")

    logger_tb = TensorBoardLogger(checkpointpath, name="unet3d_baseline")
    logger_csv = CSVLogger(checkpointpath, name="unet3d_baseline")

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, best_checkpoint_callback],
        logger=[logger_tb, logger_csv],
        default_root_dir=checkpointpath,
    )

    trainer.fit(model=model, datamodule=datamodule)
