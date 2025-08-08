"""
3D Cloud Pipeline for Testing Models

NEEDS TO BE REFACTORED
"""

import sys
import torch
import lightning as L
import glob
import numpy as np

from abidatamodule import AbiDataModule
from models import LightningModel

"""
CONFIGURABLE PARAMATERS
"""

# MODEL_NAMES = ["satfull", "sathalf", "satquarter", "sateighth", "unetfull", "unethalf", "unetquarter", "uneteighth"]
MODEL_NAMES = ["unetfull", "satfull"]

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
TRAINING_SPLIT = (0, 0.8)
valdatapath = '/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16/'
VAL_SPLIT = (0.8, 0.9)
testdatapath = '/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16/'
TEST_SPLIT = (0.9, 1.0)

checkpointpath = '/explore/nobackup/people/szhang16/checkpoints/'
resultspath = '/explore/nobackup/projects/pix4dcloud/szhang16/test_results/'

class SaveTestResultsCallback(L.Callback):
    def __init__(self, save_dir, model_name):
        super().__init__()
        self.save_dir = save_dir
        self.test_results = []
        self.model_name = model_name

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.test_results.append(outputs["iou"].detach().cpu().numpy())

    def on_test_end(self, trainer, pl_module):
        self.test_results = np.array(self.test_results)
        np.save(self.save_dir + f"{self.model_name}.npy", self.test_results)


if __name__ == '__main__':

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

    for model_name in MODEL_NAMES:
        best_ckpt = glob.glob(
            checkpointpath + model_name + "/best-*.ckpt")
        if not best_ckpt:
            print(f"No checkpoint found for model {model_name}")
            continue

        if model_name.startswith("sat"):
            model = LightningModel.load_from_checkpoint(
                best_ckpt[0], model_name="SatVisionUNet", in_channels=IMG_CHANNELS, num_classes=1, lr=LEARNING_RATE, freeze_encoder=True, final_size=(91, 40))
        elif model_name.startswith("unet"):
            model = LightningModel.load_from_checkpoint(
                best_ckpt[0], model_name="UNet", in_channels=IMG_CHANNELS, num_classes=1, lr=LEARNING_RATE, freeze_encoder=False)

        savecallback = SaveTestResultsCallback(
            save_dir=resultspath, model_name=model_name)

        trainer = L.Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[],
            enable_checkpointing=False,
            default_root_dir=checkpointpath,
        )

        trainer.test(model=model, datamodule=datamodule)
