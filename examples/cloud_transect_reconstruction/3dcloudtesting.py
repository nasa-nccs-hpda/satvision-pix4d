import sys
import torch
import lightning as L
import glob
import numpy as np

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

MODEL_NAMES = ["satfull", "sathalf", "satquarter", "sateighth", "unetfull", "unethalf", "unetquarter", "uneteighth"]

datapath = '/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16/'

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
    datamodule = AbiDataModule(chip_dir=datapath, batch_size=1, num_workers=DATALOADER_WORKERS)

    for model_name in MODEL_NAMES:
        best_ckpt = glob.glob("/explore/nobackup/people/szhang16/checkpoints/" + model_name + "/best-*.ckpt")
        if not best_ckpt:
            print(f"No checkpoint found for model {model_name}")
            continue
        
        if model_name.startswith("sat"):
            model = LightningModel.load_from_checkpoint(best_ckpt[0], model_name="SatVisionUNet", in_channels=IMG_CHANNELS, num_classes=1, lr=LEARNING_RATE, freeze_encoder=True, final_size=(91, 40))
        elif model_name.startswith("unet"):
            model = LightningModel.load_from_checkpoint(best_ckpt[0], model_name="UNet", in_channels=IMG_CHANNELS, num_classes=1, lr=LEARNING_RATE, freeze_encoder=False)

        savecallback = SaveTestResultsCallback(save_dir="/explore/nobackup/projects/pix4dcloud/szhang16/test_results/", model_name=model_name)

        trainer = L.Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[savecallback],
            enable_checkpointing=False,
            default_root_dir="/explore/nobackup/people/szhang16/checkpoints",
        )

        # Loop through the test datasat pytorch datalaoder
        trainer.test(model=model, datamodule=datamodule)