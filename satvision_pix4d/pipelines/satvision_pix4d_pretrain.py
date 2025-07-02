import torch
import logging
import torchmetrics
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from satvision_pix4d.models.encoders.mae import build_satmae_model
from satvision_pix4d.optimizers.build import build_optimizer


# -----------------------------------------------------------------------------
# SatVisionPix4DSatMAEPretrain
# -----------------------------------------------------------------------------
class SatVisionPix4DSatMAEPretrain(pl.LightningModule):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config):
        super(SatVisionPix4DSatMAEPretrain, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.config = config

        self.model = build_satmae_model(self.config)
        if self.config.MODEL.PRETRAINED:
            self.load_checkpoint()

        #self.transform = MimTransform(self.config)
        self.batch_size = config.DATA.BATCH_SIZE
        self.num_workers = config.DATA.NUM_WORKERS
        self.img_size = config.DATA.IMG_SIZE
        self.train_data_paths = config.DATA.DATA_PATHS
        self.train_data_length = config.DATA.LENGTH
        self.pin_memory = config.DATA.PIN_MEMORY

        # Metrics
        self.train_loss_avg = torchmetrics.MeanMetric()
        self.train_psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

    # -------------------------------------------------------------------------
    # load_checkpoint
    # -------------------------------------------------------------------------
    def load_checkpoint(self):
        logging.info(f'Loading checkpoint from {self.config.MODEL.PRETRAINED}')
        checkpoint = torch.load(self.config.MODEL.PRETRAINED)
        self.model.load_state_dict(checkpoint['module'])
        logging.info('Successfully applied checkpoint')

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def forward(self, samples, timestamps):
        return self.model(
            samples, timestamps, mask_ratio=self.config.DATA.MASK_RATIO)

    # -------------------------------------------------------------------------
    # training_step
    # -------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):

        # Unpack your batch exactly as original training loop
        samples, timestamps = batch

        # Move tensors to device (Lightning usually does this, but explicitly ok)
        samples = samples.to(self.device, non_blocking=True)
        timestamps = timestamps.to(self.device, non_blocking=True)

        # Mixed precision context is handled automatically by Lightning
        loss, pred, mask = self.forward(samples, timestamps)

        # .item() if you want scalar logging
        self.train_loss_avg.update(loss)

        # Unpatchify reconstruction for metrics and visualization
        B, T, C, H, W = samples.shape
        pred_imgs = self.model.unpatchify(pred, T, H, W)
        pred_imgs = torch.clamp(pred_imgs, 0, 1)

        # Normalize ground truth
        target_imgs = samples

        # Compute metrics over first timestep only (or all if you prefer)
        psnr_val = self.train_psnr(pred_imgs[:,0], target_imgs[:,0])
        ssim_val = self.train_ssim(pred_imgs[:,0], target_imgs[:,0])

        self.log("train_loss", self.train_loss_avg.compute(), prog_bar=True, batch_size=self.batch_size)
        self.log("train_psnr", psnr_val, prog_bar=True, batch_size=self.batch_size)
        self.log("train_ssim", ssim_val, prog_bar=True, batch_size=self.batch_size)

        # Log example images once per epoch
        if batch_idx == 0:
            # Log first image in batch
            input_img = target_imgs[0, 0]  # (C,H,W)
            recon_img = pred_imgs[0, 0]

            # Convert to uint8
            grid = torch.cat([input_img, recon_img], dim=2)  # side by side

            # Works with MLflow, TensorBoard, WandB
            self.logger.experiment.add_image(
                "Input_Reconstruction",
                grid,
                self.global_step,
                dataformats="CHW"
            )

        return loss

    # -------------------------------------------------------------------------
    # configure_optimizers
    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = build_optimizer(
            self.config, self.model, is_pretrain=True)
        return optimizer

    # -------------------------------------------------------------------------
    # on_train_epoch_start
    # -------------------------------------------------------------------------
    def on_train_epoch_start(self):
        self.train_loss_avg.reset()
        self.train_psnr.reset()
        self.train_ssim.reset()

    # -------------------------------------------------------------------------
    # train_dataloader
    # -------------------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=None,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )
