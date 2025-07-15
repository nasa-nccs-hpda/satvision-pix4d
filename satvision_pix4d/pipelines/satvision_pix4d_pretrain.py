import os
import logging
import torch
import torchmetrics
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from satvision_pix4d.models.encoders.mae import build_satmae_model
from satvision_pix4d.optimizers.build import build_optimizer

# -----------------------------------------------------------------------------
# SatVisionPix4DSatMAEPretrain
# -----------------------------------------------------------------------------
class SatVisionPix4DSatMAEPretrain(pl.LightningModule):
    def __init__(self, config):
        super(SatVisionPix4DSatMAEPretrain, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.config = config

        self.model = build_satmae_model(self.config)
        if self.config.MODEL.PRETRAINED:
            self.load_checkpoint()

        self.batch_size = config.DATA.BATCH_SIZE
        self.num_workers = config.DATA.NUM_WORKERS
        self.img_size = config.DATA.IMG_SIZE
        self.pin_memory = config.DATA.PIN_MEMORY

        # Training Metrics
        self.train_loss_avg = torchmetrics.MeanMetric()
        self.train_psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

        # Validation Metrics
        self.val_loss_avg = torchmetrics.MeanMetric()
        self.val_psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

    def load_checkpoint(self):
        logging.info(f'Loading checkpoint from {self.config.MODEL.PRETRAINED}')
        checkpoint = torch.load(self.config.MODEL.PRETRAINED)
        self.model.load_state_dict(checkpoint['module'])
        logging.info('Successfully applied checkpoint')

    def forward(self, samples, timestamps):
        return self.model(samples, timestamps, mask_ratio=self.config.DATA.MASK_RATIO)

    def training_step(self, batch, batch_idx):
        samples, timestamps = batch
        samples = samples.to(self.device, non_blocking=True)
        timestamps = timestamps.to(self.device, non_blocking=True)

        loss, pred, mask = self.forward(samples, timestamps)
        self.train_loss_avg.update(loss)

        # Compute reconstruction metrics
        B, T, C, H, W = samples.shape
        pred_imgs = self.model.unpatchify(pred, T, H, W)
        pred_imgs = torch.clamp(pred_imgs, 0, 1)
        target_imgs = samples

        psnr_val = self.train_psnr(pred_imgs[:, 0], target_imgs[:, 0])
        ssim_val = self.train_ssim(pred_imgs[:, 0], target_imgs[:, 0])

        # Log metrics
        self.log("train_loss", self.train_loss_avg.compute(), prog_bar=True, batch_size=self.batch_size)
        self.log("train_psnr", psnr_val, prog_bar=True, batch_size=self.batch_size)
        self.log("train_ssim", ssim_val, prog_bar=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        samples, timestamps = batch
        samples = samples.to(self.device, non_blocking=True)
        timestamps = timestamps.to(self.device, non_blocking=True)

        loss, pred, mask = self.forward(samples, timestamps)
        self.val_loss_avg.update(loss)

        # Compute reconstruction metrics
        B, T, C, H, W = samples.shape
        pred_imgs = self.model.unpatchify(pred, T, H, W)
        pred_imgs = torch.clamp(pred_imgs, 0, 1)
        target_imgs = samples

        psnr_val = self.val_psnr(pred_imgs[:, 0], target_imgs[:, 0])
        ssim_val = self.val_ssim(pred_imgs[:, 0], target_imgs[:, 0])

        # Log metrics
        self.log("val_loss", self.val_loss_avg.compute(), prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("val_psnr", psnr_val, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("val_ssim", ssim_val, prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        optimizer = build_optimizer(self.config, self.model, is_pretrain=True)

        # Compute total steps
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.config.TRAIN.WARMUP_EPOCHS * (total_steps / self.config.TRAIN.EPOCHS))
        cosine_steps = total_steps - warmup_steps

        # Warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        # Cosine scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=self.config.TRAIN.MIN_LR
        )

        # Combine schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def on_train_epoch_start(self):
        self.train_loss_avg.reset()
        self.train_psnr.reset()
        self.train_ssim.reset()

    def on_validation_epoch_start(self):
        self.val_loss_avg.reset()
        self.val_psnr.reset()
        self.val_ssim.reset()
