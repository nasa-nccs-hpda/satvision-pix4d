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

        # ------------------------------
        # Build model
        # ------------------------------
        self.model = build_satmae_model(self.config)

        # ------------------------------
        # Training/data params
        # ------------------------------
        self.batch_size = config.DATA.BATCH_SIZE
        self.num_workers = config.DATA.NUM_WORKERS
        self.img_size = config.DATA.IMG_SIZE
        self.pin_memory = config.DATA.PIN_MEMORY
        self.mask_ratio  = config.DATA.MASK_RATIO

        # ------------------------------
        # Per-channel stats (buffers for broadcast over (B,T,C,H,W))
        # ------------------------------
        mean = torch.as_tensor(self.config.DATA.MEAN, dtype=torch.float32)  # (C,)
        std  = torch.as_tensor(self.config.DATA.STD,  dtype=torch.float32)  # (C,)
        self.register_buffer("ch_mean", mean.view(1, 1, -1, 1, 1), persistent=False)
        self.register_buffer("ch_std",  std.view(1, 1, -1, 1, 1),  persistent=False)

        # Optional global min/max → scalar data_range for TorchMetrics
        data_min = getattr(self.config.DATA, "MIN", None)
        data_max = getattr(self.config.DATA, "MAX", None)
        if data_min is not None and data_max is not None:
            mn = torch.as_tensor(data_min, dtype=torch.float32).min().item()
            mx = torch.as_tensor(data_max, dtype=torch.float32).max().item()
            metric_data_range = max(mx - mn, 1e-6)
        else:
            # Fallback: rough coverage using +/-4σ around mean across channels
            approx_min = (mean - 4 * std).min().item()
            approx_max = (mean + 4 * std).max().item()
            metric_data_range = max(approx_max - approx_min, 1e-6)
        self.metric_data_range = float(metric_data_range)

        # ------------------------------
        # Metrics (computed in RAW scale)
        # ------------------------------
        self.train_loss_avg = torchmetrics.MeanMetric()
        self.val_loss_avg = torchmetrics.MeanMetric()

        self.train_psnr = torchmetrics.image.PeakSignalNoiseRatio(
            data_range=self.metric_data_range
        )
        self.train_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(
            data_range=self.metric_data_range
        )
        self.val_psnr = torchmetrics.image.PeakSignalNoiseRatio(
            data_range=self.metric_data_range
        )
        self.val_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(
            data_range=self.metric_data_range
        )

        # Optional: masked-only variants (we'll compute 
        # manually if we can build a pixel mask)
        # We'll log these as scalars if available.
        self.log_masked_metrics = True

        # Training Metrics
        #self.train_loss_avg = torchmetrics.MeanMetric()
        #self.train_psnr = torchmetrics.image.PeakSignalNoiseRatio(
        #    data_range=1.0)
        #self.train_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(
        #    data_range=1.0)

        # Validation Metrics
        #self.val_loss_avg = torchmetrics.MeanMetric()
        #self.val_psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        #self.val_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(
        #    data_range=1.0)

    # ------------------------------
    # Helpers
    # ------------------------------
    @torch.no_grad()
    def _inverse_standardize(self, x):
        """x: (B,T,C,H,W) in z-score → returns raw physical scale."""
        return x * self.ch_std + self.ch_mean

    def _avg_over_time(self, metric_fn, pred_raw, tgt_raw, pixel_mask=None):
        """
        Compute metric averaged over all timesteps T.
        If pixel_mask is given, apply masked-only metric manually (PSNR via MSE).
        """
        B, T, C, H, W = pred_raw.shape

        # If we have to do masked-only PSNR/SSIM, we implement custom masked PSNR;
        # SSIM with masks isn't directly supported; we skip masked SSIM unless full image.
        if pixel_mask is not None and metric_fn is self.train_psnr or metric_fn is self.val_psnr:
            psnr_vals = []
            # scalar data_range
            dr2 = self.metric_data_range ** 2
            for t in range(T):
                # masked MSE over all channels
                mask = pixel_mask  # (B,1,H,W)
                num = mask.sum().clamp_min(1)
                mse = (((pred_raw[:, t] - tgt_raw[:, t]) ** 2) * mask).sum() / num
                psnr_t = 10.0 * torch.log10(dr2 / mse.clamp_min(1e-12))
                psnr_vals.append(psnr_t)
            return torch.stack(psnr_vals).mean()

        # Otherwise, use TorchMetrics on full frame; average over T
        vals = []
        for t in range(T):
            vals.append(metric_fn(pred_raw[:, t], tgt_raw[:, t]))
        return torch.stack(vals).mean()

    def _maybe_build_pixel_mask(self, mask_tokens, T, H, W):
        """
        Attempt to upsample the MAE patch mask to a pixel mask (B,1,H,W).
        This assumes:
          - mask_tokens: (B, N_patches) with 1 for masked, 0 for visible.
          - self.model has an 'unpatchify' that maps token grids back to (B,T,C,H,W).
        We build a dummy token map with 1s at masked tokens and 0 elsewhere for a single channel,
        unpatchify it, and then threshold to {0,1}. If anything fails, return None.
        """
        try:
            B, Np = mask_tokens.shape
            # Build a dummy token tensor shaped like the decoder target tokens.
            # We need tokens in the same shape expected by unpatchify's input.
            # Commonly pred (B, N_tokens, patch_dim). We'll mimic pred with channels=1.
            device = mask_tokens.device
            ones = torch.ones((B, Np, 1), device=device, dtype=torch.float32)
            zeros = torch.zeros((B, Np, 1), device=device, dtype=torch.float32)
            token_mask = torch.where(mask_tokens.unsqueeze(-1) > 0.5, ones, zeros)

            # Unpatchify expects (B, N_tokens, patch_dim) and returns (B,T,C,H,W).
            # We'll get a (B,T,C=1,H,W) that marks masked regions ≈1.
            pixel_mask_raw = self.model.unpatchify(token_mask, T, H, W)  # (B,T,1,H,W) if implemented that way
            # If unpatchify returns (B,T,1,H,W), we collapse time by OR (any frame masked → masked)
            if pixel_mask_raw.ndim == 5:
                pixel_mask_raw = pixel_mask_raw.max(dim=1, keepdim=False).values  # (B,1,H,W)
            # Binarize
            pixel_mask = (pixel_mask_raw > 0.5).float()
            # Ensure shape (B,1,H,W)
            if pixel_mask.ndim == 3:
                pixel_mask = pixel_mask.unsqueeze(1)
            return pixel_mask
        except Exception:
            # If the model's unpatchify cannot handle this path, silently fall back.
            return None

    @classmethod
    def load_checkpoint(cls, ckpt_path, config):
        """
        Load model from either a Lightning .ckpt 
        file or a DeepSpeed .pt checkpoint.
        """
        if ckpt_path.endswith(".pt") or "mp_rank" in ckpt_path:
            logging.info(f"Loading DeepSpeed checkpoint: {ckpt_path}")
            model = cls(config)

            checkpoint = torch.load(
                ckpt_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("module", checkpoint)

            # Strip "model." prefix if present
            cleaned_state_dict = {
                k.replace("model.", "", 1) if k.startswith("model.") else k: v
                for k, v in state_dict.items()
            }

            missing_keys, unexpected_keys = model.model.load_state_dict(
                cleaned_state_dict, strict=False
            )

            logging.info(
                f"Loaded DeepSpeed weights with {len(missing_keys)} missing and {len(unexpected_keys)} unexpected keys.")
            return model

        else:
            logging.info(f"Loading Lightning checkpoint: {ckpt_path}")
            return cls.load_from_checkpoint(ckpt_path, config=config)


    def forward(self, samples, timestamps):
        return self.model(
            samples, timestamps, mask_ratio=self.mask_ratio)

    def training_stepxxx(self, batch, batch_idx):
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

    def training_step(self, batch, batch_idx):
        samples, timestamps = batch
        samples = samples.to(self.device, non_blocking=True)   # (B,T,C,H,W), z-scored
        timestamps = timestamps.to(self.device, non_blocking=True)

        loss, pred, mask_tokens = self.forward(samples, timestamps)
        self.train_loss_avg.update(loss)

        # Reconstruct prediction in z-score space, then inverse-standardize
        B, T, C, H, W = samples.shape
        pred_imgs = self.model.unpatchify(pred, T, H, W)       # (B,T,C,H,W), z-scored
        pred_raw  = self._inverse_standardize(pred_imgs)
        tgt_raw   = self._inverse_standardize(samples)

        # Try to construct a pixel mask for masked-only metrics
        pixel_mask = None
        if self.log_masked_metrics and mask_tokens is not None:
            pixel_mask = self._maybe_build_pixel_mask(mask_tokens, T, H, W)

        # Full-image metrics (averaged over T)
        train_psnr_full = self._avg_over_time(self.train_psnr, pred_raw, tgt_raw, pixel_mask=None)
        train_ssim_full = self._avg_over_time(self.train_ssim, pred_raw, tgt_raw, pixel_mask=None)

        self.log("train_loss", self.train_loss_avg.compute(), prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("train_psnr", train_psnr_full,               prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("train_ssim", train_ssim_full,               prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        # Optional masked-only PSNR (if we could build pixel_mask)
        if pixel_mask is not None:
            train_psnr_masked = self._avg_over_time(self.train_psnr, pred_raw, tgt_raw, pixel_mask=pixel_mask)
            self.log("train_psnr_masked", train_psnr_masked, prog_bar=False, sync_dist=True, batch_size=self.batch_size)

        return loss

    def validation_stepxxx(self, batch, batch_idx):
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

    def validation_step(self, batch, batch_idx):
        samples, timestamps = batch
        samples = samples.to(self.device, non_blocking=True)   # (B,T,C,H,W), z-scored
        timestamps = timestamps.to(self.device, non_blocking=True)

        loss, pred, mask_tokens = self.forward(samples, timestamps)
        self.val_loss_avg.update(loss)

        B, T, C, H, W = samples.shape
        pred_imgs = self.model.unpatchify(pred, T, H, W)       # z-scored
        pred_raw  = self._inverse_standardize(pred_imgs)
        tgt_raw   = self._inverse_standardize(samples)

        pixel_mask = None
        if self.log_masked_metrics and mask_tokens is not None:
            pixel_mask = self._maybe_build_pixel_mask(mask_tokens, T, H, W)

        val_psnr_full = self._avg_over_time(self.val_psnr, pred_raw, tgt_raw, pixel_mask=None)
        val_ssim_full = self._avg_over_time(self.val_ssim, pred_raw, tgt_raw, pixel_mask=None)

        self.log("val_loss", self.val_loss_avg.compute(), prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("val_psnr", val_psnr_full,                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("val_ssim", val_ssim_full,                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        if pixel_mask is not None:
            val_psnr_masked = self._avg_over_time(self.val_psnr, pred_raw, tgt_raw, pixel_mask=pixel_mask)
            self.log("val_psnr_masked", val_psnr_masked, prog_bar=False, sync_dist=True, batch_size=self.batch_size)

        return loss

    def configure_optimizersxxx(self):

        optimizer = build_optimizer(self.config, self.model, is_pretrain=True)

        # Compute total steps
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(
            self.config.TRAIN.WARMUP_EPOCHS * (total_steps / self.config.TRAIN.EPOCHS))
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
    def configure_optimizers(self):
        optimizer = build_optimizer(self.config, self.model, is_pretrain=True)

        # Robust scheduler when dataset is tiny
        total_steps  = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = int(self.config.TRAIN.WARMUP_EPOCHS * (total_steps / self.config.TRAIN.EPOCHS))
        warmup_steps = min(warmup_steps, max(0, total_steps - 1))
        cosine_steps = max(1, total_steps - warmup_steps)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=self.config.TRAIN.MIN_LR
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
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
