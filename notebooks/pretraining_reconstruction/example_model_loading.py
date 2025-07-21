import sys

sys.path.append('/explore/nobackup/people/jacaraba/development/satvision-pix4d')

import os
import torch
import logging
import argparse

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from satvision_pix4d.configs.config import _C, _update_config_from_file
from satvision_pix4d.utils import get_strategy, get_distributed_train_batches
from satvision_pix4d.pipelines import PIPELINES, get_available_pipelines
from satvision_pix4d.datamodules import DATAMODULES, get_available_datamodules
from satvision_pix4d.models.encoders.mae import build_satmae_model
from satvision_pix4d.datasets.abi_temporal_benchmark_dataset import ABITemporalBenchmarkDataset

model_filename = '/explore/nobackup/projects/pix4dcloud/jacaraba/model_development/satmae/' + \
    'satmae_satvision_pix4d_pretrain-dev/satmae_satvision_pix4d_pretrain-dev/epoch-epoch=40.ckpt/checkpoint/mp_rank_00_model_states.pt'

config_filename = '/explore/nobackup/people/jacaraba/development/satvision-pix4d/tests/configs/test_satmae_dev.yaml'

config = _C.clone()
_update_config_from_file(config, config_filename)
print("Loaded configuration file.")

# Add checkpoint (MODEL.PRETRAINED), 
# validation tile dir (DATA.DATA_PATHS),
# and output dir (OUTPUT) to config file
config.defrost()
config.MODEL.PRETRAINED = model_filename
config.OUTPUT = '.'
config.freeze()
print("Updated configuration file.")

# Get the proper pipeline
available_pipelines = get_available_pipelines()
print("Available pipelines:", available_pipelines)

pipeline = PIPELINES[config.PIPELINE]
print(f'Using {pipeline}')

ptlPipeline = pipeline(config)

# Resume from checkpoint
print(f'Attempting to resume from checkpoint {config.MODEL.RESUME}')
model = ptlPipeline.load_checkpoint(config.MODEL.PRETRAINED, config)
print(model)
print('Successfully applied checkpoint')

model.cpu()
model.eval()
print('Successfully moved to GPU')

# 1. Create dummy dataset
dummy_dataset = ABITemporalBenchmarkDataset(
    data_paths=[],
    split="val",
    img_size=512,
    in_chans=16,
    num_timesteps=7
)

# 2. Get one sample
imgs, ts = dummy_dataset[0]  # imgs: (T, C, H, W), ts: (T, 3)

# 3. Move inputs to GPU
imgs = imgs.unsqueeze(0).cpu()  # (B=1, T, C, H, W)
ts = torch.from_numpy(ts).unsqueeze(0).cpu()  # (B=1, T, 3)

# 4. Run inference
with torch.no_grad():
    loss, pred, mask = model(imgs, ts)

# 5. Post-process output
print("🧪 Inference done!")
print(f"Loss: {loss.item():.4f}")
print(f"Pred shape: {pred.shape}  # should be (B, T*P, D)")
print(f"Mask shape: {mask.shape}  # should be (B, T*P)")

# Optional: Convert prediction to images
B, T, C, H, W = imgs.shape
pred_imgs = model.model.unpatchify(pred, T, H, W)
pred_imgs = torch.clamp(pred_imgs, 0, 1)

print(f"Reconstructed images shape: {pred_imgs.shape}  # (B, T, C, H, W)")
