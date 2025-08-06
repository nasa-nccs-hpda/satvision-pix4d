#Importing libraries
import os
import sys
import torch
import glob
import tqdm
import subprocess
import torch.nn as nn
import torch.nn.functional as Fx
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset, DataLoader
import matplotlib as plt

sys.path.append('/explore/nobackup/people/jacaraba/development/satvision-toa')
from satvision_toa.utils import load_config
from satvision_toa.models.mim import build_mim_model

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

MODIS_ORDER_INDICES = [1, 2, 0, 4, 5, 6, 3, 8, 9, 10, 11, 13, 14, 15]

ABI_COEFFS = {
    # ABI_CH: [ESUN, FK1, FK2, BC1, BC2, SCALE, OFFSET, SV_MIN, SV_MAX]
    1:  [2017.1648, None, None, None, None, -25.936647,  0.8121064,   None, None],
    2:  [1631.3351, None, None, None, None, -20.289911,  0.15859237,  None, None],
    3:  [957.0699,  None, None, None, None, -12.037643,  0.37691253,  None, None],
    4:  [360.90176, None, None, None, None, -4.5223684,  0.07073108,  None, None],
    5:  [242.54037, None, None, None, None, -3.0596137,  0.09580004,  None, None],
    6:  [76.8999,   None, None, None, None, -0.9609507,  0.030088475, None, None],
    7:  [None, 202263.0, 3698.19, 0.43361, 0.99939, -0.0376, 0.001564351, 223.122,  352.7182],
    8:  [None, 50687.1,  2331.58, 1.55228, 0.99667, -0.5586, 0.007104763, None, None],
    9:  [None, 35828.3,  2076.95, 0.34427, 0.99918, -0.8236, 0.022539102, 178.9174, 261.2920],
    10: [None, 30174.0,  1961.38, 0.05651, 0.99986, -0.9561, 0.02004128,  204.3739, 282.5529],
    11: [None, 19779.9,  1703.83, 0.18733, 0.99948, -1.3022, 0.03335779,  204.7677, 319.0373],
    12: [None, 13432.1,  1497.61, 0.09102, 0.99971, -1.5394, 0.05443998,  194.8686, 295.0209],
    13: [None, 10803.3,  1392.74, 0.0755,  0.99975, -1.6443, 0.04572892,  None, None],
    14: [None, 8510.22,  1286.27, 0.22516, 0.9992,  -1.7187, 0.049492206, 202.1759, 324.0677],
    15: [None, 6454.62,  1173.03, 0.21702, 0.99916, -1.7558, 0.05277411,  201.3823, 321.5254],
    16: [None, 5101.27,  1084.53, 0.06266, 0.99974, -5.2392, 0.17605852,  203.3537, 285.9848],
}

# --- ADD: Physics-based calibration functions ---
def _vis_calibrate(data, esun):
    """Calibrate visible channels to reflectance."""
    solar_irradiance = esun
    esd = 0.99
    factor = np.pi * esd * esd / solar_irradiance
    reflectance = data * np.float32(factor)
    return np.clip(reflectance, 0, 1) # Clip to valid physical range

def _ir_calibrate(data, fk1, fk2, bc1, bc2, scale_factor, add_offset):
    """Calibrate IR channels to Brightness Temperature."""
    # Calculate radmin to ensure no negative BTs
    count_zero_rad = -add_offset / scale_factor
    count_pos = np.ceil(count_zero_rad)
    radmin = count_pos * scale_factor + add_offset
    
    data = data.clip(min=radmin)
    BT = (fk2 / np.log(fk1 / data + 1) - bc1) / bc2
    return BT


class CloudSatDataset(Dataset):
    '''
    Custom PyTorch dataset for loading and preprocessing each NPZ sample. Each sample contains ABI chip and corresponding CloudSat mask.
    '''
    def __init__(self, file_paths, target_size=(96, 40)):
        self.file_paths = file_paths
        self.target_height, self.target_width = target_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        '''
        Loads, preprocesses, and returns a single data sample. Returns a dictionary containing the processed chip, mask, and the original path.
        '''
        with np.load(self.file_paths[idx], allow_pickle=True) as data:
            abi_chip_raw = data['chip'].astype(np.float32)
            
            # Create an empty array for the calibrated data
            processed_chip = np.zeros_like(abi_chip_raw)

            # Calibrate each channel based on its physical type
            for i in range(16):
                channel_data = abi_chip_raw[:, :, i]
                coeffs = ABI_COEFFS[i + 1]
                
                if i < 6: # Visible channels (1-6)
                    esun = coeffs[0]
                    calibrated_data = _vis_calibrate(channel_data, esun)
                    processed_chip[:, :, i] = calibrated_data # Already normalized to [0,1]
                else: # Infrared channels (7-16)
                    fk1, fk2, bc1, bc2, scale, offset, sv_min, sv_max = coeffs[1:]
                    
                    bt = _ir_calibrate(channel_data, fk1, fk2, bc1, bc2, scale, offset)
                    
                    # Normalize BT using SatVision's pre-defined min/max
                    if sv_min is not None and sv_max is not None:
                        normalized_bt = (bt - sv_min) / (sv_max - sv_min)
                        processed_chip[:, :, i] = np.clip(normalized_bt, 0, 1)
                    else:
                        # For channels not used by SatVision (8 & 13), you can leave them as 0
                        processed_chip[:, :, i] = 0

            # Reorder channels to match MODIS and select the 14 channels SatVision uses
            final_chip = processed_chip[:, :, MODIS_ORDER_INDICES]
            final_chip_transposed = np.transpose(final_chip, (2, 0, 1))

            # Process the mask as before
            cloud_mask_raw = data['data'].item()['Cloud_mask'].astype(np.int64)[:, :34]
            pad_h = self.target_height - cloud_mask_raw.shape[0]
            pad_w = self.target_width - cloud_mask_raw.shape[1]
            cloud_mask_padded = np.pad(
                cloud_mask_raw,
                ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                'constant', constant_values=0
            )
            
            return {
                "chip": torch.from_numpy(final_chip_transposed).float(),
                "mask": torch.from_numpy(cloud_mask_padded),
                "path": self.file_paths[idx]
            }


class CloudSatDataModule(pl.LightningDataModule):
    '''
    PyTorch Lightning DataModule to handle the data: 
     - Splitting the data into train, val, and test sets.
     - Creating PyTorch DataLoaders for each set.
    '''
    def __init__(self, data_dir, new_data_dir, batch_size=16, num_workers=0, train_val_test_split=(0.8, 0.1, 0.1)):
        super().__init__()
        self.data_dir = data_dir
        self.new_data_dir = new_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        original_files = sorted(glob.glob(os.path.join(self.data_dir, '*.npz')))
        new_files = sorted(glob.glob(os.path.join(self.new_data_dir, '*.npz')))
        original_files.extend(new_files[:24000])
        self.file_paths = sorted(original_files)


    def setup(self, stage=None):
        '''
        Splits the data into train, val, and test sets. 
        The split is: 80% train, 10% val, and 10% test.
        '''
        
        n_total = len(self.file_paths)
        test_split_index = int(n_total*0.9)
        self.test_files = self.file_paths[test_split_index:]

        train_val_files = self.file_paths[:test_split_index]
        np.random.shuffle(train_val_files)

        val_split_index = int(len(train_val_files) * (1/9))
        self.val_files = train_val_files[:val_split_index]
        self.train_files = train_val_files[val_split_index:]

        self.train_dataset = CloudSatDataset(self.train_files)
        self.val_dataset = CloudSatDataset(self.val_files)
        self.test_dataset = CloudSatDataset(self.test_files)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False, collate_fn=self._collate_fn)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False, collate_fn=self._collate_fn)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False, collate_fn=self._collate_fn)


    def _collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch: return None
        return torch.utils.data.dataloader.default_collate(batch)


class DiceLoss(nn.Module):
    '''
    Calculates the Dice Loss. It's less sensitive to class imbalance than CE.
    '''
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth


    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)

        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()


class SatVisionUNet(nn.Module):
    '''
    U-Net architecture with SatVision.
    '''
    def __init__(self, out_channels=9, freeze_encoder=True, input_channels=14, final_size=(96, 40)):
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

        # Decoder
        self.fusion_conv3 = nn.Conv2d(4096, 1024, 1)
        self.fusion_conv2 = nn.Conv2d(2048, 512, 1)
        self.fusion_conv1 = nn.Conv2d(1024, 256, 1)

        self.upconv4 = nn.Sequential(nn.Conv2d(4096, 2048, 3, padding=1), nn.ReLU(), nn.Conv2d(2048, 1024, 3, padding=1))
        self.upconv3 = nn.Sequential(nn.Conv2d(1024 + 1024, 1024, 3, padding=1), nn.ReLU(), nn.Conv2d(1024, 512, 3, padding=1))
        self.upconv2 = nn.Sequential(nn.Conv2d(512 + 512, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 256, 3, padding=1))
        self.upconv1 = nn.Sequential(nn.Conv2d(256 + 256, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 128, 3, padding=1))
        self.upconv0 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 64, 3, padding=1))

        # Final prediction layer (multi-class)
        self.final = nn.Conv2d(64, out_channels, 1)

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
        x = self.final(x)  # [B, 9, 128, 128]

        # Resize to target (96, 40)
        x = F.interpolate(x, size=self.final_size, mode="bilinear", align_corners=False)
        return x


class SatVisionUNetLightning(pl.LightningModule):
    '''
    PyTorch Lightning wrapper for the SatVision U-Net model. It handles the training, validation, and testing logic. It includes the loss calculuation, the optimization, and the logging.
    '''
    def __init__(self, lr=1e-4, dice_weight=0.5, freeze_encoder=True, num_classes=9):
        super().__init__()
        self.save_hyperparameters()
        self.model = SatVisionUNet(out_channels=num_classes, freeze_encoder=freeze_encoder, input_channels=14, final_size=(96, 40))
                
        #Combination of both CE and Dice
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, stage):
        chips, masks = batch["chip"], batch["mask"]
        logits = self.forward(chips)  # (B, C, H, W)
                
        #calculate individual losses
        ce = self.ce_loss(logits, masks)
        dice = self.dice_loss(logits, masks)

        #combine losses
        loss = self.dice_weight * dice + (1 - self.dice_weight) * ce

        #logging metrics
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True)

        return loss

    def training_step(self, b, i): return self._common_step(b, 'train')
    def validation_step(self, b, i): return self._common_step(b, 'val')
    def test_step(self, b, i): return self._common_step(b, 'test')

    #using adam optimizer    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def visualize_prediction(chip_tensor, true_mask_padded, pred_mask_padded, file_path):
    '''
    Generates and saves a visualization comparing the input chip, the true mask, and the predicted mask.
    '''
    chip = chip_tensor[0].cpu().numpy()
    true_mask = true_mask_padded.cpu().numpy()
    pred_mask = pred_mask_padded.cpu().numpy()

    original_height, original_width = 91, 34
    pad_height_total = true_mask.shape[0] - original_height
    pad_top = pad_height_total // 2
    pad_width_total = true_mask.shape[1] - original_width
    pad_left = pad_width_total // 2

    true_mask_unpadded = true_mask[pad_top : pad_top + original_height, pad_left : pad_left + original_width]
    pred_mask_unpadded = pred_mask[pad_top : pad_top + original_height, pad_left : pad_left + original_width]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f"File: {os.path.basename(file_path)}", fontsize=16)

    ax1.imshow(chip, cmap='gray')
    ax1.set_title("Input ABI Chip (Channel 1)")
    ax1.axis('off')

    plot_curtain(ax2, true_mask_unpadded, "Ground Truth Classification", fig)
    plot_curtain(ax3, pred_mask_unpadded, "Predicted Classification", fig)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    #saving the figure as a png to a specified path
    output_filename = os.path.basename(file_path).replace('.npz', '.png')
    save_path = os.path.join('/explore/nobackup/people/sjaddu/test_images/7-30-2025/satvision', output_filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #plt.show()
    plt.close(fig)
    print(f"Saved visualization to: {save_path}")


def plot_curtain(ax, mask, title, fig):
    '''
    Helper function to create a vertical curtain plot for cloud masks.
    '''
    num_classes = 9
    zz = mask.T

    x_axis_points = np.arange(zz.shape[1])
    y_axis_km = np.arange(0, zz.shape[0] * 0.5, 0.5)


    mesh = ax.pcolormesh(x_axis_points, y_axis_km, zz, cmap='tab10', shading='nearest', vmin=-0.5, vmax=num_classes - 0.5)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Point Along Track", fontsize=12)
    ax.set_ylabel("Height (km)", fontsize=12)
    ax.set_ylim(0, zz.shape[0] * 0.5)
    ax.grid(True, linestyle='--', alpha=0.5)

    cbar = fig.colorbar(mesh, ax=ax, ticks=np.arange(num_classes))
    cbar.set_ticklabels(['0: Not Determined', '1: Cirrus', '2: Alotstratus', '3: Altocumulus', '4: Stratus', '5: Stratocumulus', '6: Cumulus', '7: Nimbostratus', '8: Deep Convection'])


def calculate_iou(pred_mask, true_mask, num_classes):

    '''
    Calculates the mean Intersection over Union (mIoU) and per-class IoU.
    '''

    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()

    per_class_iou = np.zeros(num_classes)

    for cls in range(num_classes):
        intersection = np.sum((pred_mask == cls) & (true_mask == cls))

        union = np.sum((pred_mask == cls) | (true_mask == cls))

        if union == 0:
            per_class_iou[cls] = np.nan
        else:
            per_class_iou[cls] = intersection / union

    mIoU = np.nanmean(per_class_iou)

    return mIoU, per_class_iou


# --- Main Execution Block (Testing and Training) ---
if __name__ == '__main__':
    # --- Configuration ---
    DATA_DIRECTORY = "/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16"
    NEW_DATA_DIRECTORY = "/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16-New"
    BATCH_SIZE = 64
    NUM_WORKERS = 0
    MAX_EPOCHS = 150
    LEARNING_RATE = 1e-5
    TARGET_HEIGHT = 96
    TARGET_WIDTH = 40
    NUM_SAMPLES_TO_TEST = 5

    pl.seed_everything(42)

    # --- PART 1: TRAINING ---
    #Initialize the main DataModule
    datamodule = CloudSatDataModule(data_dir=DATA_DIRECTORY, new_data_dir=NEW_DATA_DIRECTORY, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    #Initialize the Lightning model
    model = SatVisionUNetLightning(num_classes=9, lr=LEARNING_RATE)
        
    #Save the best model based on val loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='/explore/nobackup/people/sjaddu/checkpoints/',
        #name of saved model checkpoint
        filename='SatVision-731-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    #stop training if the val loss doesn't improve after 5 epochs
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=7,
        verbose=True,
        mode='min'
    )

    #create the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=MAX_EPOCHS,
        accelerator='auto',
        log_every_n_steps=10,
        default_root_dir='/explore/nobackup/people/sjaddu'
    )

    #training the model
    print("--- Starting Model Training ---")
    trainer.fit(model, datamodule)
    print("--- Training Complete ---")

    
    # --- PART 2: EVALUATION ---
    best_model_path = checkpoint_callback.best_model_path
    print(f"\n--- Loading best model for evaluation: {os.path.basename(best_model_path)} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #loading the best performing model from the checkpoint
    trained_model = SatVisionUNetLightning.load_from_checkpoint(best_model_path)
    trained_model.eval()

    trained_model.to(device)
    
    #get the test dataloader
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()

    all_preds = []
    all_true_masks = []

    #make predictions
    print(f"\n--- Evaluating model on the full test set ({len(datamodule.test_files)} samples) ---")
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Testing"):
            if batch is None: continue
            chips = batch['chip'].to(device)
            true_masks = batch['mask']

            logits = trained_model(chips)
            predicted_masks = torch.argmax(logits, dim=1)

            all_preds.append(predicted_masks.cpu().numpy())
            all_true_masks.append(true_masks.cpu().numpy())

    #Display Metrics   
    if all_preds:
            all_preds = np.concatenate(all_preds, axis=0)
            all_true_masks = np.concatenate(all_true_masks, axis=0)

            NUM_CLASSES = 9
            CLASS_NAMES = ['0: Not Determined', '1: Cirrus', '2: Alotstratus', '3: Altocumulus', '4: Stratus', '5: Stratocumulus', '6: Cumulus', '7: Nimbostratus', '8: Deep Convection']

            #IoU metrics
            mean_iou, class_iou = calculate_iou(all_preds, all_true_masks, NUM_CLASSES)

            iou_data = {'Class Name': CLASS_NAMES, 'IoU': class_iou}
            iou_df = pd.DataFrame(iou_data)

            print("\n--- Model Performance on Test Set ---")
            print(f"\nMean IoU (mIoU): {mean_iou:.4f}\n")
            print("Per-Class IoU:")
            print(iou_df.to_string(index=False, float_format="%.4f"))

    
    #visualization of 5 random test samples and corresponding predictions
    vis_loader = DataLoader(datamodule.test_dataset, batch_size=5, shuffle=True, collate_fn=datamodule._collate_fn)
    print(f"\n--- Visualizing {5} random test samples ---")
    vis_batch = next(iter(vis_loader))
    if vis_batch:
        vis_chips = vis_batch['chip'].to(device)
        vis_true_masks = vis_batch['mask']
        vis_file_paths = vis_batch['path']

        with torch.no_grad():
            vis_logits = trained_model(vis_chips)
            vis_predicted_masks = torch.argmax(vis_logits, dim=1)

        #function call to the save the test PNGs
        for i in range(len(vis_chips)):
            visualize_prediction(vis_chips[i], vis_true_masks[i], vis_predicted_masks[i], vis_file_paths[i])
