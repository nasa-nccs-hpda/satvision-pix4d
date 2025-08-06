# Importing Libraries
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns


class CloudSatDataset(Dataset):
    '''
    Custom PyTorch dataset for loading and preprocessing each NPZ sample. Each sample contains ABI chip and corresponding CloudSat mask.
    '''
    def __init__(self, file_paths, channel_mins, channel_maxs, target_size=(96, 40)):
        self.file_paths = file_paths
        self.target_height, self.target_width = target_size
        self.channel_mins = channel_mins
        self.channel_maxs = channel_maxs

    def __len__(self):
        return len(self.file_paths)


    def __getitem__(self, idx):
        '''
        Loads, preprocesses, and returns a single data sample. Returns a dictionary containing the processed chip, mask, and the original path.
        '''
        with np.load(self.file_paths[idx], allow_pickle=True) as data:
            abi_chip = data['chip'].astype(np.float32)

            mins = self.channel_mins.reshape(1, 1, -1)
            maxs = self.channel_maxs.reshape(1, 1, -1)
            abi_chip = (abi_chip - mins) / (maxs - mins + 1e-6)

            abi_chip = np.transpose(abi_chip, (2, 0, 1))

            cloud_mask_raw = data['data'].item()['Cloud_mask'].astype(np.int64)
            cloud_mask_raw = cloud_mask_raw[:, :34]
            pad_height_total = self.target_height - cloud_mask_raw.shape[0]
            pad_top = pad_height_total // 2
            pad_bottom = pad_height_total - pad_top
            pad_width_total = self.target_width - cloud_mask_raw.shape[1]
            pad_left = pad_width_total // 2
            pad_right = pad_width_total - pad_left
            cloud_mask_padded = np.pad(
                cloud_mask_raw,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                'constant',
                constant_values=0
            )
           
            return {
                "chip": torch.from_numpy(abi_chip),
                "mask": torch.from_numpy(cloud_mask_padded),
                "path": self.file_paths[idx]
            }


class CloudSatDataModule(pl.LightningDataModule):
    '''
    PyTorch Lightning DataModule to handle the data: 
     - Splitting the data into train, val, and test sets.
     - Creating PyTorch DataLoaders for each set.
    '''
    def __init__(self, data_dir, batch_size=16, num_workers=0, train_val_test_split=(0.8, 0.1, 0.1)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.file_paths = sorted(glob.glob(os.path.join(self.data_dir, '*.npz')))
        self.channel_mins = None
        self.channel_maxs = None

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

        self.train_dataset = CloudSatDataset(self.train_files, self.channel_mins, self.channel_maxs)
        self.val_dataset = CloudSatDataset(self.val_files, self.channel_mins, self.channel_maxs)
        self.test_dataset = CloudSatDataset(self.test_files, self.channel_mins, self.channel_maxs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False, collate_fn=self._collate_fn)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False, collate_fn=self._collate_fn)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False, collate_fn=self._collate_fn)


    def _collate_fn(self, batch):
        '''
        Collate function to filter out 'None' values.
        '''
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


class CustomUNET(nn.Module):
    '''
    Custom U-Net architecture for segmentation. Has an encoder and a decoder as well as skip connections.
    '''
    def __init__(self, in_channels=16, num_classes=9, output_shape=(96, 40)):
        super().__init__()
        #Define the number of channels at each level of the U-Net
        self.layers = [in_channels, 64, 128, 256, 512, 1024]

        #Encoder
        self.double_conv_downs = nn.ModuleList(
            [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])

        #Decoder
        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
             for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])

        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer, layer//2) for layer in self.layers[::-1][:-2]])

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #Final Layers
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_upsample = nn.Upsample(
            size=output_shape, mode='bilinear', align_corners=False
        )

    #Helper function to create a block of two conv layers 
    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv

    def forward(self, x):
        # Down layers
        concat_layers = []
        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)
        
        concat_layers = concat_layers[::-1]

        # Up layers
        for up_trans, double_conv_up, concat_layer in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:])
            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)

        #Apply the final layers to get the final segmentation map
        x = self.final_conv(x)
        x = self.final_upsample(x)
        return x


class CustomUNETLightning(pl.LightningModule):
    '''
    PyTorch Lightning wrapper for the CustomU-Net model. It handles the training, validation, and testing logic. It includes the loss calculuation, the optimization, and the logging.
    '''
    def __init__(self, in_channels=16, num_classes=9, target_height=96, target_width=40, lr=1e-4, dice_weight=0.5):
        super().__init__()
        self.save_hyperparameters()

        self.model = CustomUNET(
            in_channels=self.hparams.in_channels,
            num_classes=self.hparams.num_classes,
            output_shape=(self.hparams.target_height, self.hparams.target_width)
        )

        #Combination of both CE and Dice
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx, stage):
        if batch is None: return None
        chips, masks = batch["chip"], batch["mask"]
        logits = self.forward(chips)

        #calculate individual losses
        ce = self.ce_loss(logits, masks)
        dice = self.dice_loss(logits, masks.long())

        #combine losses
        loss = self.dice_weight * dice + (1 - self.dice_weight) * ce

        #logging metrics
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True)
        self.log(f'{stage}_ce_loss', ce, on_epoch=True, logger=True)
        self.log(f'{stage}_dice_loss', dice, on_epoch=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')

        
    #using adam optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def precompute_normalization_stats(file_paths, num_channels=16):

    '''
    Calcuates the minimum and maximum pixel values for each channel across the entire training dataset. Very important for min-max normalization.
    '''
    
    channel_mins = np.full(num_channels, np.inf, dtype=np.float32)
    channel_maxs = np.full(num_channels, -np.inf, dtype=np.float32)
    
    for file_path in tqdm.tqdm(file_paths):
        with np.load(file_path, allow_pickle=True) as data:
            abi_chip = data['chip'].astype(np.float32)
            
            current_mins = np.min(abi_chip, axis=(0, 1))
            current_maxs = np.max(abi_chip, axis=(0, 1))
            
            channel_mins = np.minimum(channel_mins, current_mins)
            channel_maxs = np.maximum(channel_maxs, current_maxs)
            
    print("\n--- Global Min/Max Stats Calculated ---")
    for i in range(num_channels):
        print(f"Channel {i+1}: Min={channel_mins[i]:.2f}, Max={channel_maxs[i]:.2f}")
        
    return channel_mins, channel_maxs


def visualize_prediction(chip_tensor, true_mask_padded, pred_mask_padded, file_path, dominant_class_name = None):
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

    title_prefix = f"Dominant Class: {dominant_class_name} | " if dominant_class_name else "Random Sample | "
    fig.suptitle(f"{title_prefix}File: {os.path.basename(file_path)}", fontsize=16)
    
    ax1.imshow(chip, cmap='gray')
    ax1.set_title("Input ABI Chip (Channel 1)")
    ax1.axis('off')

    plot_curtain(ax2, true_mask_unpadded, "Ground Truth Classification", fig)
    plot_curtain(ax3, pred_mask_unpadded, "Predicted Classification", fig)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    #saving the figure a png to specified path
    base_filename = os.path.basename(file_path).replace('.npz', '.png')
    file_prefix = f"{dominant_class_name}_dominant_" if dominant_class_name else "random_"
    output_filename = f"{file_prefix}{base_filename}"
    save_path = os.path.join('/explore/nobackup/people/sjaddu/test_images/7-23-2025/test_sanity', output_filename)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
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

# --- Main Execution Block (Training and Testing) ---
if __name__ == '__main__':
    
    # --- Configuration ---
    DATA_DIRECTORY = "/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16"
    BATCH_SIZE = 64
    NUM_WORKERS = 0
    MAX_EPOCHS = 150
    LEARNING_RATE = 1e-4
    TARGET_HEIGHT = 96
    TARGET_WIDTH = 40
    NUM_SAMPLES_TO_TEST = 5
   
    pl.seed_everything(42)

    #Creates a temporary DataModule just to get the file list for stat calculation and normalization
    temp_datamodule_for_stats = CloudSatDataModule(data_dir=DATA_DIRECTORY)
    temp_datamodule_for_stats.setup()
    channel_mins, channel_maxs = precompute_normalization_stats(temp_datamodule_for_stats.train_files)

    # --- PART 1: TRAINING ---
    
    #Initialize the main DataModule and provide it with the previously calculated data normalization stats
    datamodule = CloudSatDataModule(data_dir=DATA_DIRECTORY, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    datamodule.channel_mins = channel_mins
    datamodule.channel_maxs = channel_maxs

    #Initialize the Lightning model
    model = CustomUNETLightning(in_channels=16, num_classes=9, target_height=TARGET_HEIGHT, target_width=TARGET_WIDTH, lr=LEARNING_RATE, dice_weight=0.5)

    #Save the best model based on val loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='/explore/nobackup/people/sjaddu/checkpoints/',
        #name of saved model checkpoint
        filename='cloudsat-best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    #stop training if the val loss doesn't improve after 5 epochs
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )

    #create the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=MAX_EPOCHS,
        accelerator='auto',
        log_every_n_steps=10,
        default_root_dir='/explore/nobackup/people/sjaddu/'
    )

    #training the model
    print("--- Starting Model Training ---")
    trainer.fit(model, datamodule)
    print("--- Training Complete ---")

    
    # --- PART 2: EVALUATION ---
    best_model_path = checkpoint_callback.best_model_path
    print(f"\n--- Loading best model for evaluation: {os.path.basename(best_model_path)} ---")
    
    #loading the best performing model from the checkpoint
    trained_model = CustomUNETLightning.load_from_checkpoint(best_model_path)
    trained_model.eval()
    
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
            chips = batch['chip']#.to(device)
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

            #confusion matrix
            true_flat = all_true_masks.flatten()
            pred_flat = all_preds.flatten()
            
            cm = confusion_matrix(true_flat, pred_flat, labels=np.arange(NUM_CLASSES))
            
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            
            short_class_names = [name.split(': ')[1] for name in CLASS_NAMES]
            cm_df = pd.DataFrame(cm_normalized, index=[f"True: {name}" for name in short_class_names], columns=[f"Pred: {name}" for name in short_class_names])
            
            print("\n--- Normalized Confusion Matrix (Rows = True Class, Columns = Predicted Class) ---\n")
            print(cm_df.to_string(float_format="%.2f"))

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
        vis_chips = vis_batch['chip']#.to(device)
        vis_true_masks = vis_batch['mask']
        vis_file_paths = vis_batch['path']

        with torch.no_grad():
            vis_logits = trained_model(vis_chips)
            vis_predicted_masks = torch.argmax(vis_logits, dim=1)

        #function call to the save the test PNGs
        for i in range(len(vis_chips)):
            visualize_prediction(vis_chips[i], vis_true_masks[i], vis_predicted_masks[i], vis_file_paths[i])


