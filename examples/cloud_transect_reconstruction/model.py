import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchmetrics import JaccardIndex
from torchmetrics.functional import accuracy
import sys
import pytorch_lightning as pl

class UNET(nn.Module):
    def __init__(self, in_channels=16, decoder_classes=64, num_classes = 1, head_dropout = 0.2, output_shape=(91, 40)):
        super().__init__()
        self.layers = [in_channels, 64, 128, 256, 512, 1024]
        
        self.double_conv_downs = nn.ModuleList(
            [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])
        
        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
             for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])
            
        self.double_conv_ups = nn.ModuleList(
        [self.__double_conv(layer, layer//2) for layer in self.layers[::-1][:-2]])
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(64, decoder_classes, kernel_size=1)

        self.prediction_head = nn.Sequential(
            nn.Conv2d(decoder_classes, num_classes,
                      kernel_size=3, stride=1, padding=1),
            nn.Dropout(head_dropout),
            nn.Upsample(size=output_shape,
                        mode='bilinear',
                        align_corners=False)
        )

        
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
        # down layers
        concat_layers = []
        
        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)
        
        concat_layers = concat_layers[::-1]
        
        # up layers
        for up_trans, double_conv_up, concat_layer  in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:])
            
            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)
            
        x = self.final_conv(x)

        x = self.prediction_head(x)
        
        return x

    def predict(self, x):
        x = self.forward(x)
        x = torch.sigmoid(x)
        x = (x > 0.5).float()
        return x

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
 
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = torch.sum(probs * targets)
        cardinality = torch.sum(probs + targets)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score

class LightningUNET(pl.LightningModule):
    def __init__(self, in_channels=3, classes=1, dice_weight=0.5, learning_rate=0.001):
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.learning_rate = learning_rate
        self.model = UNET(in_channels=self.in_channels, num_classes=self.classes)
        self.iou = JaccardIndex(task="binary", num_classes=2)
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.forward(inputs)

        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        loss = self.dice_weight * dice + (1 - self.dice_weight) * ce
 
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.forward(inputs)

        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        loss = self.dice_weight * dice + (1 - self.dice_weight) * ce
 
        self.log("val_loss", loss, sync_dist=True)
        iou = self.iou(torch.sigmoid(logits) > 0.5, targets)
        self.log("val_iou", iou, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.forward(inputs)
        preds = torch.sigmoid(logits) > 0.5
        loss = self.iou(preds, targets)

        self.log("test_iou", loss, prog_bar=True)        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
