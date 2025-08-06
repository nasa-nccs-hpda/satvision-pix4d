import sys

sys.path.append('/explore/nobackup/people/jacaraba/development/satvision-toa')
from satvision_toa.models.mim import build_mim_model
from satvision_toa.utils import load_config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchmetrics import JaccardIndex
import lightning as L


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = torch.sum(probs * targets)
        cardinality = torch.sum(probs + targets)
        dice_score = (2. * intersection + self.smooth) / \
            (cardinality + self.smooth)
        return 1. - dice_score


class SatVisionUNet(nn.Module):
    def __init__(self, out_channels=1, freeze_encoder=True, input_channels=14, final_size=(96, 40)):
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
        checkpoint = {k.replace('model.encoder.', ''): v for k, v in checkpoint.items(
        ) if k.startswith('model.encoder')}
        self.encoder.load_state_dict(checkpoint, strict=False)

        # Freeze encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.fusion_conv3 = nn.Conv2d(4096, 1024, 1)
        self.fusion_conv2 = nn.Conv2d(2048, 512, 1)
        self.fusion_conv1 = nn.Conv2d(1024, 256, 1)

        self.upconv4 = nn.Sequential(nn.Conv2d(
            4096, 2048, 3, padding=1), nn.ReLU(), nn.Conv2d(2048, 1024, 3, padding=1))
        self.upconv3 = nn.Sequential(nn.Conv2d(
            1024 + 1024, 1024, 3, padding=1), nn.ReLU(), nn.Conv2d(1024, 512, 3, padding=1))
        self.upconv2 = nn.Sequential(nn.Conv2d(
            512 + 512, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 256, 3, padding=1))
        self.upconv1 = nn.Sequential(nn.Conv2d(
            256 + 256, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 128, 3, padding=1))
        self.upconv0 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 64, 3, padding=1))

        self.final = nn.Conv2d(64, 64, kernel_size=1)

        self.prediction_head = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            nn.Upsample(size=self.final_size,
                        mode='bilinear', align_corners=False)
        )

    def forward(self, x, mask=None):
        # Multi-scale encoder features
        # returns [stage1, stage2, stage3, stage4]
        enc_features = self.encoder.extra_features(x)
        x = enc_features[-1]  # [B, 4096, 4, 4]

        # Upconv4: 4x4 -> 8x8
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)
        x = self.upconv4(x)

        # Upconv3: 8x8 -> 16x16 with fusion
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)
        enc_feat = F.interpolate(
            enc_features[2], size=x.shape[2:], mode="bilinear", align_corners=False)
        enc_feat = self.fusion_conv3(enc_feat)
        x = torch.cat([x, enc_feat], dim=1)
        x = self.upconv3(x)

        # Upconv2: 16x16 -> 32x32 with fusion
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)
        enc_feat = F.interpolate(
            enc_features[1], size=x.shape[2:], mode="bilinear", align_corners=False)
        enc_feat = self.fusion_conv2(enc_feat)
        x = torch.cat([x, enc_feat], dim=1)
        x = self.upconv2(x)

        # Upconv1: 32x32 -> 64x64 with fusion
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)
        enc_feat = F.interpolate(
            enc_features[0], size=x.shape[2:], mode="bilinear", align_corners=False)
        enc_feat = self.fusion_conv1(enc_feat)
        x = torch.cat([x, enc_feat], dim=1)
        x = self.upconv1(x)

        # Final upconv: 64x64 -> 128x128
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)
        x = self.upconv0(x)
        x = self.final(x)  # [B, 64, 128, 128]

        x = self.prediction_head(x)  # [B, out_channels, 91, 40]

        return x


class UNET(nn.Module):
    def __init__(self, in_channels=16, decoder_classes=64, num_classes=1, head_dropout=0.2, output_shape=(91, 40)):
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
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
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
        for up_trans, double_conv_up, concat_layer in zip(self.up_trans, self.double_conv_ups, concat_layers):
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


class LightningModel(L.LightningModule):
    def __init__(self, lr=1e-4, dice_weight=0.5, freeze_encoder=True, num_classes=1, model_name="SatVisionUNet", in_channels=14, final_size=(91, 40), head_dropout=0.2):
        super().__init__()
        if model_name == "SatVisionUNet":
            self.model = SatVisionUNet(
                out_channels=num_classes, freeze_encoder=freeze_encoder, input_channels=in_channels, final_size=final_size)
        elif model_name == "UNet":
            self.model = UNET(in_channels=in_channels, decoder_classes=64,
                              num_classes=num_classes, head_dropout=head_dropout, output_shape=final_size)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.save_hyperparameters()
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight
        self.iou = JaccardIndex(task="binary", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, stage):
        chips, masks = batch["chip"], batch["mask"]
        logits = self.forward(chips)  # (B, C, H, W)
        ce = self.ce_loss(logits, masks)
        dice = self.dice_loss(logits, masks)
        loss = self.dice_weight * dice + (1 - self.dice_weight) * ce
        self.log(f'{stage}_loss', loss, prog_bar=True,
                 on_epoch=True, on_step=False)
        output = {
            "loss": loss,
        }
        if stage == 'val' or stage == 'test':
            iou = self.iou(logits.sigmoid() > 0.5, masks)
            self.log(f'{stage}_iou', iou, prog_bar=True,
                     on_epoch=True, on_step=False)
            if stage == 'test':
                output["iou"] = iou
        return output

    def training_step(self, b, i): return self._common_step(b, 'train')
    def validation_step(self, b, i): return self._common_step(b, 'val')
    def test_step(self, b, i): return self._common_step(b, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
