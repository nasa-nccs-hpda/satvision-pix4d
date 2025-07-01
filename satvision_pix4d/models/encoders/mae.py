import logging
import torch.nn as nn
from functools import partial
from satvision_pix4d.models.encoders.models_mae_temporal import \
    MaskedAutoencoderViT as MaskedAutoencoderViTTemporal
from satvision_pix4d.models.utils.precision_support import FP32LayerNorm

# -----------------------------------------------------------------------------
# build_satmae_model
# -----------------------------------------------------------------------------
def build_satmae_model(config):
    """Builds the masked-image-modeling model.

    Args:
        config: config object

    Raises:
        NotImplementedError: if the model is
        not swinv2, then this will be thrown.

    Returns:
        MiMModel: masked-image-modeling model
    """
    model_type = config.MODEL.TYPE
    if model_type == 'satmae':
        model = MaskedAutoencoderViTTemporal(
            img_size=config.DATA.IMG_SIZE,
            in_chans=config.MODEL.MAE_VIT.IN_CHANS,
            patch_size=config.MODEL.MAE_VIT.PATCH_SIZE,
            embed_dim=config.MODEL.MAE_VIT.EMBED_DIM,
            depth=config.MODEL.MAE_VIT.DEPTHS,
            num_heads=config.MODEL.MAE_VIT.NUM_HEADS,
            decoder_embed_dim=config.MODEL.MAE_VIT.DECODER_EMBED_DIM,
            decoder_depth=config.MODEL.MAE_VIT.DECODER_DEPTH,
            decoder_num_heads=config.MODEL.MAE_VIT.DECODER_NUM_HEADS,
            mlp_ratio=config.MODEL.MAE_VIT.MLP_RATIO,
            norm_layer=partial(FP32LayerNorm, eps=1e-6)
            #norm_layer=partial(
            #    nn.LayerNorm,
            #    eps=1e-6,
            #),
        )

        """
        self, img_size=224, patch_size=16, in_chans=3,
                        embed_dim=1024, depth=24, num_heads=16,
                        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                        mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, same_mask=False
        """

        logging.info(str(model))
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    return model
