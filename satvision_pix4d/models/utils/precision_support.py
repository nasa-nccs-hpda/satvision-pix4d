import torch
import torch.nn as nn
import torch.nn.functional as F

class FP32LayerNorm(nn.LayerNorm):
    def forward(self, x):
        print("FP32LayerNorm input dtype:", x.dtype)
        out = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps
        )
        out = out.to(x.dtype)
        print("FP32LayerNorm output dtype:", out.dtype)
        return out
