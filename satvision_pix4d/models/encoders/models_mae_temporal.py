import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from flash_attn.modules.mha import MHA

# These must be in your environment
from satvision_pix4d.models.utils.pos_embed import (
    get_2d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid_torch,
)


class FlashMHAWrapper(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, causal):
        super().__init__()
        self.attn = MHA(
            embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal,
        )

    def forward(self, x, attn_mask=None):
        # Always match dtype with the MHA weights
        x = x.to(self.attn.Wqkv.weight.dtype)
        return self.attn(x)


class MaskedAutoencoderViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        same_mask=False,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        for blk in self.blocks:
            blk.attn = FlashMHAWrapper(
                embed_dim,
                num_heads=num_heads,
                dropout=0.0,
                causal=False,
            )

        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        for blk in self.decoder_blocks:
            blk.attn = FlashMHAWrapper(
                decoder_embed_dim,
                num_heads=decoder_num_heads,
                dropout=0.0,
                causal=False,
            )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.same_mask = same_mask

        self.encoder_temporal_spatial_proj = None
        self.decoder_temporal_spatial_proj = None

        self.initialize_weights()

    def initialize_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        B, T, C, H, W = imgs.shape
        h = H // p
        w = W // p
        x = imgs.reshape(B, T, C, h, p, w, p)
        x = x.permute(0, 1, 3, 5, 4, 6, 2).reshape(B, T * h * w, p * p * C)
        return x

    def unpatchify(self, x, T, H, W):
        p = self.patch_embed.patch_size[0]
        B = x.shape[0]
        L_per_step = x.shape[1] // T
        h = H // p
        w = W // p
        C = x.shape[-1] // (p * p)
        x = x.reshape(B, T, h, w, p, p, C).permute(0, 1, 6, 2, 4, 3, 5).reshape(B, T, C, H, W)
        return x

    def random_masking(self, x, mask_ratio, mask=None):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1) if mask is None else mask
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def _compute_temporal_embedding(self, timestamps_flat, embed_dim_per_component):
        ts_embed_list = [
            get_1d_sincos_pos_embed_from_grid_torch(embed_dim_per_component, timestamps_flat[:, k].float())
            for k in range(timestamps_flat.shape[1])
        ]
        return torch.cat(ts_embed_list, dim=1)

    def forward_encoder(self, x, timestamps, mask_ratio, mask=None):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        x_emb = self.patch_embed(x_flat)
        L_per_step = x_emb.shape[1]
        x_emb = x_emb.reshape(B, T * L_per_step, -1)

        grid_size = int(L_per_step ** 0.5)
        pos_embed_spatial = get_2d_sincos_pos_embed(
            embed_dim=x_emb.shape[2],
            grid_size=grid_size,
            cls_token=False,
        )
        pos_embed_spatial = torch.from_numpy(pos_embed_spatial).to(x.device).to(x_emb.dtype)
        pos_embed_spatial = pos_embed_spatial.repeat(T, 1).unsqueeze(0).expand(B, -1, -1)

        timestamps_flat = timestamps.reshape(B * T, -1)
        ts_embed = self._compute_temporal_embedding(timestamps_flat, embed_dim_per_component=128)
        ts_embed = ts_embed.reshape(B, T, 1, -1).expand(B, T, L_per_step, -1).reshape(B, T * L_per_step, -1)

        embedding = torch.cat([pos_embed_spatial, ts_embed], dim=-1)

        if self.encoder_temporal_spatial_proj is None:
            self.encoder_temporal_spatial_proj = nn.Linear(
                embedding.shape[-1],
                x_emb.shape[-1]
            ).to(x.device)

        embedding = self.encoder_temporal_spatial_proj(embedding)
        x = x_emb + embedding

        # Cast to consistent dtype
        x = x.to(self.cls_token.dtype)

        x, mask, ids_restore = self.random_masking(x, mask_ratio, mask=mask)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        for blk in self.blocks:
            x = blk(x.to(self.cls_token.dtype))
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, timestamps, ids_restore):
        B = x.shape[0]
        T = timestamps.shape[1]
        L_per_step = ids_restore.shape[1] // T

        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        grid_size = int(L_per_step ** 0.5)
        pos_embed_spatial = get_2d_sincos_pos_embed(
            embed_dim=x.shape[2],
            grid_size=grid_size,
            cls_token=False,
        )
        pos_embed_spatial = torch.from_numpy(pos_embed_spatial).to(x.device).to(x.dtype)
        pos_embed_spatial = pos_embed_spatial.repeat(T, 1)
        pos_embed_spatial = torch.cat([
            torch.zeros(1, pos_embed_spatial.shape[1], device=x.device),
            pos_embed_spatial,
        ], dim=0).unsqueeze(0).expand(B, -1, -1)

        timestamps_flat = timestamps.reshape(B * T, -1)
        ts_embed = self._compute_temporal_embedding(timestamps_flat, embed_dim_per_component=64)
        ts_embed = ts_embed.reshape(B, T, 1, -1).expand(B, T, L_per_step, -1).reshape(B, T * L_per_step, -1)
        ts_embed = torch.cat([
            torch.zeros(B, 1, ts_embed.shape[-1], device=x.device),
            ts_embed,
        ], dim=1)

        embedding = torch.cat([pos_embed_spatial, ts_embed], dim=-1)

        if self.decoder_temporal_spatial_proj is None:
            self.decoder_temporal_spatial_proj = nn.Linear(
                embedding.shape[-1],
                x.shape[-1]
            ).to(x.device)

        embedding = self.decoder_temporal_spatial_proj(embedding)
        x = x + embedding

        # Cast to consistent dtype
        x = x.to(self.cls_token.dtype)

        for blk in self.decoder_blocks:
            x = blk(x.to(self.cls_token.dtype))
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, timestamps, mask_ratio=0.75, mask=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, timestamps, mask_ratio, mask)
        pred = self.forward_decoder(latent, timestamps, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

# Test
if __name__ == "__main__":
    B, T, C, H, W = 2, 3, 14, 512, 512
    TT = 5

    imgs = torch.randn(B, T, C, H, W)
    timestamps = torch.randint(low=0, high=1000, size=(B, T, TT))

    model = MaskedAutoencoderViT(
        img_size=H,
        patch_size=16,
        in_chans=C,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        same_mask=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    imgs = imgs.to(device)
    timestamps = timestamps.to(device)

    loss, pred, mask = model(imgs, timestamps)
    print("Loss:", loss.item())
    print("Pred shape:", pred.shape)
