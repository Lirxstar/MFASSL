"""MAE self-supervised backbone for MFASSL."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...models.vit import MFASSLViT
from ..symmetry import reflection_consistency_loss, mid_layer_consistency_loss

def random_masking(x: torch.Tensor, mask_ratio: float, generator=None):

    b, n, d = x.shape
    keep = max(1, int(round(n * (1.0 - mask_ratio))))
    noise = torch.rand(b, n, device=x.device, generator=generator)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :keep]
    x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
    mask = torch.ones(b, n, device=x.device)
    mask[:, :keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_masked, mask, ids_restore, ids_keep

def scatter_to_full(packed: torch.Tensor, ids_keep: torch.Tensor, n: int) -> torch.Tensor:

    b, _, d = packed.shape
    full = torch.zeros(b, n, d, device=packed.device, dtype=packed.dtype)
    full.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, d), packed)
    return full

def visible_mask(ids_keep: torch.Tensor, n: int) -> torch.Tensor:

    b, keep = ids_keep.shape
    m = torch.zeros(b, n, dtype=torch.bool, device=ids_keep.device)
    m.scatter_(1, ids_keep, True)
    return m

class MAE(nn.Module):

    def __init__(self, vit_kwargs: dict, decoder_dim: int = 512, decoder_depth: int = 2,
                 decoder_heads: int = 8, mask_ratio: float = 0.75, **_ignored):
        super().__init__()
        from timm.models.vision_transformer import Block

        self.mask_ratio = mask_ratio
        self.vit = MFASSLViT(**vit_kwargs)
        bb = self.vit.backbone
        self.patch_embed = bb.patch_embed
        self.blocks = bb.blocks
        self.norm = bb.norm
        self.cls_token = bb.cls_token
        self.pos_embed = bb.pos_embed
        self.embed_dim = self.vit.embed_dim
        self.num_patches = self.patch_embed.num_patches
        ps = self.patch_embed.patch_size
        self.patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps
        self.in_chans = vit_kwargs.get("in_chans", 3)

        self.decoder_embed = nn.Linear(self.embed_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, decoder_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_dim, decoder_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_size ** 2 * self.in_chans)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

    @property
    def encoder(self) -> MFASSLViT:
        return self.vit

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p, c = self.patch_size, self.in_chans
        b, _, h, w = imgs.shape
        gh, gw = h // p, w // p
        x = imgs.reshape(b, c, gh, p, gw, p)
        x = torch.einsum("bchpwq->bhwpqc", x)
        return x.reshape(b, gh * gw, p * p * c)

    def _embed_patches(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(imgs)
        if x.dim() == 4:
            x = x.flatten(1, 2)
        return x + self.pos_embed[:, 1:, :]

    def _prepend_cls(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.cls_token + self.pos_embed[:, :1, :]
        return torch.cat([cls.expand(x.size(0), -1, -1), x], dim=1)

    def forward_encoder(self, imgs: torch.Tensor, mask_ratio: float, generator=None):

        x = self._embed_patches(imgs)
        x, mask, ids_restore, _ = random_masking(x, mask_ratio, generator)
        x = self._prepend_cls(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x), mask, ids_restore

    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(latent)
        b = x.size(0)
        n_mask = ids_restore.size(1) + 1 - x.size(1)
        mask_tokens = self.mask_token.expand(b, n_mask, -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_.size(-1)))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)[:, 1:, :]

    def recon_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(imgs)
        loss = ((pred - target) ** 2).mean(dim=-1)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def base_loss_single(self, imgs: torch.Tensor, generator=None) -> torch.Tensor:
        latent, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio, generator)
        pred = self.forward_decoder(latent, ids_restore)
        return self.recon_loss(imgs, pred, mask)

    def _encode_to_layer(self, imgs, mask_ratio, layer, generator=None):

        x = self._embed_patches(imgs)
        x_masked, mask, ids_restore, ids_keep = random_masking(x, mask_ratio, generator)
        x = self._prepend_cls(x_masked)
        for i in range(layer):
            x = self.blocks[i](x)
        return x, mask, ids_restore, ids_keep

    def _finish_encoder(self, x, layer):
        for i in range(layer, len(self.blocks)):
            x = self.blocks[i](x)
        return self.norm(x)

    def mfassl_losses(self, batch: dict, t: float, mfa, schedule, layer: int,
                      beta: float = 1.0) -> dict:

        state = schedule.state(t)
        x_l, x_r = batch["mirror"]
        n = self.num_patches

        std_view = batch["standard_crops"][0]
        l_base_std = self.base_loss_single(std_view)

        h_l, mask_l, idr_l, idk_l = self._encode_to_layer(x_l, self.mask_ratio, layer)
        h_r, mask_r, idr_r, idk_r = self._encode_to_layer(x_r, self.mask_ratio, layer)

        npre = self.vit.num_prefix_tokens
        cls_l, vis_l = h_l[:, :npre], h_l[:, npre:]
        cls_r, vis_r = h_r[:, :npre], h_r[:, npre:]

        full_l = scatter_to_full(vis_l, idk_l, n)
        full_r = scatter_to_full(vis_r, idk_r, n)
        both = visible_mask(idk_l, n) & visible_mask(idk_r, n)

        l_mid = mid_layer_consistency_loss(full_l, full_r, valid_mask=both)

        if state["mfa_active"]:
            z_full_l, z_full_r, _ = mfa(full_l, full_r, r_t=state["r_t"], valid_mask=both)
        else:
            z_full_l, z_full_r = full_l, full_r

        new_vis_l = torch.gather(z_full_l, 1, idk_l.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        new_vis_r = torch.gather(z_full_r, 1, idk_r.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        enc_l = self._finish_encoder(torch.cat([cls_l, new_vis_l], dim=1), layer)
        enc_r = self._finish_encoder(torch.cat([cls_r, new_vis_r], dim=1), layer)

        l_eq = reflection_consistency_loss(full_l, full_r, valid_mask=both)

        pred_l = self.forward_decoder(enc_l, idr_l)
        pred_r = self.forward_decoder(enc_r, idr_r)
        l_base_mir = self.recon_loss(x_l, pred_l, mask_l) + self.recon_loss(x_r, pred_r, mask_r)
        l_base = l_base_std + beta * l_base_mir

        total = schedule.total_loss(l_base, l_eq, l_mid, t)
        return {
            "loss": total,
            "l_base": l_base.detach(),
            "l_base_std": l_base_std.detach(),
            "l_base_mir": l_base_mir.detach(),
            "l_eq": l_eq.detach(),
            "l_mid": l_mid.detach(),
            "w": state["w"],
            "r_t": state["r_t"],
            "mfa_active": state["mfa_active"],
            "gate_mean": torch.tensor(0.0),
        }
