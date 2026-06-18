"""ViT backbone wrapper with a mid-layer MFA hook."""

from typing import Optional

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:
    raise ImportError("MFASSLViT requires `timm` (pip install timm)") from exc

def build_vit(
    backbone: str = "vit_base_patch16_224",
    img_size: int = 224,
    in_chans: int = 3,
    num_classes: int = 0,
    drop_path_rate: float = 0.0,
    **kwargs,
):

    if backbone == "vit_custom":
        from timm.models.vision_transformer import VisionTransformer
        return VisionTransformer(
            img_size=img_size, in_chans=in_chans, num_classes=num_classes,
            drop_path_rate=drop_path_rate, dynamic_img_size=True, **kwargs,
        )
    return timm.create_model(
        backbone,
        pretrained=False,
        img_size=img_size,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        dynamic_img_size=True,
        **kwargs,
    )

class MFASSLViT(nn.Module):

    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        img_size: int = 224,
        in_chans: int = 3,
        num_classes: int = 0,
        drop_path_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backbone = build_vit(
            backbone, img_size, in_chans, num_classes, drop_path_rate, **kwargs
        )
        self.embed_dim = self.backbone.embed_dim
        self.num_heads = self.backbone.blocks[0].attn.num_heads
        self.depth = len(self.backbone.blocks)
        self.num_prefix_tokens = self.backbone.num_prefix_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.backbone(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:

        return self.backbone.forward_head(tokens, pre_logits=True)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:

        b = self.backbone
        x = b.patch_embed(x)
        x = b._pos_embed(x)
        x = b.patch_drop(x)
        x = b.norm_pre(x)
        return x

    def _run_blocks(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        for i in range(start, end):
            x = self.backbone.blocks[i](x)
        return x

    def forward_paired(
        self,
        x_l: torch.Tensor,
        x_r: torch.Tensor,
        mfa: Optional[nn.Module] = None,
        layer: int = 8,
        r_t: float = 1.0,
        mfa_active: bool = True,
        valid_mask: Optional[torch.Tensor] = None,
        final_norm: bool = True,
    ) -> dict:

        if not (0 < layer <= self.depth):
            raise ValueError(f"layer must be in (0, {self.depth}], got {layer}")

        npre = self.num_prefix_tokens
        h_l = self._run_blocks(self._embed(x_l), 0, layer)
        h_r = self._run_blocks(self._embed(x_r), 0, layer)

        pre_l, patch_l = h_l[:, :npre], h_l[:, npre:]
        pre_r, patch_r = h_r[:, :npre], h_r[:, npre:]

        gate = None
        if mfa_active and mfa is not None:
            z_l, z_r, gate = mfa(patch_l, patch_r, r_t=r_t, valid_mask=valid_mask)
        else:
            z_l, z_r = patch_l, patch_r

        h_l = torch.cat([pre_l, z_l], dim=1)
        h_r = torch.cat([pre_r, z_r], dim=1)

        h_l = self._run_blocks(h_l, layer, self.depth)
        h_r = self._run_blocks(h_r, layer, self.depth)

        if final_norm:
            h_l = self.backbone.norm(h_l)
            h_r = self.backbone.norm(h_r)

        return {
            "tokens_l": h_l,
            "tokens_r": h_r,
            "prefusion_l": patch_l,
            "prefusion_r": patch_r,
            "gate": gate,
        }
