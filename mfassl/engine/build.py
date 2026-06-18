"""Factories that turn a config into a runnable MFASSL pretraining job."""

from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from ..models.mfa import MirrorFusionAttention
from ..losses.ssl.dino import DINO
from ..losses.ssl.moco import MoCoV3
from ..losses.ssl.mae import MAE
from ..data.transforms import MFASSLMultiCrop, collate_mfassl
from .schedules import StagedSchedule, cosine_lr
from .pretrain import MFASSLTrainer

def resolve_device(name: str) -> str:
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name

def _vit_kwargs(cfg) -> dict:
    bk = cfg.backbone
    kw = dict(backbone=bk.name, img_size=bk.img_size, in_chans=bk.get("in_chans", 3),
              num_classes=0)
    if bk.name == "vit_custom":
        kw.update(patch_size=bk.patch_size, embed_dim=bk.embed_dim,
                  depth=bk.depth, num_heads=bk.num_heads)
    return kw

def build_method(cfg):

    if cfg.framework == "dino":
        d = cfg.dino
        return DINO(_vit_kwargs(cfg), out_dim=d.out_dim, n_global=cfg.views.n_global,
                    head_hidden_dim=d.head_hidden_dim,
                    head_bottleneck_dim=d.head_bottleneck_dim,
                    head_nlayers=d.head_nlayers, student_temp=d.student_temp,
                    teacher_temp=d.teacher_temp, center_momentum=d.center_momentum)
    if cfg.framework == "moco":
        m = cfg.get("moco", {})
        return MoCoV3(_vit_kwargs(cfg), proj_dim=m.get("proj_dim", 256),
                      proj_hidden=m.get("proj_hidden", 4096),
                      pred_hidden=m.get("pred_hidden", 4096),
                      temperature=m.get("temperature", 0.2))
    if cfg.framework == "mae":
        a = cfg.get("mae", {})
        return MAE(_vit_kwargs(cfg), decoder_dim=a.get("decoder_dim", 512),
                   decoder_depth=a.get("decoder_depth", 2),
                   decoder_heads=a.get("decoder_heads", 8),
                   mask_ratio=a.get("mask_ratio", 0.75))
    raise NotImplementedError(f"framework '{cfg.framework}' not yet implemented")

def build_mfa(cfg, method) -> MirrorFusionAttention:
    m = cfg.mfa
    return MirrorFusionAttention(
        method.encoder.embed_dim, method.encoder.num_heads, eps=m.eps,
        a_init=m.a_init, b_init=m.b_init, alpha_init=m.alpha_init, gamma_init=m.gamma_init,
    )

def build_base_dataset(cfg) -> Dataset:
    name = cfg.data.name
    if name == "chexpert":
        from ..data.datasets.chexpert import CheXpertDataset
        return CheXpertDataset(cfg.data.root, csv_file=cfg.data.get("csv_file", "train.csv"),
                               img_size=cfg.backbone.img_size,
                               u_policy=cfg.data.get("u_policy", "zeros"),
                               frontal_only=cfg.data.get("frontal_only", True),
                               limit=cfg.data.get("limit"))
    if name == "celeba":
        from ..data.datasets.celeba import CelebAHQDataset
        return CelebAHQDataset(cfg.data.root, img_size=cfg.backbone.img_size,
                               limit=cfg.data.get("limit"))
    if name == "brats_oasis":

        from ..data.datasets.brats import BraTS2DDataset
        from ..data.datasets.oasis import OASIS3Dataset, ConcatBalanced
        brats = BraTS2DDataset(cfg.data.brats_root, img_size=cfg.backbone.img_size,
                               labeled=False, limit=cfg.data.get("limit"))

        oasis = OASIS3Dataset(cfg.data.oasis_root, img_size=cfg.backbone.img_size,
                              in_chans=cfg.backbone.get("in_chans", 4),
                              limit=cfg.data.get("limit"))
        return ConcatBalanced(brats, oasis)
    raise NotImplementedError(f"dataset '{name}' not yet wired for pretraining")

class _PretrainViewDataset(Dataset):

    def __init__(self, base: Dataset, multicrop: MFASSLMultiCrop):
        self.base = base
        self.multicrop = multicrop

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img = self.base[idx][0]
        return self.multicrop(img)

def build_pretrain_loader(cfg, base_dataset) -> DataLoader:
    v = cfg.views
    mc = MFASSLMultiCrop(n_global=v.n_global, n_local=v.n_local, global_size=v.global_size,
                         local_size=v.local_size, mirror_size=v.mirror_size,
                         axis_jitter=v.axis_jitter, augment=v.get("augment", False),
                         pixel_range=v.get("pixel_range", "unit_centered"))
    ds = _PretrainViewDataset(base_dataset, mc)
    return DataLoader(ds, batch_size=cfg.pretrain.batch_size, shuffle=True,
                      num_workers=cfg.data.num_workers, collate_fn=collate_mfassl,
                      drop_last=True)

def build_trainer(cfg, method, mfa) -> Tuple[MFASSLTrainer, torch.optim.Optimizer]:
    sched = StagedSchedule(t_sym=cfg.schedule.t_sym, t_mfa=cfg.schedule.t_mfa,
                           t_gate=cfg.schedule.t_gate, lambda_eq=cfg.objective.lambda_eq,
                           lambda_mid=cfg.objective.lambda_mid)
    trainer = MFASSLTrainer(method, mfa, sched, layer=cfg.mfa.layer,
                            teacher_momentum=cfg.pretrain.teacher_momentum,
                            beta=cfg.objective.get("beta", 1.0),
                            freeze_last_layer_epochs=cfg.pretrain.get(
                                "freeze_last_layer_epochs", 0.0))
    params = [p for p in method.parameters() if p.requires_grad] + list(mfa.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.pretrain.base_lr,
                            weight_decay=cfg.pretrain.weight_decay)
    return trainer, opt

def set_lr(optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr

def load_encoder_from_checkpoint(cfg, checkpoint_path: str, device: str = "cpu"):

    method = build_method(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    method.load_state_dict(ckpt["method"])
    return method.encoder.to(device)
