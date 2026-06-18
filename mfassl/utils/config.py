"""Config loading via OmegaConf with simple base-file inheritance and CLI overrides."""

import os
from typing import List, Optional

from omegaconf import OmegaConf

def load_config(path: str, overrides: Optional[List[str]] = None):

    cfg = OmegaConf.load(path)
    base_rel = cfg.pop("base", None)
    if base_rel is not None:
        base_path = os.path.join(os.path.dirname(path), base_rel)
        base_cfg = load_config(base_path)
        cfg = OmegaConf.merge(base_cfg, cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg
