"""LR scheduler factory for staged training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "items") and not isinstance(cfg, dict):
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(cfg):
                return dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
        except Exception:
            pass
        return {str(k): v for k, v in cfg.items()}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError(f"Unsupported config type: {type(cfg)!r}")


def build_scheduler(optimizer: Optimizer, cfg: Any) -> LRScheduler | None:
    """Build a learning-rate scheduler from config.

    Supported names: ``cosine``, ``step``, ``none`` / ``null``.

    Config keys (flat or under ``scheduler``)::

        name / type: cosine|step|none
        T_max / epochs: int          # cosine
        eta_min: float               # cosine
        step_size: int               # step
        gamma: float                 # step
    """
    root = _to_plain_dict(cfg)
    if "scheduler" in root and isinstance(root["scheduler"], (Mapping, dict)):
        sch_cfg = _to_plain_dict(root["scheduler"])
    else:
        sch_cfg = root

    if not sch_cfg:
        return None

    name = sch_cfg.get("name", sch_cfg.get("type", "none"))
    if name is None:
        return None
    name = str(name).lower().strip()
    if name in {"", "none", "null", "disabled"}:
        return None

    if name in {"cosine", "cosineannealing", "cosine_annealing"}:
        t_max = int(sch_cfg.get("T_max", sch_cfg.get("epochs", sch_cfg.get("t_max", 100))))
        eta_min = float(sch_cfg.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )

    if name in {"step", "steplr"}:
        step_size = int(sch_cfg.get("step_size", 30))
        gamma = float(sch_cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    raise ValueError(
        f"Unsupported scheduler '{name}'. Expected one of: cosine, step, none."
    )
