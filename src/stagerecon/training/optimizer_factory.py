"""Optimizer factory for staged training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch.optim import Optimizer


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


def build_optimizer(
    params: Iterable[torch.nn.Parameter] | list[dict[str, Any]],
    cfg: Any,
) -> Optimizer:
    """Build an optimizer from config.

    Supported names: ``adam``, ``adamw``, ``sgd``.

    Config keys (flat or under ``optimizer``)::

        name / type: adam|adamw|sgd
        lr: float
        weight_decay: float
        betas: [b1, b2]          # adam / adamw
        momentum: float          # sgd
        nesterov: bool           # sgd
    """
    root = _to_plain_dict(cfg)
    if "optimizer" in root and isinstance(root["optimizer"], (Mapping, dict)):
        opt_cfg = _to_plain_dict(root["optimizer"])
    else:
        opt_cfg = root

    name = str(opt_cfg.get("name") or opt_cfg.get("type") or "adam").lower().strip()
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    if name == "adam":
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=betas,  # type: ignore[arg-type]
            weight_decay=weight_decay,
            eps=float(opt_cfg.get("eps", 1e-8)),
        )
    if name == "adamw":
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=betas,  # type: ignore[arg-type]
            weight_decay=weight_decay,
            eps=float(opt_cfg.get("eps", 1e-8)),
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(opt_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(opt_cfg.get("nesterov", False)),
        )
    raise ValueError(
        f"Unsupported optimizer '{name}'. Expected one of: adam, adamw, sgd."
    )
