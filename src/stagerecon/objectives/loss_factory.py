"""Factory for building loss modules from configuration."""

from __future__ import annotations

from typing import Any, Mapping

from omegaconf import OmegaConf
from torch import nn

from stagerecon.objectives.composite_loss import CompositeLoss
from stagerecon.objectives.reconstruction_losses import (
    CompositeReconstructionLoss,
    L1ReconstructionLoss,
    MSEReconstructionLoss,
)
from stagerecon.objectives.segmentation_losses import (
    BCEDiceLoss,
    BCEWithLogitsSegmentationLoss,
    DiceLoss,
)


def _to_plain(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if OmegaConf.is_config(cfg):
        container = OmegaConf.to_container(cfg, resolve=True)
        return dict(container) if isinstance(container, dict) else {"name": container}
    if isinstance(cfg, str):
        return {"name": cfg}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError(f"Unsupported loss config type: {type(cfg)!r}")


def _build_named_loss(name: str, params: Mapping[str, Any]) -> nn.Module:
    key = name.lower().strip()
    params = dict(params)

    if key in {"l1", "mae", "l1_reconstruction"}:
        return L1ReconstructionLoss(**{k: params[k] for k in ("reduction",) if k in params})

    if key in {"mse", "l2", "mse_reconstruction"}:
        return MSEReconstructionLoss(**{k: params[k] for k in ("reduction",) if k in params})

    if key in {"bce", "bce_logits", "bce_with_logits"}:
        return BCEWithLogitsSegmentationLoss(
            **{k: params[k] for k in ("reduction", "pos_weight") if k in params}
        )

    if key in {"dice", "soft_dice"}:
        kwargs = {k: params[k] for k in ("smooth", "from_logits", "reduction") if k in params}
        return DiceLoss(**kwargs)

    if key in {"bce_dice", "bce+dice", "dice_bce"}:
        kwargs = {
            k: params[k]
            for k in ("bce_weight", "dice_weight", "smooth")
            if k in params
        }
        return BCEDiceLoss(**kwargs)

    if key in {"composite_reconstruction", "recon_composite"}:
        components = params.get("losses", params.get("components"))
        weights = params.get("weights", {})
        if not isinstance(components, Mapping) or not components:
            raise ValueError(
                "composite_reconstruction requires a non-empty 'losses' mapping "
                "of {name: loss_config}."
            )
        built = {
            cname: build_loss(closs) for cname, closs in dict(components).items()
        }
        return CompositeReconstructionLoss(built, weights=dict(weights or {}))

    if key in {"composite", "composite_loss"}:
        components = params.get("losses", params.get("components"))
        weights = params.get("weights", {})
        if not isinstance(components, Mapping) or not components:
            raise ValueError(
                "composite loss requires a non-empty 'losses' mapping "
                "of {name: loss_config}."
            )
        built = {
            cname: build_loss(closs) for cname, closs in dict(components).items()
        }
        return CompositeLoss(built, weights=dict(weights or {}))

    raise ValueError(
        f"Unknown loss name '{name}'. Supported: l1, mse, composite_reconstruction, "
        "bce, dice, bce_dice, composite."
    )


def build_loss(cfg: Any) -> nn.Module:
    """Build a loss ``nn.Module`` from a name string or config mapping.

    Supported names:

    * ``l1`` / ``mae``
    * ``mse`` / ``l2``
    * ``composite_reconstruction`` – requires nested ``losses`` (+ optional ``weights``)
    * ``bce`` / ``bce_with_logits``
    * ``dice``
    * ``bce_dice``
    * ``composite`` – generic weighted composite

    Config shapes accepted::

        "dice"
        {name: dice, smooth: 1.0}
        {loss: {name: bce_dice, bce_weight: 0.5, dice_weight: 0.5}}

    Args:
        cfg: Loss name or configuration.

    Returns:
        Instantiated loss module.

    Raises:
        ValueError: If the loss name is unknown or required fields are missing.
        TypeError: If ``cfg`` has an unsupported type.
    """
    plain = _to_plain(cfg)
    if "loss" in plain and isinstance(plain["loss"], (Mapping, str)):
        plain = _to_plain(plain["loss"])

    name = plain.get("name", plain.get("type", plain.get("loss")))
    if name is None:
        raise ValueError(
            "Loss config must include a 'name' (or 'type') field, or be a name string."
        )

    params = {k: v for k, v in plain.items() if k not in {"name", "type", "loss"}}
    return _build_named_loss(str(name), params)
