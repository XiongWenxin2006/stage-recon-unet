"""Factory for building training stages from configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from stagerecon.training.checkpoint_manager import CheckpointManager
from stagerecon.training.stage_spec import StageSpec
from stagerecon.training.stages.base_stage import BaseStage
from stagerecon.training.stages.bottleneck_decoder_stage import BottleneckDecoderStage
from stagerecon.training.stages.downstream_segmentation_stage import (
    DownstreamSegmentationStage,
)
from stagerecon.training.stages.encoder_bottleneck_stage import EncoderBottleneckStage
from stagerecon.training.stages.full_reconstruction_stage import FullReconstructionStage


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


_STAGE_ALIASES: dict[str, str] = {
    "stage1": "encoder_bottleneck",
    "stage_1": "encoder_bottleneck",
    "encoder_bottleneck": "encoder_bottleneck",
    "encoder-bottleneck": "encoder_bottleneck",
    "stage2": "bottleneck_decoder",
    "stage_2": "bottleneck_decoder",
    "bottleneck_decoder": "bottleneck_decoder",
    "bottleneck-decoder": "bottleneck_decoder",
    "stage3": "full_reconstruction",
    "stage_3": "full_reconstruction",
    "full_reconstruction": "full_reconstruction",
    "full-reconstruction": "full_reconstruction",
    "downstream": "downstream_segmentation",
    "downstream_segmentation": "downstream_segmentation",
    "segmentation": "downstream_segmentation",
}


def build_stage(cfg: Any) -> BaseStage:
    """Build a :class:`BaseStage` from config.

    Accepted shapes::

        {type: stage1, ...stage spec fields...}
        {stage: {type/name: ..., ...}}
        StageSpec instance (type inferred from name when possible)

    Recognized type aliases: ``stage1`` / ``encoder_bottleneck``,
    ``stage2`` / ``bottleneck_decoder``, ``stage3`` / ``full_reconstruction``,
    ``downstream`` / ``downstream_segmentation``.
    """
    if isinstance(cfg, BaseStage):
        return cfg

    if isinstance(cfg, StageSpec):
        kind = _infer_kind_from_name(cfg.name)
        return _make_stage(kind, cfg)

    root = _to_plain_dict(cfg)
    if "stage" in root and isinstance(root["stage"], (Mapping, dict)):
        data = _to_plain_dict(root["stage"])
    else:
        data = root

    kind_raw = data.get("type") or data.get("stage_type") or data.get("kind")
    if kind_raw is None:
        kind_raw = data.get("name", "encoder_bottleneck")
    kind = _STAGE_ALIASES.get(str(kind_raw).lower().strip())
    if kind is None:
        kind = _infer_kind_from_name(str(kind_raw))

    save_dir = data.get("save_dir") or data.get("checkpoint_dir")
    ckpt_mgr = CheckpointManager(save_dir) if save_dir else None

    spec = StageSpec.from_config(data)
    return _make_stage(kind, spec, checkpoint_manager=ckpt_mgr)


def _infer_kind_from_name(name: str) -> str:
    lowered = name.lower().strip()
    if lowered in _STAGE_ALIASES:
        return _STAGE_ALIASES[lowered]
    for key, kind in _STAGE_ALIASES.items():
        if key in lowered:
            return kind
    return "encoder_bottleneck"


def _make_stage(
    kind: str,
    spec: StageSpec,
    *,
    checkpoint_manager: CheckpointManager | None = None,
) -> BaseStage:
    kwargs: dict[str, Any] = {"spec": spec}
    if checkpoint_manager is not None:
        kwargs["checkpoint_manager"] = checkpoint_manager

    if kind == "encoder_bottleneck":
        return EncoderBottleneckStage(**kwargs)
    if kind == "bottleneck_decoder":
        return BottleneckDecoderStage(**kwargs)
    if kind == "full_reconstruction":
        return FullReconstructionStage(**kwargs)
    if kind == "downstream_segmentation":
        return DownstreamSegmentationStage(**kwargs)
    raise ValueError(f"Unknown stage kind '{kind}'.")
