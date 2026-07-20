"""Device selection helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import OmegaConf


def _extract_device_spec(cfg: Any) -> str:
    """Pull a device specification string from a config or raw value."""
    if cfg is None:
        return "auto"
    if isinstance(cfg, torch.device):
        return str(cfg)
    if isinstance(cfg, str):
        return cfg
    if OmegaConf.is_config(cfg) or isinstance(cfg, Mapping):
        if OmegaConf.is_config(cfg):
            container = OmegaConf.to_container(cfg, resolve=True)
            mapping = dict(container) if isinstance(container, dict) else {}
        else:
            mapping = dict(cfg)
        for key in ("device", "accelerator", "hardware"):
            if key in mapping and mapping[key] is not None:
                value = mapping[key]
                if isinstance(value, Mapping):
                    return str(value.get("name", value.get("type", value.get("device", "auto"))))
                return str(value)
        return "auto"
    return str(cfg)


def get_device(cfg: Any = "auto") -> torch.device:
    """Resolve a :class:`torch.device` from config or a device string.

    Accepted values:

    * ``"auto"`` – CUDA if available, otherwise CPU
    * ``"cpu"``
    * ``"cuda"`` / ``"cuda:N"``
    * a mapping containing a ``device`` field
    * a full experiment config with a top-level ``device`` key

    Args:
        cfg: Device string, torch.device, or config mapping.

    Returns:
        Resolved ``torch.device``.

    Raises:
        ValueError: If CUDA was requested but is unavailable, or the spec is
            unrecognized.
    """
    spec = _extract_device_spec(cfg).strip().lower()

    if spec in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if spec == "cpu":
        return torch.device("cpu")

    if spec == "cuda" or spec.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(
                f"Requested device '{spec}' but CUDA is not available on this machine."
            )
        return torch.device(spec)

    # Allow bare integer GPU index
    if spec.isdigit():
        if not torch.cuda.is_available():
            raise ValueError(
                f"Requested CUDA device index {spec} but CUDA is not available."
            )
        return torch.device(f"cuda:{spec}")

    raise ValueError(
        f"Unrecognized device specification: {spec!r}. "
        "Expected 'auto', 'cpu', 'cuda', or 'cuda:N'."
    )
