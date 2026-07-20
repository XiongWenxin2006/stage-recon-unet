"""Seed helpers for single- and multi-seed StageRecon experiments."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from stagerecon.utils.reproducibility import set_seed as _set_seed


def set_seeds(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python / NumPy / PyTorch RNGs and return the applied seed.

    Args:
        seed: Integer seed.
        deterministic: Forwarded to :func:`stagerecon.utils.set_seed`.

    Returns:
        The integer seed that was applied.
    """
    seed = int(seed)
    _set_seed(seed, deterministic=deterministic)
    return seed


def _to_plain(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "items") and not isinstance(cfg, dict):
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(cfg):
                container = OmegaConf.to_container(cfg, resolve=True)
                return dict(container) if isinstance(container, dict) else {}
        except Exception:
            pass
        return {str(k): v for k, v in cfg.items()}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {}


def get_seeds(
    cfg: Any = None,
    *,
    seed: int | None = None,
    num_seeds: int | None = None,
    seeds: Sequence[int] | None = None,
) -> list[int]:
    """Resolve a list of seeds for multi-seed runs.

    Resolution order:

    1. Explicit ``seeds`` argument
    2. ``cfg.experiment.seeds`` / ``cfg.seeds``
    3. ``num_seeds`` (or ``cfg.experiment.num_seeds``) generating
       ``base, base+1, ...`` from ``seed`` / ``cfg.seed`` / ``cfg.experiment.seed``
    4. A single seed from ``seed`` / config (default ``0``)

    Args:
        cfg: Optional experiment config.
        seed: Optional base / single seed override.
        num_seeds: Optional number of consecutive seeds to generate.
        seeds: Explicit seed list override.

    Returns:
        Non-empty list of integer seeds.
    """
    if seeds is not None:
        out = [int(s) for s in seeds]
        if not out:
            raise ValueError("seeds list must be non-empty")
        return out

    root = _to_plain(cfg)
    exp = _to_plain(root.get("experiment"))

    cfg_seeds = exp.get("seeds", root.get("seeds"))
    if cfg_seeds is not None:
        if isinstance(cfg_seeds, (str, bytes)):
            raise TypeError("seeds must be a sequence of integers")
        out = [int(s) for s in cfg_seeds]
        if not out:
            raise ValueError("config seeds list must be non-empty")
        return out

    if num_seeds is None:
        raw_n = exp.get("num_seeds", root.get("num_seeds"))
        num_seeds = int(raw_n) if raw_n is not None else None

    base = seed
    if base is None:
        base = exp.get("seed", root.get("seed", 0))
    base = int(base)

    if num_seeds is not None:
        n = int(num_seeds)
        if n < 1:
            raise ValueError(f"num_seeds must be >= 1, got {n}")
        return [base + i for i in range(n)]

    return [base]


def prepare_seed(cfg: Any = None, *, seed: int | None = None) -> int:
    """Resolve one seed from config, apply it, and return it."""
    seeds = get_seeds(cfg, seed=seed)
    root = _to_plain(cfg)
    exp = _to_plain(root.get("experiment"))
    deterministic = bool(
        exp.get(
            "deterministic",
            root.get("deterministic", False),
        )
    )
    return set_seeds(seeds[0], deterministic=deterministic)
