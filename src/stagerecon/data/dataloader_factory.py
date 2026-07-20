"""DataLoader factory that handles map-style and iterable datasets."""

from __future__ import annotations

from typing import Any, Mapping

from torch.utils.data import DataLoader, Dataset, IterableDataset


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    """Convert OmegaConf / Mapping configs to a plain dict."""
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


def build_dataloader(dataset: Dataset, cfg: Any = None) -> DataLoader:
    """Build a ``DataLoader`` for map-style or iterable datasets.

    For :class:`~torch.utils.data.IterableDataset` instances, ``shuffle=True``
    is ignored at the DataLoader level (shuffling should happen in the
    iterable pipeline / shard buffers instead).

    Supported config keys:
    - ``batch_size`` (default 1)
    - ``num_workers`` (default 0)
    - ``pin_memory`` (default False)
    - ``drop_last`` (default False)
    - ``shuffle`` (map-style only; default False)
    - ``persistent_workers`` (only when ``num_workers > 0``)
    - ``prefetch_factor`` (only when ``num_workers > 0``)
    - ``collate_fn`` (optional callable)
    - ``worker_init_fn`` (optional callable)
    - ``generator`` (optional torch.Generator)

    Args:
        dataset: Dataset or iterable dataset instance.
        cfg: Optional dataloader configuration mapping.

    Returns:
        A configured :class:`torch.utils.data.DataLoader`.
    """
    plain = _to_plain_dict(cfg)
    # Allow nested ``dataloader: {...}`` sections.
    if "dataloader" in plain and isinstance(plain["dataloader"], (Mapping, dict)):
        plain = _to_plain_dict(plain["dataloader"])

    batch_size = int(plain.get("batch_size", 1))
    num_workers = int(plain.get("num_workers", 0))
    pin_memory = bool(plain.get("pin_memory", False))
    drop_last = bool(plain.get("drop_last", False))
    shuffle = bool(plain.get("shuffle", False))

    is_iterable = isinstance(dataset, IterableDataset)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }

    if is_iterable:
        # Iterable datasets manage their own ordering / shard shuffling.
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = shuffle

    if "collate_fn" in plain and plain["collate_fn"] is not None:
        loader_kwargs["collate_fn"] = plain["collate_fn"]
    if "worker_init_fn" in plain and plain["worker_init_fn"] is not None:
        loader_kwargs["worker_init_fn"] = plain["worker_init_fn"]
    if "generator" in plain and plain["generator"] is not None:
        loader_kwargs["generator"] = plain["generator"]

    if num_workers > 0:
        if "persistent_workers" in plain:
            loader_kwargs["persistent_workers"] = bool(plain["persistent_workers"])
        if "prefetch_factor" in plain and plain["prefetch_factor"] is not None:
            loader_kwargs["prefetch_factor"] = int(plain["prefetch_factor"])

    return DataLoader(**loader_kwargs)
