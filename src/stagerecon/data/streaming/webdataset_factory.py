"""Factory for WebDataset-based streaming reconstruction / segmentation data."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Mapping

from torch.utils.data import IterableDataset

from stagerecon.data.streaming.error_handlers import warn_and_continue
from stagerecon.data.streaming.sample_decoders import decode_sample_fields
from stagerecon.data.streaming.shard_url_builder import build_shard_urls


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


def _require_webdataset() -> Any:
    """Import webdataset or raise a clear install hint."""
    try:
        import webdataset as wds
    except ImportError as exc:
        raise ImportError(
            "webdataset is required for streaming datasets. "
            "Install it with: pip install webdataset"
        ) from exc
    return wds


def _map_to_task_sample(
    sample: Mapping[str, Any],
    task: str,
) -> dict[str, Any]:
    """Map a decoded shard sample into reconstruction or segmentation format."""
    decoded = decode_sample_fields(dict(sample))
    sample_id = str(decoded.get("sample_id", decoded.get("__key__", "unknown")))
    metadata = dict(decoded.get("metadata") or {})
    metadata.setdefault("source", "webdataset")

    task_l = task.lower()
    if task_l in {"reconstruction", "recon", "pretrain", "restore"}:
        if "image" not in decoded:
            raise KeyError("Reconstruction sample is missing 'image'")
        image = decoded["image"]
        target = decoded.get("target", image)
        return {
            "image": image,
            "target": target.clone() if hasattr(target, "clone") else target,
            "sample_id": sample_id,
            "metadata": metadata,
        }

    if "image" not in decoded or "mask" not in decoded:
        raise KeyError("Segmentation sample requires both 'image' and 'mask'")
    return {
        "image": decoded["image"],
        "mask": decoded["mask"],
        "sample_id": sample_id,
        "metadata": metadata,
    }


class _MappedIterableDataset(IterableDataset):
    """Thin IterableDataset wrapper that exposes length hints when available."""

    def __init__(
        self,
        pipeline: Iterable[Any],
        length: int | None = None,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._length = length

    def __iter__(self) -> Iterator[Any]:
        yield from self._pipeline

    def __len__(self) -> int:
        if self._length is None:
            raise TypeError("WebDataset pipeline has no defined length")
        return int(self._length)


def build_webdataset(cfg: Any) -> IterableDataset:
    """Build a WebDataset iterable dataset from config.

    Expected keys::

        shards / urls / shard_pattern: shard URL pattern(s)
        task / mode: reconstruction | segmentation
        shuffle_shards: bool (default True)
        shard_shuffle: int buffer for shard shuffle (alias)
        sample_shuffle / shuffle: sample shuffle buffer size (0 disables)
        nodesplitter: bool | "node" | "worker" (optional)
        with_epoch: int optional epoch size
        with_steps / steps: optional step count alternative to with_epoch
        handler: "warn_and_continue" (default) | "reraise"
        transforms / transform: optional transform config
        length: optional __len__ hint for progress bars
        seed: optional RNG seed for deterministic shuffle

    If ``webdataset`` is not installed, raises ``ImportError`` with an install hint.

    Args:
        cfg: Streaming dataset configuration.

    Returns:
        An :class:`torch.utils.data.IterableDataset`.
    """
    wds = _require_webdataset()
    plain = _to_plain_dict(cfg)

    urls = build_shard_urls(plain)
    task = str(plain.get("task", plain.get("mode", "segmentation"))).lower()
    seed = plain.get("seed")

    handler_name = str(plain.get("handler", "warn_and_continue")).lower()
    if handler_name in {"warn_and_continue", "warn", "continue"}:
        handler: Callable[[BaseException], bool] = warn_and_continue
    elif handler_name in {"reraise", "raise", "strict"}:
        from stagerecon.data.streaming.error_handlers import reraise

        handler = reraise
    elif callable(plain.get("handler")):
        handler = plain["handler"]
    else:
        raise KeyError(f"Unknown handler '{handler_name}'")

    # Shard shuffle
    shuffle_shards = bool(plain.get("shuffle_shards", True))
    shard_shuffle_buf = int(
        plain.get("shard_shuffle", plain.get("shard_shuffle_buffer", 100))
    )

    # Sample shuffle buffer
    sample_shuffle = int(
        plain.get(
            "sample_shuffle",
            plain.get("shuffle_buffer", plain.get("shuffle", 0)),
        )
    )

    nodesplitter_cfg = plain.get("nodesplitter", plain.get("node_splitter"))
    splitter = None
    if nodesplitter_cfg is True or (
        isinstance(nodesplitter_cfg, str)
        and nodesplitter_cfg.lower() in {"node", "nodesplitter", "true"}
    ):
        splitter = getattr(wds, "shardlists", None)
        if splitter is not None and hasattr(splitter, "split_by_node"):
            splitter = splitter.split_by_node
        elif hasattr(wds, "split_by_node"):
            splitter = wds.split_by_node
        else:
            splitter = None
    elif (
        isinstance(nodesplitter_cfg, str)
        and nodesplitter_cfg.lower() in {"worker", "split_by_worker"}
    ):
        if hasattr(wds, "split_by_worker"):
            splitter = wds.split_by_worker
        else:
            splitter = None
    elif callable(nodesplitter_cfg):
        splitter = nodesplitter_cfg

    # Build pipeline. Prefer WebDataset modern API, with fallbacks.
    dataset = wds.WebDataset(
        urls,
        shardshuffle=shuffle_shards and shard_shuffle_buf > 0,
        handler=handler,
        nodesplitter=splitter,
    )

    if shuffle_shards and hasattr(dataset, "shuffle") and shard_shuffle_buf > 0:
        # Some versions shuffle samples via .shuffle; shardshuffle covers shards.
        pass

    # Decode then map into task sample schema.
    dataset = dataset.map(decode_sample_fields, handler=handler)
    dataset = dataset.map(lambda s: _map_to_task_sample(s, task), handler=handler)

    if sample_shuffle and sample_shuffle > 0:
        if seed is not None and hasattr(dataset, "shuffle"):
            try:
                dataset = dataset.shuffle(sample_shuffle, rng=seed)
            except TypeError:
                dataset = dataset.shuffle(sample_shuffle)
        else:
            dataset = dataset.shuffle(sample_shuffle)

    # Optional transforms
    transform_cfg = plain.get("transforms", plain.get("transform"))
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    if transform_cfg not in (None, False, {}):
        if callable(transform_cfg) and not isinstance(transform_cfg, type):
            transform = transform_cfg  # type: ignore[assignment]
        elif task in {"reconstruction", "recon", "pretrain", "restore"}:
            from stagerecon.data.transforms.transform_factory import (
                build_reconstruction_transforms,
            )

            transform = build_reconstruction_transforms(transform_cfg)
        else:
            from stagerecon.data.transforms.transform_factory import (
                build_segmentation_transforms,
            )

            transform = build_segmentation_transforms(transform_cfg)

    if transform is not None:
        dataset = dataset.map(transform, handler=handler)

    with_epoch = plain.get("with_epoch", plain.get("epoch_size"))
    with_steps = plain.get("with_steps", plain.get("steps"))
    if with_epoch is not None:
        dataset = dataset.with_epoch(int(with_epoch))
    elif with_steps is not None:
        # Prefer with_epoch when available; otherwise approximate via slice.
        if hasattr(dataset, "with_epoch"):
            dataset = dataset.with_epoch(int(with_steps))
        elif hasattr(dataset, "slice"):
            dataset = dataset.slice(int(with_steps))

    length = plain.get("length", with_epoch if with_epoch is not None else with_steps)
    if not isinstance(dataset, IterableDataset):
        dataset = _MappedIterableDataset(dataset, length=length if length is not None else None)
    elif length is not None and not hasattr(dataset, "__len__"):
        dataset = _MappedIterableDataset(dataset, length=int(length))

    return dataset  # type: ignore[return-value]
