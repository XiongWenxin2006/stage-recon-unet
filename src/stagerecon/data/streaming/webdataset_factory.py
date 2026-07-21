"""Factory for WebDataset-based streaming reconstruction / segmentation data.

Supports local TAR shards, S3/GCS via ``pipe:`` rewriting, HTTP(S) URLs,
shard/sample shuffle buffers, optional disk cache, and fixed epoch lengths
via ``steps_per_epoch`` / ``with_epoch`` so trainers need not call ``len(dataset)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from torch.utils.data import IterableDataset

from stagerecon.data.streaming.error_handlers import warn_and_continue
from stagerecon.data.streaming.sample_decoders import decode_sample_fields
from stagerecon.data.streaming.shard_url_builder import build_shard_urls

logger = logging.getLogger(__name__)


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
        # Prefer an explicit clean target when present; otherwise use the image.
        # Online corruptions are applied later by reconstruction transforms so
        # the clean tensor is preserved as ``target``.
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


def _resolve_handler(plain: Mapping[str, Any]) -> Callable[[BaseException], bool]:
    handler_name = str(plain.get("handler", "warn_and_continue")).lower()
    if handler_name in {"warn_and_continue", "warn", "continue"}:
        return warn_and_continue
    if handler_name in {"reraise", "raise", "strict"}:
        from stagerecon.data.streaming.error_handlers import reraise

        return reraise
    if callable(plain.get("handler")):
        return plain["handler"]  # type: ignore[return-value]
    raise KeyError(f"Unknown handler '{handler_name}'")


def _resolve_nodesplitter(wds: Any, nodesplitter_cfg: Any) -> Any | None:
    if nodesplitter_cfg is None or nodesplitter_cfg is False:
        return None
    if callable(nodesplitter_cfg):
        return nodesplitter_cfg
    if nodesplitter_cfg is True or (
        isinstance(nodesplitter_cfg, str)
        and nodesplitter_cfg.lower() in {"node", "nodesplitter", "true"}
    ):
        splitter = getattr(wds, "shardlists", None)
        if splitter is not None and hasattr(splitter, "split_by_node"):
            return splitter.split_by_node
        if hasattr(wds, "split_by_node"):
            return wds.split_by_node
        return None
    if isinstance(nodesplitter_cfg, str) and nodesplitter_cfg.lower() in {
        "worker",
        "split_by_worker",
    }:
        if hasattr(wds, "split_by_worker"):
            return wds.split_by_worker
        return None
    return None


def _apply_cache(dataset: Any, cache_dir: str | Path | None, wds: Any) -> Any:
    """Attach a local shard/sample cache when supported by the installed webdataset."""
    if cache_dir in (None, "", False):
        return dataset
    cache_path = Path(str(cache_dir)).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)
    logger.info("WebDataset cache directory: %s", cache_path)

    # Prefer modern pipeline APIs when present.
    if hasattr(dataset, "cache"):
        try:
            return dataset.cache(str(cache_path))
        except TypeError:
            try:
                return dataset.cache(cache_path)
            except Exception as exc:  # pragma: no cover - version dependent
                logger.warning("dataset.cache() failed (%s); continuing without cache.", exc)
        except Exception as exc:  # pragma: no cover
            logger.warning("dataset.cache() failed (%s); continuing without cache.", exc)

    # Fallback: some versions expose cached pipelines via constructors only.
    # We still create the directory so users / downstream tools can use it.
    logger.info(
        "Installed webdataset does not expose .cache(); cache_dir=%s was created "
        "but not attached to the pipeline.",
        cache_path,
    )
    return dataset


def _build_transform(
    task: str,
    transform_cfg: Any,
) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    if transform_cfg in (None, False, {}):
        return None
    if callable(transform_cfg) and not isinstance(transform_cfg, type):
        return transform_cfg  # type: ignore[return-value]

    task_l = task.lower()
    if task_l in {"reconstruction", "recon", "pretrain", "restore"}:
        from stagerecon.data.transforms.transform_factory import (
            build_reconstruction_transforms,
        )

        return build_reconstruction_transforms(transform_cfg)

    from stagerecon.data.transforms.transform_factory import (
        build_segmentation_transforms,
    )

    return build_segmentation_transforms(transform_cfg)


def build_webdataset(cfg: Any) -> IterableDataset:
    """Build a WebDataset iterable dataset from config.

    Expected keys::

        shards / urls / shard_pattern: shard URL pattern(s)
        split: optional train|val|test selector for split-specific shards
        task / mode: reconstruction | segmentation
        shuffle_shards / shard_shuffle: shard shuffle enable / buffer
        sample_shuffle / shuffle: sample shuffle buffer size (0 disables)
        cache_dir: optional local cache directory for streamed shards
        nodesplitter: bool | "node" | "worker" (optional)
        steps_per_epoch / with_epoch / with_steps: fixed epoch length
        validation_steps / val_steps: used when split is val (as with_epoch)
        handler: "warn_and_continue" (default) | "reraise"
        transforms / transform: optional transform config
        length: optional __len__ hint for progress bars
        seed: optional RNG seed for deterministic shuffle
        s3_transport / gs_transport / http_transport / rewrite_remote:
            forwarded to :func:`build_shard_urls`
        resampled / repeat: endlessly cycle shards (useful with steps_per_epoch)

    If ``webdataset`` is not installed, raises ``ImportError`` with an install hint.

    Args:
        cfg: Streaming dataset configuration.

    Returns:
        An :class:`torch.utils.data.IterableDataset`.
    """
    wds = _require_webdataset()
    plain = _to_plain_dict(cfg)

    split = plain.get("split")
    urls = build_shard_urls(plain, split=str(split) if split is not None else None)
    logger.info(
        "Building WebDataset with %d shard URL(s) (split=%s, task=%s)",
        len(urls),
        split,
        plain.get("task", plain.get("mode", "segmentation")),
    )
    for preview in urls[:3]:
        logger.info("  shard: %s", preview)
    if len(urls) > 3:
        logger.info("  ... (%d more)", len(urls) - 3)

    task = str(plain.get("task", plain.get("mode", "segmentation"))).lower()
    seed = plain.get("seed")
    handler = _resolve_handler(plain)

    # Shard shuffle
    shuffle_shards = bool(
        plain.get("shuffle_shards", plain.get("shard_shuffle", True))
    )
    # When shard_shuffle is an int, treat it as buffer size and enable shuffle.
    shard_shuffle_raw = plain.get("shard_shuffle", plain.get("shard_shuffle_buffer", 100))
    if isinstance(shard_shuffle_raw, bool):
        shard_shuffle_buf = 100 if shard_shuffle_raw else 0
        shuffle_shards = bool(shard_shuffle_raw)
    else:
        shard_shuffle_buf = int(shard_shuffle_raw)
        if shard_shuffle_buf > 0:
            shuffle_shards = True

    # Sample shuffle buffer
    sample_shuffle = int(
        plain.get(
            "sample_shuffle",
            plain.get("shuffle_buffer", plain.get("shuffle", 0)),
        )
    )
    # Boolean sample_shuffle in YAML means "enable with a default buffer".
    if isinstance(plain.get("sample_shuffle"), bool):
        sample_shuffle = 1000 if plain["sample_shuffle"] else 0

    splitter = _resolve_nodesplitter(
        wds, plain.get("nodesplitter", plain.get("node_splitter"))
    )

    resampled = bool(plain.get("resampled", plain.get("repeat", False)))

    dataset_kwargs: dict[str, Any] = {
        "handler": handler,
    }
    # API differs across webdataset versions; pass only supported kwargs.
    if splitter is not None:
        dataset_kwargs["nodesplitter"] = splitter
    if resampled:
        dataset_kwargs["resampled"] = True

    # shardshuffle: prefer int buffer size for modern webdataset versions.
    shardshuffle_arg: int | bool
    if shuffle_shards and shard_shuffle_buf > 0:
        shardshuffle_arg = shard_shuffle_buf
    else:
        shardshuffle_arg = 0

    try:
        dataset = wds.WebDataset(
            urls,
            shardshuffle=shardshuffle_arg,
            **dataset_kwargs,
        )
    except TypeError:
        dataset_kwargs.pop("resampled", None)
        dataset_kwargs.pop("nodesplitter", None)
        try:
            dataset = wds.WebDataset(urls, shardshuffle=bool(shardshuffle_arg), **dataset_kwargs)
        except TypeError:
            dataset = wds.WebDataset(urls, **dataset_kwargs)
        if splitter is not None and hasattr(dataset, "nodesplitter"):
            dataset = dataset.nodesplitter(splitter)

    dataset = _apply_cache(dataset, plain.get("cache_dir"), wds)

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

    transform = _build_transform(
        task, plain.get("transforms", plain.get("transform"))
    )
    if transform is not None:
        dataset = dataset.map(transform, handler=handler)

    # Epoch length: prefer explicit with_epoch, then steps_per_epoch / val_steps.
    with_epoch = plain.get("with_epoch", plain.get("epoch_size"))
    with_steps = plain.get("with_steps", plain.get("steps"))
    steps_per_epoch = plain.get("steps_per_epoch")
    val_steps = plain.get("val_steps", plain.get("validation_steps"))

    if with_epoch is None:
        split_l = str(split or "").lower()
        if split_l in {"val", "validation", "valid", "test"} and val_steps is not None:
            with_epoch = val_steps
        elif steps_per_epoch is not None:
            with_epoch = steps_per_epoch
        elif with_steps is not None:
            with_epoch = with_steps

    if with_epoch is not None:
        if hasattr(dataset, "with_epoch"):
            dataset = dataset.with_epoch(int(with_epoch))
        elif hasattr(dataset, "slice"):
            dataset = dataset.slice(int(with_epoch))

    length = plain.get("length", with_epoch)
    if not isinstance(dataset, IterableDataset):
        dataset = _MappedIterableDataset(
            dataset, length=int(length) if length is not None else None
        )
    elif length is not None and not hasattr(dataset, "__len__"):
        dataset = _MappedIterableDataset(dataset, length=int(length))

    return dataset  # type: ignore[return-value]
