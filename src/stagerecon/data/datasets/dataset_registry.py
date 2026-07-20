"""Registry and factory for StageRecon datasets."""

from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping

from torch.utils.data import Dataset

from stagerecon.data.datasets.reconstruction_dataset import LocalReconstructionDataset
from stagerecon.data.datasets.segmentation_dataset import LocalSegmentationDataset
from stagerecon.data.datasets.synthetic_dataset import (
    SyntheticDataset,
    SyntheticReconstructionDataset,
    build_synthetic_dataset,
)

DatasetFactory = Callable[..., Dataset]

_DATASET_REGISTRY: dict[str, DatasetFactory] = {}


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


def register_dataset(
    name: str,
    factory: DatasetFactory | None = None,
) -> Callable[[DatasetFactory], DatasetFactory] | DatasetFactory:
    """Register a dataset factory under ``name``."""

    def decorator(cls_or_fn: DatasetFactory) -> DatasetFactory:
        key = name.lower()
        if key in _DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' is already registered.")
        _DATASET_REGISTRY[key] = cls_or_fn
        return cls_or_fn

    if factory is not None:
        return decorator(factory)
    return decorator


def get_dataset(name: str) -> DatasetFactory:
    """Look up a registered dataset factory by name."""
    key = name.lower()
    if key not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return _DATASET_REGISTRY[key]


def list_datasets() -> list[str]:
    """Return sorted registered dataset names."""
    return sorted(_DATASET_REGISTRY.keys())


def _pop_name(section: MutableMapping[str, Any], default: str | None = None) -> str:
    """Extract and remove the ``name`` / ``type`` key from a section dict."""
    name = section.pop("name", section.pop("type", default))
    if name is None:
        raise KeyError("Dataset config must include a 'name' or 'type' field.")
    return str(name)


def _build_transforms_for_task(task: str, transform_cfg: Any) -> Any:
    """Build transforms when a transform config section is present."""
    if transform_cfg is None or transform_cfg is False:
        return None
    plain = _to_plain_dict(transform_cfg)
    if not plain:
        return None

    task_l = task.lower()
    if task_l in {"reconstruction", "recon", "pretrain", "restore"}:
        from stagerecon.data.transforms.transform_factory import (
            build_reconstruction_transforms,
        )

        return build_reconstruction_transforms(plain)
    from stagerecon.data.transforms.transform_factory import (
        build_segmentation_transforms,
    )

    return build_segmentation_transforms(plain)


def _build_synthetic(cfg: Mapping[str, Any]) -> Dataset:
    """Build synthetic dataset; respects task/mode for recon vs seg."""
    plain = dict(cfg)
    transform_cfg = plain.pop("transforms", plain.pop("transform", None))
    task = str(plain.get("task", plain.get("mode", "segmentation"))).lower()
    dataset = build_synthetic_dataset(plain)
    transform = _build_transforms_for_task(task, transform_cfg)
    if transform is not None:
        if isinstance(dataset, SyntheticReconstructionDataset):
            dataset.transform = transform
        elif isinstance(dataset, SyntheticDataset):
            dataset.transform = transform
    return dataset


def _build_local_reconstruction(cfg: Mapping[str, Any]) -> Dataset:
    plain = dict(cfg)
    transform_cfg = plain.pop("transforms", plain.pop("transform", None))
    transform = _build_transforms_for_task("reconstruction", transform_cfg)
    return LocalReconstructionDataset(
        root=plain.get("root", plain.get("image_dir")),
        paths=plain.get("paths"),
        transform=transform,
        suffixes=plain.get("suffixes", (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")),
    )


def _build_local_segmentation(cfg: Mapping[str, Any]) -> Dataset:
    plain = dict(cfg)
    transform_cfg = plain.pop("transforms", plain.pop("transform", None))
    transform = _build_transforms_for_task("segmentation", transform_cfg)
    return LocalSegmentationDataset(
        image_dir=plain.get("image_dir", plain.get("images")),
        mask_dir=plain.get("mask_dir", plain.get("masks")),
        manifest=plain.get("manifest"),
        pairs=plain.get("pairs"),
        transform=transform,
        suffixes=plain.get("suffixes", (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")),
    )


def build_dataset(cfg: Any) -> Dataset:
    """Build a dataset from an OmegaConf/dict configuration.

    Expected structure::

        data:
          name: synthetic          # or local_reconstruction / local_segmentation
          task: segmentation       # or reconstruction
          num_samples: 100
          image_size: 64
          transforms: {...}

    The top-level key may be the data section itself or a parent dict that
    contains a ``data`` / ``dataset`` key. Streaming / webdataset configs with
    ``name: webdataset`` (or ``streaming``) are delegated to
    :func:`stagerecon.data.streaming.webdataset_factory.build_webdataset`.

    Args:
        cfg: Dataset configuration (dict or OmegaConf).

    Returns:
        A PyTorch ``Dataset`` or iterable dataset.
    """
    root = _to_plain_dict(cfg)
    if "data" in root and isinstance(root["data"], (Mapping, dict)):
        data_cfg = _to_plain_dict(root["data"])
    elif "dataset" in root and isinstance(root["dataset"], (Mapping, dict)):
        data_cfg = _to_plain_dict(root["dataset"])
    else:
        data_cfg = root

    name = _pop_name(data_cfg, default="synthetic").lower()

    if name in {"webdataset", "streaming", "wds"}:
        from stagerecon.data.streaming.webdataset_factory import build_webdataset

        return build_webdataset(data_cfg)  # type: ignore[return-value]

    factory = get_dataset(name)
    return factory(data_cfg)


register_dataset("synthetic", _build_synthetic)
register_dataset("local_reconstruction", _build_local_reconstruction)
register_dataset("local_segmentation", _build_local_segmentation)
# Aliases
register_dataset("reconstruction", _build_local_reconstruction)
register_dataset("segmentation", _build_local_segmentation)
