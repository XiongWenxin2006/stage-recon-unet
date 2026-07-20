"""Synthetic geometric datasets for reconstruction and segmentation smoke tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from stagerecon.data.sample_types import empty_metadata

if TYPE_CHECKING:
    from stagerecon.data.sample_types import ReconstructionSample, SegmentationSample


TransformFn = Callable[[dict[str, Any]], dict[str, Any]]


def _as_hw(image_size: int | Sequence[int]) -> tuple[int, int]:
    """Normalize ``image_size`` to an ``(H, W)`` tuple."""
    if isinstance(image_size, int):
        return int(image_size), int(image_size)
    if len(image_size) != 2:
        raise ValueError(f"image_size must be int or (H, W), got {image_size!r}")
    return int(image_size[0]), int(image_size[1])


def _draw_circle(
    canvas: np.ndarray,
    center_y: int,
    center_x: int,
    radius: int,
    value: float,
) -> None:
    """Rasterize a filled circle onto ``canvas`` (H, W) or (C, H, W)."""
    h = canvas.shape[-2]
    w = canvas.shape[-1]
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
    canvas[..., mask] = value


def _draw_rectangle(
    canvas: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    value: float,
) -> None:
    """Rasterize a filled axis-aligned rectangle onto ``canvas``."""
    canvas[..., y0:y1, x0:x1] = value


class SyntheticDataset(Dataset):
    """Generate simple geometric shapes (circles / rectangles) on a blank canvas.

    Each sample draws one or more shapes with a deterministic RNG seeded by
    ``seed + index``. Images are ``float32`` in ``[0, 1]``. When
    ``return_mask=True``, a binary float32 mask marking shape pixels is also
    produced.

    Args:
        num_samples: Number of samples in the dataset.
        image_size: Spatial size as ``H`` or ``(H, W)``.
        in_channels: Number of image channels.
        num_classes: Kept for API compatibility; masks are binary (foreground=1).
        seed: Base RNG seed.
        return_mask: If ``True``, ``__getitem__`` returns a segmentation sample.
        transform: Optional callable applied to the sample dict.
        max_shapes: Maximum number of shapes drawn per image.
        fill_value: Background intensity (default 0).
        shape_value: Foreground intensity (default 1).
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int | Sequence[int] = 64,
        in_channels: int = 1,
        num_classes: int = 2,
        seed: int = 0,
        return_mask: bool = True,
        transform: TransformFn | None = None,
        max_shapes: int = 3,
        fill_value: float = 0.0,
        shape_value: float = 1.0,
    ) -> None:
        if num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples}")
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if max_shapes < 1:
            raise ValueError(f"max_shapes must be >= 1, got {max_shapes}")

        self.num_samples = int(num_samples)
        self.image_size = _as_hw(image_size)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.return_mask = bool(return_mask)
        self.transform = transform
        self.max_shapes = int(max_shapes)
        self.fill_value = float(fill_value)
        self.shape_value = float(shape_value)

    def __len__(self) -> int:
        return self.num_samples

    def _generate(self, index: int) -> tuple[Tensor, Tensor]:
        """Generate a float32 image and binary mask for ``index``."""
        h, w = self.image_size
        rng = np.random.default_rng(self.seed + int(index))

        image = np.full(
            (self.in_channels, h, w),
            self.fill_value,
            dtype=np.float32,
        )
        mask = np.zeros((1, h, w), dtype=np.float32)

        n_shapes = int(rng.integers(1, self.max_shapes + 1))
        for _ in range(n_shapes):
            shape_type = str(rng.choice(["circle", "rectangle"]))
            intensity = float(
                rng.uniform(max(self.shape_value * 0.5, 0.2), self.shape_value)
            )
            if shape_type == "circle":
                radius = int(rng.integers(max(2, min(h, w) // 16), max(3, min(h, w) // 4)))
                cy = int(rng.integers(radius, max(radius + 1, h - radius)))
                cx = int(rng.integers(radius, max(radius + 1, w - radius)))
                _draw_circle(image, cy, cx, radius, intensity)
                _draw_circle(mask, cy, cx, radius, 1.0)
            else:
                rh = int(rng.integers(max(2, h // 16), max(3, h // 3)))
                rw = int(rng.integers(max(2, w // 16), max(3, w // 3)))
                y0 = int(rng.integers(0, max(1, h - rh)))
                x0 = int(rng.integers(0, max(1, w - rw)))
                _draw_rectangle(image, y0, x0, y0 + rh, x0 + rw, intensity)
                _draw_rectangle(mask, y0, x0, y0 + rh, x0 + rw, 1.0)

        image_t = torch.from_numpy(np.clip(image, 0.0, 1.0))
        mask_t = torch.from_numpy((mask > 0.5).astype(np.float32))
        return image_t, mask_t

    def __getitem__(self, index: int) -> SegmentationSample | dict[str, Any]:
        if index < 0:
            index = self.num_samples + index
        if not 0 <= index < self.num_samples:
            raise IndexError(f"Index {index} out of range for size {self.num_samples}")

        image, mask = self._generate(index)
        sample_id = f"synthetic_{index:06d}"
        metadata = empty_metadata()
        metadata.update(
            {
                "source": "synthetic",
                "seed": self.seed,
                "index": int(index),
                "image_size": list(self.image_size),
                "in_channels": self.in_channels,
                "num_classes": self.num_classes,
            }
        )

        if self.return_mask:
            sample: dict[str, Any] = {
                "image": image,
                "mask": mask,
                "sample_id": sample_id,
                "metadata": metadata,
            }
        else:
            sample = {
                "image": image,
                "target": image.clone(),
                "sample_id": sample_id,
                "metadata": metadata,
            }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample  # type: ignore[return-value]


class SyntheticReconstructionDataset(Dataset):
    """Synthetic reconstruction dataset with optional online corruption.

    Wraps :class:`SyntheticDataset` (without masks) and either:
    - applies a ``corruption`` / ``transform`` callable that receives a sample
      dict whose ``image`` is clean (caller may corrupt a copy), or
    - if ``corruption`` is a callable taking a single tensor, applies it online
      so that ``target`` stays clean and ``image`` becomes the corrupted copy.

    Args:
        num_samples: Number of samples.
        image_size: Spatial size as ``H`` or ``(H, W)``.
        in_channels: Number of image channels.
        seed: Base RNG seed.
        corruption: Optional tensor->tensor corruption or sample transform.
        transform: Optional sample-level transform applied after corruption.
        online_corruption: If ``True`` and ``corruption`` is a tensor callable,
            apply it online inside ``__getitem__``.
        **kwargs: Forwarded to :class:`SyntheticDataset`.
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int | Sequence[int] = 64,
        in_channels: int = 1,
        seed: int = 0,
        corruption: Callable[..., Any] | None = None,
        transform: TransformFn | None = None,
        online_corruption: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("return_mask", None)
        self.base = SyntheticDataset(
            num_samples=num_samples,
            image_size=image_size,
            in_channels=in_channels,
            seed=seed,
            return_mask=False,
            transform=None,
            **kwargs,
        )
        self.corruption = corruption
        self.transform = transform
        self.online_corruption = bool(online_corruption)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> ReconstructionSample:
        sample = dict(self.base[index])
        clean: Tensor = sample["image"]
        # Always keep an untouched clean clone as the reconstruction target.
        target = clean.clone()
        image = clean.clone()

        if self.corruption is not None and self.online_corruption:
            corrupted = self.corruption(image)
            # Support both tensor->tensor corruptions and sample-level callables.
            if isinstance(corrupted, Mapping):
                sample = dict(corrupted)
                image = sample.get("image", image)
                target = sample.get("target", target)
            else:
                image = corrupted

        sample["image"] = image
        sample["target"] = target
        sample.setdefault("metadata", empty_metadata())
        if isinstance(sample["metadata"], dict):
            sample["metadata"] = dict(sample["metadata"])
            sample["metadata"]["task"] = "reconstruction"

        if self.transform is not None:
            sample = self.transform(sample)

        return sample  # type: ignore[return-value]


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


def build_synthetic_dataset(cfg: Mapping[str, Any] | Any) -> Dataset:
    """Build a synthetic dataset for reconstruction or segmentation.

    Selects mode from ``cfg.task`` or ``cfg.mode`` (values such as
    ``reconstruction`` / ``recon`` / ``segmentation`` / ``seg``).

    Args:
        cfg: Dataset configuration mapping (dict or OmegaConf-like).

    Returns:
        A :class:`SyntheticReconstructionDataset` or :class:`SyntheticDataset`.
    """
    plain = _to_plain_dict(cfg)
    task = str(plain.get("task", plain.get("mode", "segmentation"))).lower()
    common = {
        "num_samples": int(plain.get("num_samples", 100)),
        "image_size": plain.get("image_size", plain.get("size", 64)),
        "in_channels": int(plain.get("in_channels", 1)),
        "seed": int(plain.get("seed", 0)),
        "max_shapes": int(plain.get("max_shapes", 3)),
    }
    if task in {"reconstruction", "recon", "pretrain", "restore"}:
        return SyntheticReconstructionDataset(
            **common,
            online_corruption=bool(plain.get("online_corruption", True)),
        )
    return SyntheticDataset(
        **common,
        num_classes=int(plain.get("num_classes", 2)),
        return_mask=bool(plain.get("return_mask", True)),
    )
