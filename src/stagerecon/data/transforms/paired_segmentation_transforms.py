"""Paired image/mask transforms for segmentation datasets."""

from __future__ import annotations

import random
from typing import Any, Callable, Sequence

import torch
from torch import Tensor


class PairedRandomHorizontalFlip:
    """Horizontally flip image and mask together with probability ``p``."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def __call__(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Maybe flip both tensors along width."""
        if random.random() < self.p:
            image = torch.flip(image, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        return image, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class PairedRandomVerticalFlip:
    """Vertically flip image and mask together with probability ``p``."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def __call__(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Maybe flip both tensors along height."""
        if random.random() < self.p:
            image = torch.flip(image, dims=(-2,))
            mask = torch.flip(mask, dims=(-2,))
        return image, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class PairedRandomRotate90:
    """Rotate image and mask by ``k * 90`` degrees with probability ``p``."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def __call__(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Maybe rotate both tensors by a random multiple of 90 degrees."""
        if random.random() < self.p:
            k = int(random.randint(0, 3))
            if k:
                image = torch.rot90(image, k=k, dims=(-2, -1))
                mask = torch.rot90(mask, k=k, dims=(-2, -1))
        return image, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomBrightnessContrast:
    """Random brightness / contrast jitter applied to the image only.

    Args:
        brightness: Max absolute brightness shift in ``[-b, b]``.
        contrast: Contrast factor sampled from ``[1-c, 1+c]``.
        p: Probability of applying the transform.
        clip: Clamp result to ``[0, 1]``.
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        p: float = 0.5,
        clip: bool = True,
    ) -> None:
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.p = float(p)
        self.clip = bool(clip)

    def __call__(self, image: Tensor) -> Tensor:
        """Jitter brightness/contrast of ``image`` (returns a new tensor)."""
        if random.random() >= self.p:
            return image
        out = image.clone()
        if self.brightness > 0:
            out = out + random.uniform(-self.brightness, self.brightness)
        if self.contrast > 0:
            factor = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            mean = out.mean(dim=(-2, -1), keepdim=True)
            out = (out - mean) * factor + mean
        if self.clip:
            out = out.clamp(0.0, 1.0)
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(brightness={self.brightness}, "
            f"contrast={self.contrast}, p={self.p})"
        )


SpatialTransform = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]
IntensityTransform = Callable[[Tensor], Tensor]


class PairedCompose:
    """Compose spatial transforms on (image, mask) and intensity on image only.

    Args:
        spatial: Transforms with signature ``(image, mask) -> (image, mask)``.
        intensity: Transforms with signature ``image -> image``.
    """

    def __init__(
        self,
        spatial: Sequence[SpatialTransform] | None = None,
        intensity: Sequence[IntensityTransform] | None = None,
    ) -> None:
        self.spatial: list[SpatialTransform] = list(spatial or [])
        self.intensity: list[IntensityTransform] = list(intensity or [])

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply paired spatial then image-only intensity transforms."""
        out = dict(sample)
        image: Tensor = out["image"]
        mask: Tensor = out["mask"]

        for transform in self.spatial:
            image, mask = transform(image, mask)

        for transform in self.intensity:
            image = transform(image)

        out["image"] = image
        out["mask"] = mask
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(spatial={self.spatial!r}, "
            f"intensity={self.intensity!r})"
        )
