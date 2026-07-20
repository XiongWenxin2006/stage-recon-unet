"""Online corruption transforms for reconstruction pretraining."""

from __future__ import annotations

import random
from typing import Any, Callable, Sequence

import torch
from torch import Tensor


CorruptionFn = Callable[[Tensor], Tensor]


class GaussianNoise:
    """Add i.i.d. Gaussian noise and clamp to ``[0, 1]``.

    Args:
        std: Noise standard deviation.
        clip: If ``True``, clamp the result to ``[0, 1]``.
    """

    def __init__(self, std: float = 0.1, clip: bool = True) -> None:
        if std < 0:
            raise ValueError(f"std must be >= 0, got {std}")
        self.std = float(std)
        self.clip = bool(clip)

    def __call__(self, image: Tensor) -> Tensor:
        """Return a new tensor with Gaussian noise applied."""
        out = image.clone()
        if self.std > 0:
            out = out + torch.randn_like(out) * self.std
        if self.clip:
            out = out.clamp(0.0, 1.0)
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std}, clip={self.clip})"


class RandomPatchMask:
    """Zero out (or fill) random square patches.

    Args:
        num_patches: Number of patches to mask.
        patch_size: Side length of each square patch.
        mask_value: Fill value for masked pixels.
    """

    def __init__(
        self,
        num_patches: int = 1,
        patch_size: int = 16,
        mask_value: float = 0.0,
    ) -> None:
        if num_patches < 0:
            raise ValueError(f"num_patches must be >= 0, got {num_patches}")
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        self.num_patches = int(num_patches)
        self.patch_size = int(patch_size)
        self.mask_value = float(mask_value)

    def __call__(self, image: Tensor) -> Tensor:
        """Return a new tensor with random patches masked."""
        if image.ndim != 3:
            raise ValueError(f"Expected CHW tensor, got shape {tuple(image.shape)}")
        out = image.clone()
        _, h, w = out.shape
        ph = min(self.patch_size, h)
        pw = min(self.patch_size, w)
        for _ in range(self.num_patches):
            y0 = int(torch.randint(0, max(h - ph + 1, 1), (1,)).item())
            x0 = int(torch.randint(0, max(w - pw + 1, 1), (1,)).item())
            out[:, y0 : y0 + ph, x0 : x0 + pw] = self.mask_value
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_patches={self.num_patches}, "
            f"patch_size={self.patch_size}, mask_value={self.mask_value})"
        )


class LocalPixelShuffle:
    """Randomly permute pixels inside non-overlapping local patches.

    Args:
        patch_size: Side length of local patches used for shuffling.
    """

    def __init__(self, patch_size: int = 8) -> None:
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        self.patch_size = int(patch_size)

    def __call__(self, image: Tensor) -> Tensor:
        """Return a new tensor with locally shuffled pixels."""
        if image.ndim != 3:
            raise ValueError(f"Expected CHW tensor, got shape {tuple(image.shape)}")
        out = image.clone()
        c, h, w = out.shape
        ps = self.patch_size
        for y0 in range(0, h - ps + 1, ps):
            for x0 in range(0, w - ps + 1, ps):
                patch = out[:, y0 : y0 + ps, x0 : x0 + ps].reshape(c, -1)
                perm = torch.randperm(patch.shape[1], device=patch.device)
                patch = patch[:, perm]
                out[:, y0 : y0 + ps, x0 : x0 + ps] = patch.reshape(c, ps, ps)
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(patch_size={self.patch_size})"


class CorruptionComposer:
    """Compose reconstruction corruptions while preserving a clean target.

    Critical contract:
    - ``target`` is always a clone of the original clean image
    - ``image`` is a corrupted copy
    - the original clean tensor is never modified in-place

    When called on a sample dict, randomly applies a subset of the configured
    corruptions (each kept with probability ``p``). When called on a tensor,
    returns only the corrupted tensor.

    Args:
        corruptions: Sequence of tensor->tensor corruption callables.
        p: Probability of keeping each corruption when sampling a subset.
        min_corruptions: Minimum number of corruptions to apply (0 allowed).
        max_corruptions: Optional cap on number of corruptions applied.
    """

    def __init__(
        self,
        corruptions: Sequence[CorruptionFn] | None = None,
        p: float = 0.5,
        min_corruptions: int = 0,
        max_corruptions: int | None = None,
    ) -> None:
        self.corruptions: list[CorruptionFn] = list(corruptions or [])
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)
        self.min_corruptions = int(min_corruptions)
        self.max_corruptions = max_corruptions if max_corruptions is None else int(max_corruptions)

    def _choose(self) -> list[CorruptionFn]:
        if not self.corruptions:
            return []
        chosen = [c for c in self.corruptions if random.random() < self.p]
        if len(chosen) < self.min_corruptions:
            remaining = [c for c in self.corruptions if c not in chosen]
            need = self.min_corruptions - len(chosen)
            if remaining and need > 0:
                chosen.extend(random.sample(remaining, k=min(need, len(remaining))))
        if self.max_corruptions is not None and len(chosen) > self.max_corruptions:
            chosen = random.sample(chosen, k=self.max_corruptions)
        return chosen

    def corrupt_tensor(self, image: Tensor) -> Tensor:
        """Apply a random corruption subset to a clone of ``image``."""
        # Never mutate the caller's clean tensor.
        out = image.clone()
        for corruption in self._choose():
            out = corruption(out)
            if out is image:
                # Guard against misbehaving corruptions that return the input.
                out = out.clone()
        return out

    def __call__(self, sample_or_image: dict[str, Any] | Tensor) -> dict[str, Any] | Tensor:
        """Corrupt a sample dict or a bare image tensor.

        For dicts, uses ``target`` if present else ``image`` as the clean source.
        """
        if isinstance(sample_or_image, Tensor):
            return self.corrupt_tensor(sample_or_image)

        sample = dict(sample_or_image)
        clean: Tensor = sample.get("target", sample["image"])
        # Freeze the clean target first; corrupt only a separate copy.
        target = clean.clone()
        image = self.corrupt_tensor(clean)
        sample["target"] = target
        sample["image"] = image

        metadata = dict(sample.get("metadata") or {})
        metadata["corruption"] = [type(c).__name__ for c in self.corruptions]
        sample["metadata"] = metadata
        return sample

    def __repr__(self) -> str:
        names = [type(c).__name__ for c in self.corruptions]
        return (
            f"{self.__class__.__name__}(corruptions={names}, p={self.p}, "
            f"min_corruptions={self.min_corruptions}, max_corruptions={self.max_corruptions})"
        )
