"""Shared image transform helpers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor


class Identity:
    """No-op transform that returns its input unchanged."""

    def __call__(self, x: Any) -> Any:
        """Return ``x`` unmodified."""
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normalize:
    """Channel-wise normalize a CHW float tensor: ``(x - mean) / std``.

    Args:
        mean: Per-channel mean (scalar or sequence of length C).
        std: Per-channel std (scalar or sequence of length C).
        inplace: If ``True``, modify the input tensor in place.
    """

    def __init__(
        self,
        mean: float | Sequence[float] = 0.0,
        std: float | Sequence[float] = 1.0,
        inplace: bool = False,
    ) -> None:
        self.mean = mean
        self.std = std
        self.inplace = bool(inplace)

    def _as_view(self, values: float | Sequence[float], channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if isinstance(values, (int, float)):
            tensor = torch.full((channels, 1, 1), float(values), device=device, dtype=dtype)
        else:
            seq = list(values)
            if len(seq) == 1 and channels != 1:
                seq = seq * channels
            if len(seq) != channels:
                raise ValueError(
                    f"Expected {channels} channel stats, got {len(seq)} values: {values!r}"
                )
            tensor = torch.tensor(seq, device=device, dtype=dtype).view(channels, 1, 1)
        return tensor

    def __call__(self, image: Tensor) -> Tensor:
        """Normalize a CHW image tensor."""
        if image.ndim != 3:
            raise ValueError(f"Normalize expects CHW tensor, got shape {tuple(image.shape)}")
        out = image if self.inplace else image.clone()
        mean = self._as_view(self.mean, out.shape[0], out.device, out.dtype)
        std = self._as_view(self.std, out.shape[0], out.device, out.dtype)
        std = torch.where(std == 0, torch.ones_like(std), std)
        out.sub_(mean).div_(std)
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean!r}, std={self.std!r})"


def to_tensor(image: Any) -> Tensor:
    """Convert a numpy array / list / tensor to a float32 CHW tensor.

    - ``HW`` arrays become ``1HW``
    - ``HWC`` arrays with small channel dim become ``CHW``
    - uint8 values are scaled to ``[0, 1]``
    """
    if isinstance(image, Tensor):
        tensor = image.detach().clone()
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
            arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.as_tensor(arr)

    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()
    return tensor


class ToTensor:
    """Callable wrapper around :func:`to_tensor`."""

    def __call__(self, image: Any) -> Tensor:
        """Convert ``image`` to a float32 CHW tensor."""
        return to_tensor(image)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
