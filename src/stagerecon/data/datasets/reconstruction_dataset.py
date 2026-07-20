"""Local filesystem dataset for image reconstruction / restoration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from stagerecon.data.sample_types import ReconstructionSample, empty_metadata


TransformFn = Callable[[dict[str, Any]], dict[str, Any]]

_IMAGE_SUFFIXES = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _load_image_array(path: Path) -> np.ndarray:
    """Load an image from ``.npy`` or a raster format supported by PIL/numpy."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(str(path))
    else:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Pillow is required to load raster images. Install with: pip install pillow"
            ) from exc
        with Image.open(path) as img:
            arr = np.asarray(img)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        # HWC -> CHW when channels are last and small
        if arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
            arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported image array shape {arr.shape} from {path}")
    return np.asarray(arr)


def array_to_float_tensor(arr: np.ndarray) -> Tensor:
    """Convert a numpy image array to a float32 CHW tensor in ``[0, 1]``."""
    tensor = torch.as_tensor(np.asarray(arr))
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()
        max_val = float(tensor.max()) if tensor.numel() else 0.0
        if max_val > 1.5:
            tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def collect_image_paths(
    root: str | Path | None = None,
    paths: Sequence[str | Path] | None = None,
    suffixes: Iterable[str] = _IMAGE_SUFFIXES,
) -> list[Path]:
    """Collect image paths from an explicit list and/or a directory tree."""
    collected: list[Path] = []
    suffix_set = {s.lower() if s.startswith(".") else f".{s.lower()}" for s in suffixes}

    if paths is not None:
        collected.extend(Path(p) for p in paths)

    if root is not None:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Image root does not exist: {root_path}")
        if root_path.is_file():
            collected.append(root_path)
        else:
            for path in sorted(root_path.rglob("*")):
                if path.is_file() and path.suffix.lower() in suffix_set:
                    collected.append(path)

    # Preserve order while dropping duplicates
    unique: list[Path] = []
    seen: set[str] = set()
    for path in collected:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise FileNotFoundError(
            f"No images found (root={root!r}, paths={None if paths is None else len(paths)})"
        )
    return unique


class LocalReconstructionDataset(Dataset):
    """Load clean images from a directory or path list for reconstruction.

    Corruption is applied via ``transform``, which receives a sample whose
    ``image`` (and ``target``) start as the clean loaded tensor. Typical
    reconstruction transforms clone the clean image into ``target`` and
    corrupt a copy into ``image``.

    Args:
        root: Optional directory to recursively scan for images.
        paths: Optional explicit list of image paths.
        transform: Optional sample transform (receives clean image).
        suffixes: Allowed file suffixes when scanning ``root``.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        paths: Sequence[str | Path] | None = None,
        transform: TransformFn | None = None,
        suffixes: Iterable[str] = _IMAGE_SUFFIXES,
    ) -> None:
        self.paths = collect_image_paths(root=root, paths=paths, suffixes=suffixes)
        self.transform = transform
        self.root = Path(root) if root is not None else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> ReconstructionSample:
        if index < 0:
            index = len(self) + index
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range for size {len(self)}")

        path = self.paths[index]
        arr = _load_image_array(path)
        clean = array_to_float_tensor(arr)
        sample_id = path.stem

        sample: dict[str, Any] = {
            "image": clean.clone(),
            "target": clean.clone(),
            "sample_id": sample_id,
            "metadata": {
                **empty_metadata(),
                "path": str(path),
                "source": "local_reconstruction",
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample  # type: ignore[return-value]
