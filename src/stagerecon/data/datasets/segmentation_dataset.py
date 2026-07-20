"""Local filesystem dataset for image / mask segmentation pairs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from stagerecon.data.datasets.reconstruction_dataset import (
    _IMAGE_SUFFIXES,
    _load_image_array,
    array_to_float_tensor,
)
from stagerecon.data.sample_types import SegmentationSample, empty_metadata


TransformFn = Callable[[dict[str, Any]], dict[str, Any]]


def _load_mask_tensor(path: Path) -> Tensor:
    """Load a mask and return a float32 binary/multi-label tensor with channel dim."""
    arr = _load_image_array(path)
    tensor = torch.as_tensor(np.asarray(arr))
    if tensor.dtype == torch.uint8:
        # Common PNG masks use 0/255
        tensor = (tensor.float() > 127.0).float()
    else:
        tensor = tensor.float()
        # Collapse soft masks / label maps into a channel-first float tensor
        if float(tensor.max()) > 1.5 and float(tensor.max()) <= 255.0:
            tensor = (tensor > 127.0).float()
        elif float(tensor.max()) > 1.0:
            # Integer class ids — keep as float labels
            tensor = tensor.float()
        else:
            tensor = (tensor > 0.5).float()

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def _pair_from_directories(
    image_dir: Path,
    mask_dir: Path,
    suffixes: Iterable[str],
) -> list[tuple[Path, Path]]:
    """Pair images and masks that share the same stem."""
    suffix_set = {s.lower() if s.startswith(".") else f".{s.lower()}" for s in suffixes}
    images = {
        p.stem: p
        for p in sorted(image_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in suffix_set
    }
    masks = {
        p.stem: p
        for p in sorted(mask_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in suffix_set
    }
    common = sorted(set(images) & set(masks))
    if not common:
        raise FileNotFoundError(
            f"No matching image/mask stems between {image_dir} and {mask_dir}"
        )
    return [(images[stem], masks[stem]) for stem in common]


def _pair_from_manifest(manifest: str | Path) -> list[tuple[Path, Path, dict[str, Any]]]:
    """Load image/mask pairs from a JSON list or CSV manifest."""
    path = Path(manifest)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    pairs: list[tuple[Path, Path, dict[str, Any]]] = []
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError("JSON manifest must be a list of records")
        for rec in records:
            if not isinstance(rec, Mapping):
                raise ValueError(f"Manifest record must be a mapping, got {type(rec)!r}")
            image_path = Path(str(rec["image"]))
            mask_path = Path(str(rec["mask"]))
            meta = {k: v for k, v in rec.items() if k not in {"image", "mask"}}
            pairs.append((image_path, mask_path, meta))
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV manifest is missing a header row")
            fields = {name.lower(): name for name in reader.fieldnames}
            if "image" not in fields or "mask" not in fields:
                raise ValueError("CSV manifest must include 'image' and 'mask' columns")
            for row in reader:
                image_path = Path(row[fields["image"]])
                mask_path = Path(row[fields["mask"]])
                meta = {
                    k: v
                    for k, v in row.items()
                    if k not in {fields["image"], fields["mask"]}
                }
                pairs.append((image_path, mask_path, meta))

    if not pairs:
        raise FileNotFoundError(f"Manifest contained no pairs: {path}")
    return pairs


class LocalSegmentationDataset(Dataset):
    """Load image + mask pairs from directories or a manifest file.

    Provide either:
    - ``image_dir`` + ``mask_dir`` (paired by filename stem), or
    - ``manifest`` (``.json`` list or ``.csv`` with ``image``/``mask`` columns), or
    - explicit ``pairs`` as ``[(image_path, mask_path), ...]``.

    Args:
        image_dir: Directory of input images.
        mask_dir: Directory of masks with matching stems.
        manifest: Optional JSON/CSV manifest of pairs.
        pairs: Optional explicit list of ``(image, mask)`` paths.
        transform: Optional paired sample transform.
        suffixes: Allowed suffixes when scanning directories.
    """

    def __init__(
        self,
        image_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
        manifest: str | Path | None = None,
        pairs: Sequence[tuple[str | Path, str | Path]] | None = None,
        transform: TransformFn | None = None,
        suffixes: Iterable[str] = _IMAGE_SUFFIXES,
    ) -> None:
        self.transform = transform
        self._pairs: list[tuple[Path, Path, dict[str, Any]]] = []

        if pairs is not None:
            self._pairs.extend((Path(i), Path(m), {}) for i, m in pairs)
        if manifest is not None:
            self._pairs.extend(_pair_from_manifest(manifest))
        if image_dir is not None or mask_dir is not None:
            if image_dir is None or mask_dir is None:
                raise ValueError("Both image_dir and mask_dir are required together")
            for image_path, mask_path in _pair_from_directories(
                Path(image_dir), Path(mask_dir), suffixes=suffixes
            ):
                self._pairs.append((image_path, mask_path, {}))

        if not self._pairs:
            raise ValueError(
                "LocalSegmentationDataset requires pairs, manifest, or image_dir+mask_dir"
            )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> SegmentationSample:
        if index < 0:
            index = len(self) + index
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range for size {len(self)}")

        image_path, mask_path, extra_meta = self._pairs[index]
        image = array_to_float_tensor(_load_image_array(image_path))
        mask = _load_mask_tensor(mask_path)

        sample: dict[str, Any] = {
            "image": image,
            "mask": mask,
            "sample_id": image_path.stem,
            "metadata": {
                **empty_metadata(),
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "source": "local_segmentation",
                **extra_meta,
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample  # type: ignore[return-value]
