"""Decoders for WebDataset ``.npy`` / ``.json`` sample payloads."""

from __future__ import annotations

import io
import json
from typing import Any, Callable, Mapping, MutableMapping

import numpy as np
import torch
from torch import Tensor


def decode_npy_bytes(data: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode a NumPy ``.npy`` payload from raw bytes."""
    with io.BytesIO(bytes(data)) as buf:
        arr = np.load(buf, allow_pickle=False)
    return np.asarray(arr)


def decode_json_bytes(data: bytes | bytearray | memoryview | str) -> Any:
    """Decode a JSON payload from bytes or text."""
    if isinstance(data, str):
        return json.loads(data)
    return json.loads(bytes(data).decode("utf-8"))


def npy_to_float_tensor(arr: np.ndarray) -> Tensor:
    """Convert a decoded numpy array into a float32 CHW-ish tensor."""
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
        arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.as_tensor(arr)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()
        max_val = float(tensor.max()) if tensor.numel() else 0.0
        if max_val > 1.5:
            tensor = tensor / 255.0
    return tensor


def _find_key(sample: Mapping[str, Any], candidates: tuple[str, ...]) -> str | None:
    """Return the first matching key present in ``sample``."""
    for key in candidates:
        if key in sample:
            return key
    # Also allow dotted / nested-style keys that webdataset keeps literally.
    lower_map = {str(k).lower(): k for k in sample.keys()}
    for key in candidates:
        if key.lower() in lower_map:
            return lower_map[key.lower()]
    return None


def decode_sample_fields(sample: MutableMapping[str, Any]) -> dict[str, Any]:
    """Decode known ``.npy`` / ``.json`` fields on a WebDataset sample dict.

    Recognized keys include:
    - ``sample.image.npy`` / ``image.npy`` / ``image``
    - ``sample.mask.npy`` / ``mask.npy`` / ``mask``
    - ``sample.target.npy`` / ``target.npy`` / ``target``
    - ``sample.meta.json`` / ``meta.json`` / ``metadata.json`` / ``json``
    """
    out: dict[str, Any] = dict(sample)

    image_key = _find_key(
        out,
        (
            "sample.image.npy",
            "image.npy",
            "sample.image",
            "image",
            "npy",
        ),
    )
    mask_key = _find_key(
        out,
        (
            "sample.mask.npy",
            "mask.npy",
            "sample.mask",
            "mask",
        ),
    )
    target_key = _find_key(
        out,
        (
            "sample.target.npy",
            "target.npy",
            "sample.target",
            "target",
        ),
    )
    meta_key = _find_key(
        out,
        (
            "sample.meta.json",
            "meta.json",
            "metadata.json",
            "sample.meta",
            "json",
            "meta",
        ),
    )

    if image_key is not None:
        value = out[image_key]
        if isinstance(value, (bytes, bytearray, memoryview)):
            value = decode_npy_bytes(value)
        if isinstance(value, np.ndarray):
            out["image"] = npy_to_float_tensor(value)
        elif isinstance(value, Tensor):
            out["image"] = value.float()
        else:
            out["image"] = value

    if mask_key is not None:
        value = out[mask_key]
        if isinstance(value, (bytes, bytearray, memoryview)):
            value = decode_npy_bytes(value)
        if isinstance(value, np.ndarray):
            tensor = npy_to_float_tensor(value)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            out["mask"] = (tensor > 0.5).float() if float(tensor.max()) <= 1.5 else tensor
        elif isinstance(value, Tensor):
            out["mask"] = value.float()
        else:
            out["mask"] = value

    if target_key is not None:
        value = out[target_key]
        if isinstance(value, (bytes, bytearray, memoryview)):
            value = decode_npy_bytes(value)
        if isinstance(value, np.ndarray):
            out["target"] = npy_to_float_tensor(value)
        elif isinstance(value, Tensor):
            out["target"] = value.float()
        else:
            out["target"] = value

    if meta_key is not None:
        value = out[meta_key]
        if isinstance(value, (bytes, bytearray, memoryview, str)):
            value = decode_json_bytes(value)
        out["metadata"] = value if isinstance(value, dict) else {"raw": value}

    # Prefer __key__ as sample_id when present.
    if "sample_id" not in out:
        key = out.get("__key__") or out.get("key")
        if key is not None:
            out["sample_id"] = str(key)

    out.setdefault("metadata", {})
    if not isinstance(out["metadata"], dict):
        out["metadata"] = {"raw": out["metadata"]}
    return out


def make_webdataset_decoder() -> Callable[[MutableMapping[str, Any]], dict[str, Any]]:
    """Return a WebDataset handler that decodes ``.npy`` / ``.json`` sample keys."""

    def _handler(sample: MutableMapping[str, Any]) -> dict[str, Any]:
        return decode_sample_fields(sample)

    return _handler


# Convenience aliases matching common WebDataset handler naming.
decode_npy = decode_npy_bytes
decode_json = decode_json_bytes
