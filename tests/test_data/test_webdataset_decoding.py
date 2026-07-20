"""Tests for WebDataset shard writing / decoding (skipped without webdataset)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

wds = pytest.importorskip("webdataset")

from stagerecon.data.streaming.sample_decoders import (  # noqa: E402
    decode_sample_fields,
    make_webdataset_decoder,
)
from stagerecon.data.streaming.webdataset_factory import build_webdataset  # noqa: E402


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    return buf.getvalue()


def _write_tiny_shard(shard_dir: Path, n: int = 3, size: int = 8) -> Path:
    """Write a tiny tar shard with image.npy + mask.npy + meta.json keys."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(shard_dir / "shard-%06d.tar")
    with wds.ShardWriter(pattern, maxcount=n + 1) as sink:
        for i in range(n):
            image = np.full((1, size, size), fill_value=(i + 1) * 0.1, dtype=np.float32)
            mask = np.zeros((1, size, size), dtype=np.float32)
            mask[:, : size // 2, : size // 2] = 1.0
            meta = {"index": i, "split": "train", "source": "unit_test"}
            sink.write(
                {
                    "__key__": f"sample_{i:04d}",
                    "image.npy": _npy_bytes(image),
                    "mask.npy": _npy_bytes(mask),
                    "meta.json": json.dumps(meta).encode("utf-8"),
                }
            )
    shards = sorted(shard_dir.glob("shard-*.tar"))
    assert shards, f"ShardWriter produced no tar under {shard_dir}"
    return shards[0]


def test_decode_sample_fields_from_raw_bytes():
    image = np.random.randn(1, 4, 4).astype(np.float32)
    mask = (np.random.rand(1, 4, 4) > 0.5).astype(np.float32)
    meta = {"foo": "bar"}
    raw = {
        "__key__": "abc",
        "image.npy": _npy_bytes(image),
        "mask.npy": _npy_bytes(mask),
        "meta.json": json.dumps(meta).encode("utf-8"),
    }
    decoded = decode_sample_fields(raw)
    assert isinstance(decoded["image"], torch.Tensor)
    assert decoded["image"].shape == (1, 4, 4)
    assert isinstance(decoded["mask"], torch.Tensor)
    assert decoded["mask"].shape == (1, 4, 4)
    assert decoded["sample_id"] == "abc"
    assert decoded["metadata"]["foo"] == "bar"


def test_make_webdataset_decoder_alias():
    decoder = make_webdataset_decoder()
    image = np.ones((2, 2), dtype=np.float32)
    out = decoder({"__key__": "k0", "image.npy": _npy_bytes(image)})
    assert out["image"].shape[0] == 1  # HWC/HW promoted to CHW
    assert out["sample_id"] == "k0"


def test_build_webdataset_reads_tiny_shard(tmp_path):
    shard = _write_tiny_shard(tmp_path / "shards", n=3, size=8)

    ds = build_webdataset(
        {
            "shards": str(shard),
            "task": "segmentation",
            "shuffle_shards": False,
            "sample_shuffle": 0,
            "with_epoch": 3,
            "length": 3,
            "handler": "reraise",
        }
    )
    samples = list(iter(ds))
    assert len(samples) == 3
    s0 = samples[0]
    assert "image" in s0 and "mask" in s0 and "sample_id" in s0
    assert s0["image"].ndim == 3
    assert s0["mask"].shape[-2:] == s0["image"].shape[-2:]


def test_build_webdataset_reconstruction_task(tmp_path):
    shard = _write_tiny_shard(tmp_path / "recon_shards", n=2, size=8)
    ds = build_webdataset(
        {
            "urls": str(shard),
            "task": "reconstruction",
            "shuffle_shards": False,
            "sample_shuffle": 0,
            "with_epoch": 2,
            "handler": "reraise",
        }
    )
    sample = next(iter(ds))
    assert "image" in sample and "target" in sample
    assert "sample_id" in sample
