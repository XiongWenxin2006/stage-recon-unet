"""Integration tests for WebDataset streaming factory and data_source merge."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

wds = pytest.importorskip("webdataset")

from stagerecon.data import build_dataloader, build_dataset  # noqa: E402
from stagerecon.data.streaming.webdataset_factory import build_webdataset  # noqa: E402
from stagerecon.experiments.config_access import (  # noqa: E402
    get_data_cfg,
    get_dataloader_cfg,
    merge_data_and_source,
)


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    return buf.getvalue()


def _write_split_shards(root: Path, *, split: str, n: int = 4, size: int = 8) -> Path:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(split_dir / f"{split}-%06d.tar")
    with wds.ShardWriter(pattern, maxcount=n + 1) as sink:
        for i in range(n):
            image = np.full((1, size, size), fill_value=(i + 1) * 0.05, dtype=np.float32)
            mask = np.zeros((1, size, size), dtype=np.float32)
            mask[:, i % size, :] = 1.0
            sink.write(
                {
                    "__key__": f"{split}_{i:04d}",
                    "image.npy": _npy_bytes(image),
                    "mask.npy": _npy_bytes(mask),
                    "meta.json": json.dumps({"split": split, "index": i}).encode("utf-8"),
                }
            )
    shards = sorted(split_dir.glob(f"{split}-*.tar"))
    assert shards
    return shards[0].parent


def test_merge_data_and_source_promotes_webdataset():
    merged = merge_data_and_source(
        {"name": "synthetic", "in_channels": 1, "image_size": 64},
        {
            "type": "webdataset",
            "shards": "/tmp/x-{000000..000001}.tar",
            "batch_size": 2,
            "steps_per_epoch": 10,
        },
    )
    assert merged["name"] == "webdataset"
    assert merged["in_channels"] == 1
    assert merged["batch_size"] == 2
    assert merged["steps_per_epoch"] == 10
    assert merged["dataset_name"] == "synthetic"


def test_get_data_cfg_merges_hydra_style_sections():
    cfg = {
        "data": {
            "name": "custom",
            "in_channels": 1,
            "num_classes": 1,
            "transforms": {"reconstruction": {"corruptions": []}},
        },
        "data_source": {
            "type": "webdataset_local",
            "shards": {
                "train": "/data/train-{000000..000001}.tar",
                "val": "/data/val-000000.tar",
            },
            "batch_size": 4,
            "steps_per_epoch": 50,
            "val_steps": 5,
        },
    }
    train_cfg = get_data_cfg(cfg, split="train", task="reconstruction")
    assert train_cfg["name"] == "webdataset"
    assert train_cfg["split"] == "train"
    assert train_cfg["task"] == "reconstruction"
    assert train_cfg["steps_per_epoch"] == 50
    assert "train-{000000..000001}.tar" in str(train_cfg["shards"]["train"])

    loader_cfg = get_dataloader_cfg(cfg, split="train")
    assert loader_cfg["batch_size"] == 4
    assert loader_cfg["shuffle"] is False  # iterable source


def test_build_webdataset_with_cache_and_steps(tmp_path):
    shard_root = _write_split_shards(tmp_path / "shards", split="train", n=4, size=8)
    cache_dir = tmp_path / "cache"
    ds = build_webdataset(
        {
            "shards": str(shard_root / "train-000000.tar"),
            "task": "reconstruction",
            "shuffle_shards": False,
            "sample_shuffle": 0,
            "steps_per_epoch": 3,
            "cache_dir": str(cache_dir),
            "handler": "reraise",
            "length": 3,
        }
    )
    assert cache_dir.is_dir()
    samples = list(iter(ds))
    assert len(samples) == 3
    assert "image" in samples[0] and "target" in samples[0]


def test_build_dataset_and_dataloader_from_streaming_cfg(tmp_path):
    train_dir = _write_split_shards(tmp_path / "data", split="train", n=4, size=8)
    val_dir = _write_split_shards(tmp_path / "data", split="val", n=2, size=8)

    cfg = {
        "name": "webdataset",
        "task": "segmentation",
        "split": "train",
        "shards": {
            "train": str(train_dir / "train-000000.tar"),
            "val": str(val_dir / "val-000000.tar"),
        },
        "shuffle_shards": False,
        "sample_shuffle": 0,
        "steps_per_epoch": 2,
        "handler": "reraise",
        "batch_size": 2,
        "num_workers": 0,
    }
    dataset = build_dataset(cfg)
    loader = build_dataloader(dataset, cfg)
    batch = next(iter(loader))
    assert isinstance(batch["image"], torch.Tensor)
    assert batch["image"].shape[0] == 2
    assert "mask" in batch
