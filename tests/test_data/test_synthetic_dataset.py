"""Tests for synthetic reconstruction / segmentation datasets."""

from __future__ import annotations

import torch

from stagerecon.data.datasets.synthetic_dataset import (
    SyntheticDataset,
    SyntheticReconstructionDataset,
)


def test_segmentation_sample_shapes():
    ds = SyntheticDataset(
        num_samples=4,
        image_size=32,
        in_channels=1,
        seed=0,
        return_mask=True,
    )
    sample = ds[0]
    assert sample["image"].shape == (1, 32, 32)
    assert sample["mask"].shape == (1, 32, 32)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32
    assert "sample_id" in sample
    assert sample["mask"].min() >= 0.0 and sample["mask"].max() <= 1.0


def test_segmentation_non_square_size():
    ds = SyntheticDataset(num_samples=2, image_size=(24, 40), in_channels=1, seed=1)
    sample = ds[0]
    assert sample["image"].shape == (1, 24, 40)
    assert sample["mask"].shape == (1, 24, 40)


def test_reconstruction_sample_keys_and_shapes():
    ds = SyntheticReconstructionDataset(
        num_samples=4,
        image_size=32,
        in_channels=1,
        seed=0,
        corruption=None,
    )
    sample = ds[0]
    assert set(sample.keys()) >= {"image", "target", "sample_id"}
    assert sample["image"].shape == (1, 32, 32)
    assert sample["target"].shape == (1, 32, 32)
    assert isinstance(sample["sample_id"], str)
    # Without corruption, image and target match (clean autoencoder pair).
    assert torch.equal(sample["image"], sample["target"])


def test_reconstruction_online_corruption_keeps_clean_target():
    def _corrupt(x: torch.Tensor) -> torch.Tensor:
        return x + 0.25

    ds = SyntheticReconstructionDataset(
        num_samples=2,
        image_size=16,
        seed=2,
        corruption=_corrupt,
        online_corruption=True,
    )
    sample = ds[0]
    assert not torch.equal(sample["image"], sample["target"])
    assert torch.allclose(sample["image"], sample["target"] + 0.25)
