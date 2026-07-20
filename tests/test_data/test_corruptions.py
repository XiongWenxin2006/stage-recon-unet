"""Tests for reconstruction corruptions and paired spatial transforms."""

from __future__ import annotations

import pytest
import torch

from stagerecon.data.transforms.paired_segmentation_transforms import (
    PairedCompose,
    PairedRandomHorizontalFlip,
    PairedRandomRotate90,
    PairedRandomVerticalFlip,
)
from stagerecon.data.transforms.reconstruction_corruptions import (
    CorruptionComposer,
    GaussianNoise,
    LocalPixelShuffle,
    RandomPatchMask,
)


def test_gaussian_noise_does_not_modify_original():
    clean = torch.rand(1, 16, 16)
    clean_id = id(clean)
    clean_clone = clean.clone()
    out = GaussianNoise(std=0.2, clip=True)(clean)
    assert id(clean) == clean_id
    assert torch.equal(clean, clean_clone)
    assert out is not clean
    assert not torch.equal(out, clean)


def test_random_patch_mask_does_not_modify_original():
    clean = torch.ones(1, 32, 32)
    clean_id = id(clean)
    clean_clone = clean.clone()
    out = RandomPatchMask(num_patches=3, patch_size=8, mask_value=0.0)(clean)
    assert id(clean) == clean_id
    assert torch.equal(clean, clean_clone)
    assert out is not clean
    assert float(out.min()) == 0.0


def test_local_pixel_shuffle_does_not_modify_original():
    clean = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8) / 63.0
    clean_id = id(clean)
    clean_clone = clean.clone()
    out = LocalPixelShuffle(patch_size=4)(clean)
    assert id(clean) == clean_id
    assert torch.equal(clean, clean_clone)
    assert out is not clean


def test_corruption_composer_target_equals_clean_image_differs():
    clean = torch.rand(1, 24, 24)
    clean_id = id(clean)
    clean_clone = clean.clone()
    composer = CorruptionComposer(
        corruptions=[GaussianNoise(std=0.3, clip=False)],
        p=1.0,
        min_corruptions=1,
        max_corruptions=1,
    )
    sample = {"image": clean, "sample_id": "x", "metadata": {}}
    out = composer(sample)
    assert id(clean) == clean_id
    assert torch.equal(clean, clean_clone)
    assert torch.equal(out["target"], clean_clone)
    assert not torch.equal(out["image"], out["target"])
    assert out["image"] is not clean
    assert out["target"] is not clean


def test_corruption_composer_tensor_call_returns_tensor():
    clean = torch.zeros(1, 8, 8)
    composer = CorruptionComposer(
        corruptions=[GaussianNoise(std=0.1)],
        p=1.0,
        min_corruptions=1,
    )
    out = composer(clean)
    assert isinstance(out, torch.Tensor)
    assert torch.equal(clean, torch.zeros_like(clean))


def test_paired_spatial_transforms_sync_image_and_mask():
    """Spatial flips/rotations must apply the same geometric op to image and mask."""
    image = torch.zeros(1, 8, 8)
    mask = torch.zeros(1, 8, 8)
    image[:, :4, :4] = 1.0
    mask[:, :4, :4] = 1.0

    flip_h = PairedRandomHorizontalFlip(p=1.0)
    flip_v = PairedRandomVerticalFlip(p=1.0)
    rot = PairedRandomRotate90(p=1.0)

    img_h, msk_h = flip_h(image.clone(), mask.clone())
    assert torch.equal(img_h, torch.flip(image, dims=(-1,)))
    assert torch.equal(msk_h, torch.flip(mask, dims=(-1,)))
    assert torch.equal((img_h > 0.5).float(), msk_h)

    img_v, msk_v = flip_v(image.clone(), mask.clone())
    assert torch.equal(img_v, torch.flip(image, dims=(-2,)))
    assert torch.equal(msk_v, torch.flip(mask, dims=(-2,)))

    for _ in range(20):
        img_r, msk_r = rot(image.clone(), mask.clone())
        assert torch.equal((img_r > 0.5).float(), msk_r)


def test_paired_compose_marker_moves_with_flip():
    image = torch.zeros(1, 16, 16)
    mask = torch.zeros(1, 16, 16)
    image[:, 0, 0] = 0.99
    mask[:, 0, 0] = 1.0

    compose = PairedCompose(spatial=[PairedRandomHorizontalFlip(p=1.0)])
    out = compose({"image": image, "mask": mask, "sample_id": "p0"})
    assert float(out["image"][0, 0, -1]) == pytest.approx(0.99)
    assert float(out["mask"][0, 0, -1]) == 1.0
    assert float(out["image"][0, 0, 0]) == pytest.approx(0.0)
    assert float(out["mask"][0, 0, 0]) == 0.0
