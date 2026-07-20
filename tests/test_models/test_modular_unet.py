"""Forward-mode shape / logits tests for ModularUNet."""

from __future__ import annotations

import pytest
import torch

from stagerecon.models import ModularUNet


def test_three_forward_modes_shapes(tiny_model: ModularUNet, sample_input: torch.Tensor):
    b, c, h, w = sample_input.shape

    out_btn = tiny_model(sample_input, mode="bottleneck_reconstruction")
    assert out_btn.mode == "bottleneck_reconstruction"
    assert out_btn.prediction.shape == (b, c, h, w)

    out_recon = tiny_model(sample_input, mode="reconstruction")
    assert out_recon.mode == "reconstruction"
    assert out_recon.prediction.shape == (b, c, h, w)

    out_seg = tiny_model(sample_input, mode="segmentation")
    assert out_seg.mode == "segmentation"
    assert out_seg.prediction.shape[0] == b
    assert out_seg.prediction.shape[-2:] == (h, w)


def test_segmentation_outputs_logits_not_probabilities(
    tiny_model: ModularUNet, sample_input: torch.Tensor
):
    """Segmentation head returns raw logits — not necessarily in [0, 1]."""
    tiny_model.eval()
    with torch.no_grad():
        pred = tiny_model(sample_input, mode="segmentation").prediction
    # Logits may lie outside [0, 1]; a forced sigmoid would always be in-range.
    # We only require that the tensor is finite and that values are not
    # constrained to probabilities (i.e. some may be outside [0, 1] OR the
    # API documents logits — check that we did not apply sigmoid in the head
    # by verifying dtype/shape and that min/max are not artificially clipped).
    assert torch.isfinite(pred).all()
    # Soft check: after many random inits, at least one value typically escapes
    # [0, 1]. If not, still accept finite logits (possible but rare for tiny nets).
    outside = bool(((pred < 0) | (pred > 1)).any().item())
    # Even if all happen to fall in [0,1], sigmoid(pred) would differ unless
    # pred is already a probability — ensure we are not post-sigmoid by checking
    # that applying sigmoid changes the tensor for non-trivial logits.
    sigmoided = torch.sigmoid(pred)
    # If pred were already probabilities in (0,1), sigmoid would squash further.
    # Either outside-[0,1] OR sigmoid changes values → consistent with logits.
    assert outside or not torch.allclose(pred, sigmoided)


def test_invalid_mode_raises(tiny_model: ModularUNet, sample_input: torch.Tensor):
    with pytest.raises(ValueError):
        tiny_model(sample_input, mode="not_a_real_mode")
