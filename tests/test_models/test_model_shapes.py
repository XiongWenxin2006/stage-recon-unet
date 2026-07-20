"""Encoder / bottleneck / decoder shape contracts for ModularUNet."""

from __future__ import annotations

import torch

from stagerecon.models import ModularUNet, build_model


def test_encoder_returns_correct_number_of_scales(
    tiny_model: ModularUNet, sample_input: torch.Tensor
):
    channels = list(tiny_model.encoder.out_channels)
    features = tiny_model.encoder(sample_input)
    assert len(features) == len(channels)
    h, w = sample_input.shape[-2:]
    for i, (feat, ch) in enumerate(zip(features, channels)):
        scale = 2**i
        assert feat.shape[1] == ch
        assert feat.shape[-2] == h // scale
        assert feat.shape[-1] == w // scale


def test_bottleneck_and_decoder_shapes(
    tiny_model: ModularUNet, sample_input: torch.Tensor
):
    encoder_features = tiny_model.encoder(sample_input)
    deepest = encoder_features[-1]
    btn = tiny_model.bottleneck(deepest)
    assert btn.shape[0] == sample_input.shape[0]
    assert btn.shape[1] == tiny_model.bottleneck.out_channels
    assert btn.shape[-2:] == deepest.shape[-2:]

    decoded = tiny_model.decoder(btn, encoder_features)
    assert decoded.shape[0] == sample_input.shape[0]
    assert decoded.shape[1] == tiny_model.decoder.out_channels
    assert decoded.shape[-2:] == sample_input.shape[-2:]


def test_build_model_three_scale_channels(device: torch.device):
    cfg = {
        "in_channels": 1,
        "out_channels": 1,
        "num_classes": 1,
        "spatial_dims": 2,
        "norm": "instance",
        "encoder": {"name": "unet", "channels": [8, 16, 32]},
        "bottleneck": {"name": "conv"},
        "decoder": {"name": "unet"},
    }
    model = build_model(cfg).to(device)
    x = torch.rand(1, 1, 32, 32, device=device)
    feats = model.encoder(x)
    assert len(feats) == 3
    assert [f.shape[1] for f in feats] == [8, 16, 32]
