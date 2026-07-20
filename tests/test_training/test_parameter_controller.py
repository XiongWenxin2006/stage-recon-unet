"""Tests for ParameterController freeze / optimizer exclusion rules."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from stagerecon.models import ModularUNet, build_model
from stagerecon.training import ParameterController, build_optimizer
from stagerecon.training.stage_spec import StageSpec


def test_no_overlap_validation(tiny_model: ModularUNet):
    with pytest.raises(ValueError, match="overlap"):
        ParameterController.apply_trainable_frozen(
            tiny_model,
            trainable=["encoder", "bottleneck"],
            frozen=["bottleneck", "decoder"],
        )


def test_stage_spec_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        StageSpec(
            name="bad",
            forward_mode="reconstruction",
            trainable_modules=["encoder"],
            frozen_modules=["encoder"],
        )


def test_frozen_params_require_grad_false(tiny_model: ModularUNet):
    ParameterController.apply_trainable_frozen(
        tiny_model,
        trainable=["bottleneck", "decoder", "reconstruction_head"],
        frozen=["encoder"],
    )
    for p in tiny_model.encoder.parameters():
        assert p.requires_grad is False
    for p in tiny_model.bottleneck.parameters():
        assert p.requires_grad is True


def test_optimizer_excludes_frozen(tiny_model: ModularUNet):
    ParameterController.apply_trainable_frozen(
        tiny_model,
        trainable=["bottleneck", "decoder", "reconstruction_head"],
        frozen=["encoder", "segmentation_head", "bottleneck_reconstruction_head"],
    )
    groups = ParameterController.get_trainable_param_groups(tiny_model, lr=1e-3)
    opt = build_optimizer(groups, {"name": "adam", "lr": 1e-3})
    ParameterController.validate_optimizer_excludes_frozen(opt)

    trainable_ids = {id(p) for p in ParameterController.iter_trainable_parameters(tiny_model)}
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert opt_ids == trainable_ids
    for p in tiny_model.encoder.parameters():
        assert id(p) not in opt_ids


def test_validate_optimizer_raises_when_frozen_included(
    tiny_model_cfg: dict, device: torch.device
):
    model = build_model(deepcopy(tiny_model_cfg)).to(device)
    ParameterController.freeze_modules(model, ["encoder"])
    # Intentionally include all params (including frozen) in the optimizer.
    bad_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    with pytest.raises(RuntimeError, match="frozen"):
        ParameterController.validate_optimizer_excludes_frozen(bad_opt)
