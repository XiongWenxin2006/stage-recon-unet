"""Gradient routing tests for stage1 / stage2 / stage3 prepare + forward."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from stagerecon.models import ModularUNet, build_model
from stagerecon.training import (
    CheckpointManager,
    ParameterController,
    build_stage,
    default_stage1_spec,
    default_stage2_spec,
    default_stage3_spec,
)


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in module.parameters()
        if p.requires_grad
    )


def _any_grad_allocated(module: torch.nn.Module) -> bool:
    return any(p.grad is not None for p in module.parameters())


def _run_recon_backward(model: ModularUNet, x: torch.Tensor) -> None:
    model.zero_grad(set_to_none=True)
    out = model(x, mode="reconstruction")
    loss = out.prediction.float().mean()
    loss.backward()


def test_stage1_specified_modules_get_gradients(
    tiny_model_cfg: dict, sample_input: torch.Tensor, device: torch.device
):
    model = build_model(deepcopy(tiny_model_cfg)).to(device)
    stage = build_stage(default_stage1_spec())
    stage.prepare(model)

    trainable = set(stage.get_spec().trainable_modules)
    _run_recon_backward(model, sample_input)

    for name in trainable:
        module = model.get_module(name)
        assert _has_nonzero_grad(module) or _any_grad_allocated(module), (
            f"Stage1 trainable module '{name}' did not receive gradients"
        )


def test_stage2_encoder_no_gradients(
    tiny_model_cfg: dict,
    sample_input: torch.Tensor,
    device: torch.device,
    ckpt_dir: Path,
):
    # Create a stage1 checkpoint so stage2 can load bottleneck.
    s1 = build_model(deepcopy(tiny_model_cfg)).to(device)
    # Stamp bottleneck with a known fill so load path is exercised.
    with torch.no_grad():
        for p in s1.bottleneck.parameters():
            p.fill_(0.123)
    mgr = CheckpointManager(ckpt_dir)
    s1_path = mgr.save(s1, "stage1_best.pt", stage="stage1")

    model = build_model(deepcopy(tiny_model_cfg)).to(device)
    stage = build_stage(default_stage2_spec(str(s1_path)))
    stage.prepare(model)

    assert all(not p.requires_grad for p in model.encoder.parameters())
    _run_recon_backward(model, sample_input)

    # Frozen encoder must not accumulate gradients.
    leaked = [
        n
        for n, p in model.encoder.named_parameters()
        if p.grad is not None and torch.any(p.grad != 0)
    ]
    assert leaked == [], f"Stage2 frozen encoder received grads: {leaked}"

    # Trainable modules should still get grads.
    for name in ("bottleneck", "decoder", "reconstruction_head"):
        assert _any_grad_allocated(model.get_module(name)), (
            f"Stage2 trainable '{name}' missing grads"
        )


def test_stage3_core_network_gets_gradients(
    tiny_model_cfg: dict,
    sample_input: torch.Tensor,
    device: torch.device,
    ckpt_dir: Path,
):
    mgr = CheckpointManager(ckpt_dir)
    s1 = build_model(deepcopy(tiny_model_cfg)).to(device)
    s2 = build_model(deepcopy(tiny_model_cfg)).to(device)
    s1_path = mgr.save(s1, "stage1_best.pt", stage="stage1")
    s2_path = mgr.save(s2, "stage2_best.pt", stage="stage2")

    model = build_model(deepcopy(tiny_model_cfg)).to(device)
    stage = build_stage(default_stage3_spec(str(s1_path), str(s2_path)))
    stage.prepare(model)

    _run_recon_backward(model, sample_input)

    for name in ("encoder", "bottleneck", "decoder", "reconstruction_head"):
        module = model.get_module(name)
        assert all(p.requires_grad for p in module.parameters())
        assert _any_grad_allocated(module), f"Stage3 '{name}' missing gradients"

    # Optional structural check via ParameterController helper.
    ParameterController.validate_trainable_can_receive_grad(
        model,
        trainable=stage.get_spec().trainable_modules,
        frozen=stage.get_spec().frozen_modules,
        sample_input=sample_input,
        forward_fn=lambda m: m(sample_input, mode="reconstruction"),
    )
