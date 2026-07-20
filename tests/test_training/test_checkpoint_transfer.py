"""CRITICAL: module-wise checkpoint transfer across stages (no blind full load)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from stagerecon.models import ModularUNet, build_model
from stagerecon.training import CheckpointManager
from stagerecon.training.stage_spec import (
    default_downstream_spec,
    default_stage2_spec,
    default_stage3_spec,
)


def modules_equal(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def _fill_module(module: torch.nn.Module, value: float) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.fill_(value)


def _clone_module_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def test_stage2_loads_only_bottleneck_from_stage1(
    tiny_model_cfg: dict, device: torch.device, ckpt_dir: Path
):
    """Stage2 init: bottleneck equal to stage1; encoder/decoder NOT equal."""
    stage1 = build_model(deepcopy(tiny_model_cfg)).to(device)
    _fill_module(stage1.encoder, 0.11)
    _fill_module(stage1.bottleneck, 0.22)
    _fill_module(stage1.decoder, 0.33)
    _fill_module(stage1.reconstruction_head, 0.44)

    mgr = CheckpointManager(ckpt_dir)
    s1_path = mgr.save(stage1, "stage1_best.pt", stage="stage1")

    # Fresh stage2 model with different random (then filled) weights.
    stage2 = build_model(deepcopy(tiny_model_cfg)).to(device)
    _fill_module(stage2.encoder, 0.91)
    _fill_module(stage2.bottleneck, 0.92)
    _fill_module(stage2.decoder, 0.93)
    _fill_module(stage2.reconstruction_head, 0.94)

    enc_before = _clone_module_state(stage2.encoder)
    dec_before = _clone_module_state(stage2.decoder)
    recon_before = _clone_module_state(stage2.reconstruction_head)

    spec = default_stage2_spec(str(s1_path))
    mgr.initialize_modules(stage2, spec.module_initialization)

    assert modules_equal(stage2.bottleneck, stage1.bottleneck)
    assert not modules_equal(stage2.encoder, stage1.encoder)
    assert not modules_equal(stage2.decoder, stage1.decoder)

    # Encoder / decoder / recon head stayed at their pre-init values (random source).
    assert all(
        torch.equal(enc_before[k], stage2.encoder.state_dict()[k].cpu()) for k in enc_before
    )
    assert all(
        torch.equal(dec_before[k], stage2.decoder.state_dict()[k].cpu()) for k in dec_before
    )
    assert all(
        torch.equal(recon_before[k], stage2.reconstruction_head.state_dict()[k].cpu())
        for k in recon_before
    )


def test_stage3_assembly_from_stage1_and_stage2(
    tiny_model_cfg: dict, device: torch.device, ckpt_dir: Path
):
    """Stage3: encoder←s1, bottleneck+decoder←s2, recon_head random; no blind load."""
    mgr = CheckpointManager(ckpt_dir)

    stage1 = build_model(deepcopy(tiny_model_cfg)).to(device)
    _fill_module(stage1.encoder, 1.0)
    _fill_module(stage1.bottleneck, 1.1)
    _fill_module(stage1.decoder, 1.2)
    s1_path = mgr.save(stage1, "stage1_best.pt", stage="stage1")

    stage2 = build_model(deepcopy(tiny_model_cfg)).to(device)
    _fill_module(stage2.encoder, 2.0)
    _fill_module(stage2.bottleneck, 2.1)
    _fill_module(stage2.decoder, 2.2)
    s2_path = mgr.save(stage2, "stage2_best.pt", stage="stage2")

    stage3 = build_model(deepcopy(tiny_model_cfg)).to(device)
    _fill_module(stage3.encoder, 9.0)
    _fill_module(stage3.bottleneck, 9.1)
    _fill_module(stage3.decoder, 9.2)
    _fill_module(stage3.reconstruction_head, 0.5)
    recon_before = _clone_module_state(stage3.reconstruction_head)

    spec = default_stage3_spec(str(s1_path), str(s2_path))
    mgr.initialize_modules(stage3, spec.module_initialization)

    assert modules_equal(stage3.encoder, stage1.encoder)
    assert modules_equal(stage3.bottleneck, stage2.bottleneck)
    assert modules_equal(stage3.decoder, stage2.decoder)
    # Not taken from stage1 bottleneck/decoder or stage2 encoder.
    assert not modules_equal(stage3.bottleneck, stage1.bottleneck)
    assert not modules_equal(stage3.decoder, stage1.decoder)
    assert not modules_equal(stage3.encoder, stage2.encoder)
    # recon_head remains random (source=random).
    assert all(
        torch.equal(recon_before[k], stage3.reconstruction_head.state_dict()[k].cpu())
        for k in recon_before
    )

    # Verify no full-model blind load: a monolithic state_dict is rejected.
    monolithic = {
        "state_dict": {f"encoder.{k}": v for k, v in stage1.encoder.state_dict().items()}
    }
    bad_path = ckpt_dir / "monolithic.pt"
    torch.save(monolithic, bad_path)
    with pytest.raises(ValueError, match="[Rr]efus|blind|state_dict|Unrecognized"):
        mgr.load_modules(stage3, bad_path, module_names=["encoder"])


def test_downstream_does_not_load_recon_head_into_seg_head(
    tiny_model_cfg: dict, device: torch.device, ckpt_dir: Path
):
    """Downstream: backbone from stage3; seg head stays random (not recon_head)."""
    mgr = CheckpointManager(ckpt_dir)

    stage3 = build_model(deepcopy(tiny_model_cfg)).to(device)
    _fill_module(stage3.encoder, 3.0)
    _fill_module(stage3.bottleneck, 3.1)
    _fill_module(stage3.decoder, 3.2)
    _fill_module(stage3.reconstruction_head, 7.7)
    _fill_module(stage3.segmentation_head, 0.0)
    s3_path = mgr.save(stage3, "stage3_best.pt", stage="stage3")

    downstream = build_model(deepcopy(tiny_model_cfg)).to(device)
    _fill_module(downstream.encoder, 0.0)
    _fill_module(downstream.bottleneck, 0.0)
    _fill_module(downstream.decoder, 0.0)
    _fill_module(downstream.segmentation_head, 0.42)
    _fill_module(downstream.reconstruction_head, 0.0)
    seg_before = _clone_module_state(downstream.segmentation_head)

    spec = default_downstream_spec(str(s3_path))
    mgr.initialize_modules(downstream, spec.module_initialization)

    assert modules_equal(downstream.encoder, stage3.encoder)
    assert modules_equal(downstream.bottleneck, stage3.bottleneck)
    assert modules_equal(downstream.decoder, stage3.decoder)

    # Segmentation head must remain at its random init — never receive recon_head.
    assert all(
        torch.equal(seg_before[k], downstream.segmentation_head.state_dict()[k].cpu())
        for k in seg_before
    )
    assert not modules_equal(downstream.segmentation_head, stage3.reconstruction_head)
    assert not modules_equal(downstream.segmentation_head, stage3.segmentation_head)

    # Explicitly ensure source_module remapping recon→seg is not used by default.
    for mod_name, init in spec.module_initialization.items():
        if mod_name == "segmentation_head":
            assert init.source == "random"
            assert init.source_module is None


def test_load_modules_selective_only(
    tiny_model_cfg: dict, device: torch.device, ckpt_dir: Path
):
    mgr = CheckpointManager(ckpt_dir)
    src = build_model(deepcopy(tiny_model_cfg)).to(device)
    _fill_module(src.bottleneck, 5.5)
    _fill_module(src.encoder, 1.1)
    path = mgr.save(src, "src.pt")

    dst = build_model(deepcopy(tiny_model_cfg)).to(device)
    _fill_module(dst.bottleneck, 0.0)
    _fill_module(dst.encoder, 0.0)
    enc_before = _clone_module_state(dst.encoder)

    mgr.load_modules(dst, path, module_names=["bottleneck"])
    assert modules_equal(dst.bottleneck, src.bottleneck)
    assert all(
        torch.equal(enc_before[k], dst.encoder.state_dict()[k].cpu()) for k in enc_before
    )


def test_initialize_modules_checkpoint_without_path_raises(
    tiny_model: ModularUNet,
):
    """Missing checkpoint_path is rejected (dataclass or initialize_modules)."""
    mgr = CheckpointManager()
    with pytest.raises(ValueError, match="checkpoint_path"):
        # Dict form reaches initialize_modules → ModuleInitializationSpec.from_config
        # which also validates; constructing the dataclass directly raises too.
        mgr.initialize_modules(
            tiny_model,
            {"encoder": {"source": "checkpoint"}},
        )
