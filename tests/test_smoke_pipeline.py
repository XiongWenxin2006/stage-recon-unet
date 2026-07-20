"""Minimal CPU smoke test for stage1→2→3→downstream via PipelineRunner."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from stagerecon.experiments.pipeline_runner import PipelineRunner


def _tiny_pipeline_cfg(tmp_path: Path) -> dict[str, Any]:
    out = tmp_path / "smoke_out"
    ckpt = out / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    s1 = str(ckpt / "stage1_best.pt")
    s2 = str(ckpt / "stage2_best.pt")
    s3 = str(ckpt / "stage3_best.pt")

    return {
        "seed": 0,
        "device": "cpu",
        "amp": False,
        "experiment": {
            "name": "pytest_smoke",
            "type": "staged_pipeline",
            "stages": ["stage1", "stage2", "stage3", "downstream"],
        },
        "paths": {
            "output_dir": str(out),
            "checkpoint_dir": str(ckpt),
            "stage1_checkpoint": s1,
            "stage2_checkpoint": s2,
            "stage3_checkpoint": s3,
        },
        "data": {
            "name": "synthetic",
            "task": "reconstruction",
            "num_samples": 4,
            "image_size": 32,
            "in_channels": 1,
            "num_classes": 1,
            "spatial_dims": 2,
            "seed": 0,
        },
        "dataloader": {
            "batch_size": 2,
            "num_workers": 0,
            "shuffle": True,
        },
        "model": {
            "name": "unet",
            "in_channels": 1,
            "out_channels": 1,
            "num_classes": 1,
            "spatial_dims": 2,
            "norm": "instance",
            "activation": "relu",
            "encoder": {"name": "unet", "channels": [8, 16]},
            "bottleneck": {"name": "conv"},
            "decoder": {"name": "unet"},
            "heads": {
                "bottleneck_reconstruction": {"name": "bottleneck_reconstruction"},
                "reconstruction": {"name": "image_reconstruction"},
                "segmentation": {"name": "segmentation"},
            },
        },
        "optimizer": {"name": "adam", "lr": 1e-3, "weight_decay": 0.0},
        "scheduler": {"name": "none"},
        "trainer": {
            "epochs": 1,
            "max_epochs": 1,
            "steps_per_epoch": 1,
            "amp": False,
            "device": "cpu",
            "skip_validation": True,
            "no_val": True,
        },
        "stages": {
            "stage1": {
                "name": "stage1",
                "type": "stage1",
                "forward_mode": "reconstruction",
                "module_initialization": {
                    "encoder": {"source": "random"},
                    "bottleneck": {"source": "random"},
                    "decoder": {"source": "random"},
                    "reconstruction_head": {"source": "random"},
                },
                "trainable_modules": [
                    "encoder",
                    "bottleneck",
                    "decoder",
                    "reconstruction_head",
                ],
                "frozen_modules": [
                    "segmentation_head",
                    "bottleneck_reconstruction_head",
                ],
                "checkpoint_output": s1,
                "loss_name": "mse",
                "data_task": "reconstruction",
            },
            "stage2": {
                "name": "stage2",
                "type": "stage2",
                "forward_mode": "reconstruction",
                "module_initialization": {
                    "encoder": {"source": "random"},
                    "bottleneck": {
                        "source": "checkpoint",
                        "checkpoint_path": s1,
                        "source_module": "bottleneck",
                    },
                    "decoder": {"source": "random"},
                    "reconstruction_head": {"source": "random"},
                },
                "trainable_modules": [
                    "bottleneck",
                    "decoder",
                    "reconstruction_head",
                ],
                "frozen_modules": [
                    "encoder",
                    "segmentation_head",
                    "bottleneck_reconstruction_head",
                ],
                "checkpoint_output": s2,
                "loss_name": "mse",
                "data_task": "reconstruction",
            },
            "stage3": {
                "name": "stage3",
                "type": "stage3",
                "forward_mode": "reconstruction",
                "module_initialization": {
                    "encoder": {
                        "source": "checkpoint",
                        "checkpoint_path": s1,
                        "source_module": "encoder",
                    },
                    "bottleneck": {
                        "source": "checkpoint",
                        "checkpoint_path": s2,
                        "source_module": "bottleneck",
                    },
                    "decoder": {
                        "source": "checkpoint",
                        "checkpoint_path": s2,
                        "source_module": "decoder",
                    },
                    "reconstruction_head": {"source": "random"},
                },
                "trainable_modules": [
                    "encoder",
                    "bottleneck",
                    "decoder",
                    "reconstruction_head",
                ],
                "frozen_modules": [
                    "segmentation_head",
                    "bottleneck_reconstruction_head",
                ],
                "checkpoint_output": s3,
                "loss_name": "mse",
                "data_task": "reconstruction",
            },
            "downstream": {
                "name": "downstream",
                "type": "downstream",
                "forward_mode": "segmentation",
                "module_initialization": {
                    "encoder": {
                        "source": "checkpoint",
                        "checkpoint_path": s3,
                        "source_module": "encoder",
                    },
                    "bottleneck": {
                        "source": "checkpoint",
                        "checkpoint_path": s3,
                        "source_module": "bottleneck",
                    },
                    "decoder": {
                        "source": "checkpoint",
                        "checkpoint_path": s3,
                        "source_module": "decoder",
                    },
                    "segmentation_head": {"source": "random"},
                },
                "trainable_modules": [
                    "encoder",
                    "bottleneck",
                    "decoder",
                    "segmentation_head",
                ],
                "frozen_modules": [
                    "reconstruction_head",
                    "bottleneck_reconstruction_head",
                ],
                "checkpoint_output": str(ckpt / "downstream_best.pt"),
                "loss_name": "bce_dice",
                "data_task": "segmentation",
            },
        },
    }


@pytest.mark.slow
def test_smoke_pipeline_stage1_to_downstream(tmp_path):
    """stage1 train+save → stage2 load bottleneck → stage3 assemble → downstream.

    Must complete on CPU without raising. Kept tiny for <60s runtime.
    """
    cfg = _tiny_pipeline_cfg(tmp_path)
    runner = PipelineRunner(deepcopy(cfg), seed=0)
    result = runner.run()

    assert result["stages_order"] == ["stage1", "stage2", "stage3", "downstream"]
    for name in result["stages_order"]:
        stage_res = result["stages"][name]
        assert Path(stage_res["checkpoint"]).is_file(), f"missing ckpt for {name}"
        assert stage_res["stage"] == name
