#!/usr/bin/env python3
"""Hydra entry point: evaluate a checkpoint with :class:`ExperimentRunner`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import hydra
from omegaconf import DictConfig, OmegaConf

from _bootstrap import ensure_src_on_path


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="experiments/smoke_test",
)
def main(cfg: DictConfig) -> None:
    ensure_src_on_path()
    from stagerecon.experiments import ExperimentRunner

    # Force evaluation mode without mutating caller configs permanently
    OmegaConf.set_struct(cfg, False)
    if "experiment" not in cfg or cfg.experiment is None:
        cfg.experiment = OmegaConf.create({})
    cfg.experiment.type = "evaluate"
    result = ExperimentRunner(cfg).run()
    print(result)


if __name__ == "__main__":
    main()
