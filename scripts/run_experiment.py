#!/usr/bin/env python3
"""Hydra entry point: full experiment via :class:`ExperimentRunner`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import hydra
from omegaconf import DictConfig

from _bootstrap import run_experiment


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="experiments/smoke_test",
)
def main(cfg: DictConfig) -> None:
    result = run_experiment(cfg)
    print(result)


if __name__ == "__main__":
    main()
