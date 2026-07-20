#!/usr/bin/env python3
"""Hydra entry point: run Stage-3 (full reconstruction) pretraining only."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import hydra
from omegaconf import DictConfig

from _bootstrap import run_named_stage


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="experiments/smoke_test",
)
def main(cfg: DictConfig) -> None:
    result = run_named_stage(cfg, "stage3")
    print(result)


if __name__ == "__main__":
    main()
