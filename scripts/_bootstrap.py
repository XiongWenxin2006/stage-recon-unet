"""Shared bootstrap helpers for StageRecon CLI scripts."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Repository root (parent of scripts/)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def ensure_src_on_path() -> Path:
    """Ensure ``src/`` is importable and return the project root."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    return ROOT


def configs_dir() -> Path:
    """Return the absolute path to ``configs/``."""
    return ROOT / "configs"


def run_named_stage(cfg: Any, stage_name: str) -> dict[str, Any]:
    """Run a single stage through :func:`stagerecon.experiments.run_stage`."""
    ensure_src_on_path()
    from stagerecon.experiments import run_stage
    from stagerecon.experiments.seed_manager import prepare_seed
    from stagerecon.utils import setup_logger, validate_config

    setup_logger("stagerecon")
    try:
        validate_config(cfg)
    except ValueError as exc:
        print(f"Config validation warning: {exc}")
    prepare_seed(cfg)
    return run_stage(cfg, stage_name)


def run_pipeline(cfg: Any, stages: Sequence[str] | None = None) -> dict[str, Any]:
    """Run a stage pipeline via :class:`PipelineRunner`."""
    ensure_src_on_path()
    from stagerecon.experiments import PipelineRunner
    from stagerecon.utils import setup_logger, validate_config

    setup_logger("stagerecon")
    try:
        validate_config(cfg)
    except ValueError as exc:
        print(f"Config validation warning: {exc}")
    return PipelineRunner(cfg, stages=stages).run()


def run_experiment(cfg: Any) -> dict[str, Any]:
    """Run a full experiment via :class:`ExperimentRunner`."""
    ensure_src_on_path()
    from stagerecon.experiments import ExperimentRunner

    return ExperimentRunner(cfg).run()


def run_ablation(cfg: Any) -> dict[str, Any]:
    """Run an ablation via :class:`AblationRunner`."""
    ensure_src_on_path()
    from stagerecon.experiments import AblationRunner
    from stagerecon.utils import setup_logger

    setup_logger("stagerecon")
    return AblationRunner(cfg).run()
