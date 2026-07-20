"""Ablation experiment helpers (e.g. skip_stage2)."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from stagerecon.experiments.config_access import (
    extract_stage_cfg,
    get_checkpoint_dir,
    get_section,
    normalize_stage_name,
    resolve_path_string,
    to_plain,
)
from stagerecon.experiments.pipeline_runner import PipelineRunner
from stagerecon.experiments.seed_manager import get_seeds, prepare_seed
from stagerecon.utils import setup_logger

logger = logging.getLogger(__name__)


def _path_exists(path: str) -> bool:
    try:
        return Path(path).is_file()
    except Exception:
        return False


def apply_skip_stage2_defaults(cfg: Any) -> Any:
    """Adjust config so stage3 can initialize without a stage2 checkpoint.

    When ``module_initialization`` for stage3 still points at a missing
    stage2 checkpoint, rewrite bottleneck / decoder / reconstruction_head
    sources to load from the stage1 checkpoint (or stay random) so the
    ablation remains runnable.
    """
    root = to_plain(cfg, resolve=False)
    stage3 = extract_stage_cfg(cfg, "stage3")
    init = dict(stage3.get("module_initialization") or {})
    if not init:
        return cfg

    paths = get_section(root, "paths")
    stage1_ckpt = paths.get("stage1_checkpoint") or "stage1_best.pt"
    stage1_ckpt = resolve_path_string(str(stage1_ckpt), cfg)

    changed = False
    for module_name in ("bottleneck", "decoder", "reconstruction_head"):
        mod = dict(init.get(module_name) or {})
        if not mod:
            continue
        source = str(mod.get("source", "random")).lower()
        ckpt = mod.get("checkpoint_path")
        needs_fix = False
        if source == "checkpoint":
            if ckpt is None:
                needs_fix = True
            else:
                resolved = resolve_path_string(str(ckpt), cfg)
                # Heuristic: stage2 references should be remapped for this ablation
                if "stage2" in str(ckpt).lower() or "stage2" in resolved.lower():
                    needs_fix = True
                elif not _path_exists(resolved):
                    # Missing checkpoint that looks stage2-related
                    if "stage2" in resolved.lower() or mod.get("source_module"):
                        needs_fix = True
        if needs_fix:
            if module_name in {"bottleneck"}:
                mod = {
                    "source": "checkpoint",
                    "checkpoint_path": stage1_ckpt,
                    "source_module": mod.get("source_module") or module_name,
                    "strict": bool(mod.get("strict", True)),
                }
            else:
                # Decoder / head: random init when stage2 is skipped
                mod = {"source": "random"}
            init[module_name] = mod
            changed = True

    if not changed:
        return cfg

    # Write back into a mutable plain overlay under experiment.stages_config
    overlay = {
        "experiment": {
            "stages": ["stage1", "stage3", "downstream"],
            "stages_config": {
                "stage3": {
                    **stage3,
                    "module_initialization": init,
                    "type": stage3.get("type", "stage3"),
                    "name": stage3.get("name", "stage3"),
                }
            },
        }
    }
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.merge(cfg, OmegaConf.create(overlay))
    except Exception:
        pass

    merged = deepcopy(root) if isinstance(root, dict) else dict(root)
    exp = dict(merged.get("experiment") or {})
    exp["stages"] = ["stage1", "stage3", "downstream"]
    stages_config = dict(exp.get("stages_config") or {})
    stages_config["stage3"] = overlay["experiment"]["stages_config"]["stage3"]
    exp["stages_config"] = stages_config
    merged["experiment"] = exp
    return merged


class AblationRunner:
    """Run ablation variants such as ``skip_stage2``.

    Supported ablation names (via ``cfg.experiment.ablation`` or
    ``cfg.experiment.type``):

    * ``skip_stage2`` – stages ``[stage1, stage3, downstream]`` with stage3
      init rewritten to avoid stage2 checkpoints when needed.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        ablation: str | None = None,
        stages: Sequence[str] | None = None,
        logger_name: str = "stagerecon.ablation",
    ) -> None:
        self.cfg = cfg
        self.log = setup_logger(logger_name)
        root = to_plain(cfg)
        exp = get_section(root, "experiment")
        self.ablation = (
            ablation
            or exp.get("ablation")
            or exp.get("name")
            or exp.get("type")
            or "skip_stage2"
        )
        self.ablation = str(self.ablation).lower().strip()
        self.stages_override = (
            [normalize_stage_name(s) for s in stages] if stages is not None else None
        )

    def run(self) -> dict[str, Any]:
        """Execute the configured ablation."""
        seeds = get_seeds(self.cfg)
        seed = seeds[0]
        prepare_seed(self.cfg, seed=seed)

        if self.ablation in {
            "skip_stage2",
            "ablation_skip_stage2",
            "no_stage2",
            "ablation",
        }:
            return self._run_skip_stage2(seed=seed)

        self.log.warning(
            "Unknown ablation %r; falling back to skip_stage2 behavior.",
            self.ablation,
        )
        return self._run_skip_stage2(seed=seed)

    def _run_skip_stage2(self, *, seed: int) -> dict[str, Any]:
        self.log.info("Ablation skip_stage2 | seed=%s", seed)
        cfg = apply_skip_stage2_defaults(self.cfg)
        stages = self.stages_override or ["stage1", "stage3", "downstream"]
        # Honour explicit experiment.stages if present after overlay
        exp = get_section(to_plain(cfg), "experiment")
        if self.stages_override is None and exp.get("stages"):
            stages = [normalize_stage_name(s) for s in exp["stages"]]

        # Ensure checkpoint dir exists
        get_checkpoint_dir(cfg).mkdir(parents=True, exist_ok=True)

        result = PipelineRunner(cfg, stages=stages, seed=seed).run()
        result["ablation"] = "skip_stage2"
        return result
