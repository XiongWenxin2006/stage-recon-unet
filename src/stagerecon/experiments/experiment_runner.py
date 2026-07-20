"""Top-level experiment orchestration for StageRecon."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stagerecon.experiments.config_access import (
    get_checkpoint_dir,
    get_data_cfg,
    get_dataloader_cfg,
    get_model_cfg,
    get_output_dir,
    normalize_stage_name,
    resolve_stages_list,
    to_plain,
)
from stagerecon.experiments.pipeline_runner import PipelineRunner, run_stage
from stagerecon.experiments.result_aggregator import ResultAggregator
from stagerecon.experiments.seed_manager import get_seeds, prepare_seed
from stagerecon.utils import get_device, setup_logger, validate_config

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Load / validate a config and run a pipeline or single stage.

    Behavior is driven by ``cfg.experiment.type``:

    * ``pipeline`` / ``full`` / ``pretrain`` – :class:`PipelineRunner`
    * ``stage`` / ``single`` / ``stage1``… – single stage via :func:`run_stage`
    * ``evaluate`` – run evaluation only (no training)
    * ``ablation`` / ``skip_stage2`` – delegated to :class:`AblationRunner`
    """

    def __init__(
        self,
        cfg: Any,
        *,
        validate: bool = True,
        logger_name: str = "stagerecon.experiment",
    ) -> None:
        self.cfg = cfg
        self.log = setup_logger(logger_name)
        if validate:
            try:
                validate_config(cfg)
            except ValueError as exc:
                # Soft-fail on incomplete smoke configs: log and continue
                self.log.warning("Config validation warning: %s", exc)

        root = to_plain(cfg)
        get_output_dir(cfg).mkdir(parents=True, exist_ok=True)
        get_checkpoint_dir(cfg).mkdir(parents=True, exist_ok=True)
        self.exp = to_plain(root.get("experiment"))
        self.exp_type = str(
            self.exp.get("type", root.get("type", "pipeline"))
        ).lower().strip()

    def run(self) -> dict[str, Any]:
        """Execute the experiment described by the config."""
        seeds = get_seeds(self.cfg)
        multi_seed = len(seeds) > 1 or bool(self.exp.get("multi_seed"))

        if self.exp_type in {"ablation", "skip_stage2", "ablation_skip_stage2"}:
            from stagerecon.experiments.ablation_runner import AblationRunner

            return AblationRunner(self.cfg).run()

        if self.exp_type in {"evaluate", "eval", "evaluation"}:
            return self._run_evaluation(seed=seeds[0])

        if multi_seed and self.exp_type not in {"evaluate", "eval"}:
            return self._run_multi_seed(seeds)

        return self._run_once(seeds[0])

    def _run_once(self, seed: int) -> dict[str, Any]:
        prepare_seed(self.cfg, seed=seed)
        self.log.info("Experiment type=%s seed=%s", self.exp_type, seed)

        if self.exp_type in {
            "pipeline",
            "full",
            "full_pipeline",
            "end_to_end",
            "staged_pipeline",
            "staged_pretrain",
            "pretrain",
            "pretrain_pipeline",
            "pretraining",
        }:
            result = PipelineRunner(self.cfg, seed=seed).run()
        elif self.exp_type in {
            "stage",
            "single",
            "single_stage",
            "stage1",
            "stage2",
            "stage3",
            "downstream",
            "segmentation",
            "pretrain_stage1",
            "pretrain_stage2",
            "pretrain_stage3",
        }:
            stages = resolve_stages_list(self.cfg)
            if self.exp_type not in {"stage", "single", "single_stage"}:
                stages = [normalize_stage_name(self.exp_type)]
            if not stages:
                stages = ["stage1"]
            if len(stages) == 1:
                result = {
                    "stages": {stages[0]: run_stage(self.cfg, stages[0], seed=seed)},
                    "seed": seed,
                    "stages_order": stages,
                }
            else:
                result = PipelineRunner(self.cfg, stages=stages, seed=seed).run()
        else:
            # Unknown type: treat as pipeline with whatever stages are listed
            self.log.warning(
                "Unrecognized experiment.type=%r; running PipelineRunner.",
                self.exp_type,
            )
            result = PipelineRunner(self.cfg, seed=seed).run()

        self._maybe_save_result(result, seed=seed)
        return result

    def _run_multi_seed(self, seeds: Sequence[int]) -> dict[str, Any]:
        agg = ResultAggregator()
        per_seed: dict[str, Any] = {}
        for seed in seeds:
            self.log.info("=== Multi-seed run seed=%s ===", seed)
            result = self._run_once(int(seed))
            per_seed[str(seed)] = result
            # Flatten last-stage best_metric for aggregation when available
            metrics: dict[str, Any] = {"seed": int(seed)}
            stages = result.get("stages") or {}
            for stage_name, stage_result in stages.items():
                best = stage_result.get("best_metric")
                if best is not None:
                    metrics[f"{stage_name}_best_metric"] = float(best)
            agg.add(f"seed_{seed}", metrics)

        summary = agg.aggregate()
        out_dir = get_output_dir(self.cfg) / "multi_seed"
        out_dir.mkdir(parents=True, exist_ok=True)
        agg.save_json(out_dir / "summary.json")
        agg.save_csv(out_dir / "summary.csv")
        return {
            "multi_seed": True,
            "seeds": list(seeds),
            "runs": per_seed,
            "aggregate": summary,
        }

    def _run_evaluation(self, *, seed: int) -> dict[str, Any]:
        """Evaluate a checkpoint without training."""
        from stagerecon.data import build_dataloader, build_dataset
        from stagerecon.evaluation import Evaluator
        from stagerecon.models import build_model
        from stagerecon.training import CheckpointManager

        prepare_seed(self.cfg, seed=seed)
        root = to_plain(self.cfg)
        eval_cfg = to_plain(root.get("evaluation") or root.get("eval") or {})
        device = get_device(root.get("device", eval_cfg.get("device", "auto")))

        model = build_model(get_model_cfg(self.cfg))
        ckpt = (
            eval_cfg.get("checkpoint")
            or eval_cfg.get("checkpoint_path")
            or root.get("checkpoint")
        )
        if ckpt:
            from stagerecon.experiments.config_access import resolve_path_string

            ckpt_path = resolve_path_string(str(ckpt), self.cfg)
            mgr = CheckpointManager(get_checkpoint_dir(self.cfg))
            # Load all modules present in the checkpoint
            payload = mgr._read_checkpoint(ckpt_path)
            modules = mgr.extract_modules_dict(payload)
            mgr.load_modules(model, ckpt_path, module_names=list(modules.keys()))
        else:
            self.log.warning("No evaluation checkpoint configured; using random weights.")

        mode = str(
            eval_cfg.get("mode", eval_cfg.get("forward_mode", "segmentation"))
        )
        task = "segmentation" if "seg" in mode else "reconstruction"
        data_cfg = get_data_cfg(self.cfg, split=eval_cfg.get("split", "test"), task=task)
        if not data_cfg:
            data_cfg = get_data_cfg(self.cfg, split="val", task=task)
        dataset = build_dataset(data_cfg)
        loader = build_dataloader(
            dataset, get_dataloader_cfg(self.cfg, split="test")
        )

        evaluator = Evaluator(
            model,
            device=device,
            mode=mode,
            compute_hd95=bool(eval_cfg.get("compute_hd95", False)),
        )
        metrics = evaluator.evaluate(
            loader,
            max_batches=eval_cfg.get("max_batches"),
        )
        out_dir = get_output_dir(self.cfg)
        out_path = out_dir / "evaluation_metrics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        self.log.info("Evaluation metrics: %s", metrics)
        return {"metrics": metrics, "checkpoint": ckpt, "seed": seed}

    def _maybe_save_result(self, result: dict[str, Any], *, seed: int) -> None:
        out_dir = get_output_dir(self.cfg)
        path = out_dir / f"experiment_seed{seed}.json"
        try:
            serializable = _json_safe(result)
            with path.open("w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, sort_keys=True)
            self.log.info("Wrote experiment result to %s", path)
        except Exception as exc:  # pragma: no cover
            self.log.warning("Could not serialize experiment result: %s", exc)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
