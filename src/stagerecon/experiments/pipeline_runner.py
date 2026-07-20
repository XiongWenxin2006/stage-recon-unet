"""Run configurable StageRecon stage pipelines (stage1→2→3→downstream)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Sequence

from stagerecon.data import build_dataloader, build_dataset
from stagerecon.experiments.config_access import (
    extract_stage_cfg,
    get_checkpoint_dir,
    get_data_cfg,
    get_dataloader_cfg,
    get_loss_cfg,
    get_model_cfg,
    get_optimizer_cfg,
    get_output_dir,
    get_scheduler_cfg,
    get_trainer_cfg,
    is_downstream_stage,
    normalize_stage_name,
    resolve_stage_paths,
    resolve_stages_list,
    to_plain,
)
from stagerecon.experiments.seed_manager import prepare_seed
from stagerecon.models import build_model
from stagerecon.objectives import build_loss
from stagerecon.training import (
    CheckpointManager,
    ParameterController,
    Trainer,
    build_optimizer,
    build_scheduler,
    build_stage,
)
from stagerecon.utils import get_device, setup_logger

logger = logging.getLogger(__name__)


def _task_for_stage(stage_name: str, stage_cfg: dict[str, Any]) -> str:
    explicit = stage_cfg.get("data_task") or stage_cfg.get("task")
    if explicit:
        task = str(explicit).lower().strip()
        if task in {"seg", "segmentation", "downstream"}:
            return "segmentation"
        if task in {"recon", "reconstruction", "pretrain", "restore"}:
            return "reconstruction"
        return task
    forward = str(stage_cfg.get("forward_mode", "")).lower()
    if is_downstream_stage(stage_name) or forward == "segmentation":
        return "segmentation"
    return "reconstruction"


def _build_split_loader(
    cfg: Any,
    *,
    split: str,
    task: str,
) -> Any:
    data_cfg = get_data_cfg(cfg, split=split, task=task)
    if not data_cfg:
        # Minimal synthetic fallback so smoke configs without data still run
        data_cfg = {
            "name": "synthetic",
            "task": task,
            "num_samples": 8 if split == "train" else 4,
            "image_size": 32,
            "in_channels": 1,
            "seed": 0 if split == "train" else 1,
        }
        logger.warning(
            "No data/data_source config found; using synthetic fallback for split=%s task=%s",
            split,
            task,
        )
    dataset = build_dataset(data_cfg)
    loader_cfg = get_dataloader_cfg(cfg, split=split)
    return build_dataloader(dataset, loader_cfg)


def _ensure_best_checkpoint(
    *,
    model: Any,
    stage: Any,
    checkpoint_manager: CheckpointManager,
    trainer_result: dict[str, Any],
    optimizer: Any,
    scheduler: Any,
    trainer_cfg: dict[str, Any],
    seed: int | None,
) -> Path:
    """Ensure the stage's configured best checkpoint path exists on disk."""
    spec = stage.get_spec()
    ckpt_name = spec.checkpoint_output or f"{spec.name}_best.pt"
    out_path = checkpoint_manager.resolve_path(ckpt_name)

    if out_path.is_file():
        logger.info("Best checkpoint present at %s", out_path)
        return out_path

    # Trainer may not have written best (e.g. missing monitor metric). Save now.
    logger.warning(
        "Best checkpoint missing at %s; writing final weights as best.",
        out_path,
    )
    epoch = 0
    history = trainer_result.get("history") or {}
    if isinstance(history, dict):
        train_hist = history.get("train_loss") or []
        if train_hist:
            epoch = len(train_hist)

    return checkpoint_manager.save(
        model,
        out_path,
        stage=spec.name,
        epoch=epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        best_metric=trainer_result.get("best_metric"),
        config=trainer_cfg,
        seed=seed,
        extra={"is_best": True, "ensured_by": "PipelineRunner"},
    )


def run_stage(
    cfg: Any,
    stage_name: str,
    *,
    seed: int | None = None,
    device: Any | None = None,
) -> dict[str, Any]:
    """Build, prepare, and train a single named stage.

    Call order matches the training package contract::

        model = build_model(cfg.model)
        stage = build_stage(resolved_stage_cfg)
        stage.prepare(model)                 # module-wise load + freeze
        optimizer = build_optimizer(
            ParameterController.get_trainable_param_groups(model), opt_cfg
        )
        Trainer(...).fit()

    Args:
        cfg: Full experiment config.
        stage_name: ``stage1`` / ``stage2`` / ``stage3`` / ``downstream`` (aliases ok).
        seed: Optional seed override (applied when not ``None``).
        device: Optional device override.

    Returns:
        Dict with stage name, history, best metric, and checkpoint path.
    """
    canonical = normalize_stage_name(stage_name)
    if seed is not None:
        prepare_seed(cfg, seed=seed)

    root = to_plain(cfg)
    ckpt_dir = get_checkpoint_dir(cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    get_output_dir(cfg).mkdir(parents=True, exist_ok=True)

    stage_cfg = resolve_stage_paths(extract_stage_cfg(cfg, canonical), cfg)
    # Default checkpoint_output filenames when unspecified
    if not stage_cfg.get("checkpoint_output"):
        stage_cfg["checkpoint_output"] = str(ckpt_dir / f"{canonical}_best.pt")
    stage_cfg.setdefault("save_dir", str(ckpt_dir))
    stage_cfg.setdefault("checkpoint_dir", str(ckpt_dir))
    stage_cfg.setdefault("type", stage_cfg.get("type", canonical))
    stage_cfg.setdefault("name", stage_cfg.get("name", canonical))

    logger.info("=== Running stage '%s' ===", canonical)
    logger.info("Stage config keys: %s", sorted(stage_cfg.keys()))

    model = build_model(get_model_cfg(cfg))
    stage = build_stage(stage_cfg)
    # API: prepare(model) — CheckpointManager lives on the stage
    stage.prepare(model)

    loss_cfg = get_loss_cfg(cfg, stage.get_spec().loss_name)
    loss_fn = build_loss(loss_cfg)

    param_groups = ParameterController.get_trainable_param_groups(model)
    optimizer = build_optimizer(param_groups, get_optimizer_cfg(cfg))
    scheduler = build_scheduler(optimizer, get_scheduler_cfg(cfg))

    task = _task_for_stage(canonical, stage_cfg)
    train_loader = _build_split_loader(cfg, split="train", task=task)
    # Validation is optional
    val_loader = None
    data_root = to_plain(root.get("data") or root.get("data_source") or {})
    wants_val = True
    if "val" in data_root or "validation" in data_root or data_root.get("name") == "synthetic":
        wants_val = True
    trainer_cfg = get_trainer_cfg(cfg)
    if trainer_cfg.get("skip_validation") or trainer_cfg.get("no_val"):
        wants_val = False
    if wants_val:
        try:
            val_loader = _build_split_loader(cfg, split="val", task=task)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not build val loader (%s); continuing without.", exc)
            val_loader = None

    if device is None:
        device = get_device(root.get("device", trainer_cfg.get("device", "auto")))

    # Merge trainer settings for Trainer
    fit_cfg: dict[str, Any] = {
        **trainer_cfg,
        "device": str(device),
        "save_dir": str(ckpt_dir),
        "output_dir": str(get_output_dir(cfg)),
        "seed": seed if seed is not None else root.get("seed"),
    }
    # Prefer stage-local epoch overrides
    for key in ("epochs", "steps_per_epoch", "amp", "grad_clip", "monitor"):
        if key in stage_cfg and stage_cfg[key] is not None:
            fit_cfg[key] = stage_cfg[key]

    checkpoint_manager = getattr(stage, "checkpoint_manager", None) or CheckpointManager(
        ckpt_dir
    )
    # Keep save_dir aligned
    checkpoint_manager.save_dir = Path(ckpt_dir)
    checkpoint_manager.save_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        stage=stage,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=fit_cfg,
        checkpoint_manager=checkpoint_manager,
    )
    result = trainer.fit()

    best_path = _ensure_best_checkpoint(
        model=model,
        stage=stage,
        checkpoint_manager=checkpoint_manager,
        trainer_result=result,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_cfg=fit_cfg,
        seed=fit_cfg.get("seed"),
    )

    # Also copy/symlink to canonical alias under checkpoint dir when needed
    alias = ckpt_dir / f"{canonical}_best.pt"
    if best_path.resolve() != alias.resolve():
        try:
            if not alias.exists():
                shutil.copy2(best_path, alias)
                logger.info("Copied best checkpoint to alias %s", alias)
        except Exception as exc:  # pragma: no cover
            logger.debug("Could not copy alias checkpoint: %s", exc)

    return {
        "stage": canonical,
        "history": result.get("history"),
        "best_metric": result.get("best_metric"),
        "checkpoint": str(best_path),
        "checkpoint_alias": str(alias),
    }


class PipelineRunner:
    """Execute a sequence of stages: stage1 → stage2 → stage3 → downstream.

    Stages are configurable via ``cfg.experiment.stages`` (or constructor
    ``stages``). Supports ablations such as ``skip_stage2`` where
    ``stages = [stage1, stage3, downstream]`` and stage3 init is adjusted in
    the Hydra config (e.g. load encoder+bottleneck from stage1 only).
    """

    def __init__(
        self,
        cfg: Any,
        *,
        stages: Sequence[str] | None = None,
        seed: int | None = None,
        logger_name: str = "stagerecon.pipeline",
    ) -> None:
        self.cfg = cfg
        self.stages = resolve_stages_list(cfg, stages)
        self.seed = seed
        self.log = setup_logger(logger_name)

    def run(self) -> dict[str, Any]:
        """Run all configured stages sequentially.

        Returns:
            ``{"stages": {name: stage_result}, "seed": int, "stages_order": [...]}``
        """
        applied_seed = prepare_seed(self.cfg, seed=self.seed)
        self.log.info(
            "Pipeline start | seed=%s | stages=%s",
            applied_seed,
            self.stages,
        )

        results: dict[str, Any] = {}
        for stage_name in self.stages:
            stage_result = run_stage(
                self.cfg,
                stage_name,
                seed=None,  # already seeded for the pipeline
                device=None,
            )
            results[stage_name] = stage_result
            self.log.info(
                "Finished %s | best_metric=%s | checkpoint=%s",
                stage_name,
                stage_result.get("best_metric"),
                stage_result.get("checkpoint"),
            )

        payload = {
            "stages": results,
            "seed": applied_seed,
            "stages_order": list(self.stages),
        }
        self.log.info("Pipeline complete | stages=%s", self.stages)
        return payload
