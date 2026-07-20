"""Generic training loop for staged ModularUNet experiments.

The Trainer does **not** hardcode stage1/2/3 logic. Initialization order is
enforced by the experiment runner / stage before ``fit``::

    1. build model
    2. random init (default construction)
    3. initialize_modules  (via stage.prepare)
    4. apply freeze        (via stage.prepare)
    5. build optimizer from trainable params only
    6. train               (Trainer.fit)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stagerecon.training.callbacks import CheckpointCallback, LoggingCallback
from stagerecon.training.checkpoint_manager import CheckpointManager
from stagerecon.training.early_stopping import EarlyStopping
from stagerecon.training.parameter_controller import ParameterController
from stagerecon.training.stages.base_stage import BaseStage

logger = logging.getLogger(__name__)

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "items") and not isinstance(cfg, dict):
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(cfg):
                return dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
        except Exception:
            pass
        return {str(k): v for k, v in cfg.items()}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {}


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    if isinstance(batch, (list, tuple)):
        moved = [
            v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for v in batch
        ]
        return type(batch)(moved) if not isinstance(batch, list) else moved
    return batch


class Trainer:
    """Stage-agnostic training / validation loop.

    Assumes ``stage.prepare(model)`` has already been called and that
    ``optimizer`` was built from trainable parameters only.
    """

    def __init__(
        self,
        model: nn.Module,
        stage: BaseStage,
        train_loader: DataLoader | Iterable[Any],
        val_loader: DataLoader | Iterable[Any] | None,
        loss_fn: LossFn,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        device: str | torch.device | None = None,
        config: Any | None = None,
        callbacks: Sequence[Any] | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        early_stopping: EarlyStopping | None = None,
    ) -> None:
        self.model = model
        self.stage = stage
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = _to_plain_dict(config)

        if device is None:
            device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        train_cfg = self.config.get("training", self.config)
        if isinstance(train_cfg, Mapping):
            train_cfg = dict(train_cfg)
        else:
            train_cfg = self.config

        self.epochs = int(train_cfg.get("epochs", self.config.get("epochs", 1)))
        self.use_amp = bool(train_cfg.get("amp", train_cfg.get("use_amp", False)))
        self.grad_clip = train_cfg.get("grad_clip", train_cfg.get("max_grad_norm"))
        self.grad_clip = float(self.grad_clip) if self.grad_clip is not None else None
        self.accumulation_steps = int(
            train_cfg.get(
                "accumulation_steps",
                train_cfg.get("grad_accumulation", train_cfg.get("gradient_accumulation", 1)),
            )
        )
        if self.accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")
        self.steps_per_epoch = train_cfg.get("steps_per_epoch")
        if self.steps_per_epoch is not None:
            self.steps_per_epoch = int(self.steps_per_epoch)

        save_dir = (
            train_cfg.get("save_dir")
            or self.config.get("save_dir")
            or self.config.get("output_dir")
            or "."
        )
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(save_dir)
        self.seed = train_cfg.get("seed", self.config.get("seed"))

        # Early stopping
        if early_stopping is not None:
            self.early_stopping = early_stopping
        else:
            es_cfg = train_cfg.get("early_stopping") or self.config.get("early_stopping")
            if es_cfg:
                es = _to_plain_dict(es_cfg)
                self.early_stopping = EarlyStopping(
                    patience=int(es.get("patience", 10)),
                    mode=str(es.get("mode", "min")),
                    min_delta=float(es.get("min_delta", 0.0)),
                )
            else:
                self.early_stopping = None

        self.monitor = str(
            train_cfg.get("monitor", self.config.get("monitor", "val_loss"))
        )
        self.monitor_mode = str(
            train_cfg.get("monitor_mode", self.config.get("monitor_mode", "min"))
        ).lower()

        default_callbacks: list[Any] = [
            LoggingCallback(log_every_n_steps=int(train_cfg.get("log_every_n_steps", 0))),
            CheckpointCallback(
                monitor=self.monitor,
                mode=self.monitor_mode,
                filename_best=self.stage.get_spec().checkpoint_output or "best.pt",
            ),
        ]
        self.callbacks: list[Any] = list(callbacks) if callbacks is not None else default_callbacks

        self.best_metric: float | None = None
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        self.model.to(self.device)
        self._scaler: Any | None = None
        if self.use_amp and self.device.type == "cuda":
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                self._scaler = torch.amp.GradScaler("cuda")
            else:
                self._scaler = torch.cuda.amp.GradScaler()
        elif self.use_amp and self.device.type != "cuda":
            logger.warning("AMP requested but device is %s; disabling AMP.", self.device)
            self.use_amp = False

        # Validate optimizer does not include frozen params
        ParameterController.validate_optimizer_excludes_frozen(self.optimizer)

    def _autocast(self):
        """Return an autocast context compatible with the active device."""
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp and self.device.type == "cuda",
            )
        return torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == "cuda")

    # ------------------------------------------------------------------ utils
    def _call_callbacks(self, method: str, *args: Any, **kwargs: Any) -> None:
        for cb in self.callbacks:
            fn = getattr(cb, method, None)
            if callable(fn):
                fn(*args, **kwargs)

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int,
        is_best: bool = False,
    ) -> Path:
        """Persist a module-wise checkpoint via CheckpointManager."""
        return self.checkpoint_manager.save(
            self.model,
            path,
            stage=self.stage.get_spec().name,
            epoch=epoch,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            best_metric=self.best_metric,
            config=self.config,
            seed=self.seed,
            extra={"is_best": is_best},
        )

    # ----------------------------------------------------------------- loops
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        # Keep frozen modules in eval (BN/Dropout)
        spec = self.stage.get_spec()
        ParameterController.set_train_eval_modes(
            self.model,
            trainable=spec.trainable_modules,
            frozen=spec.frozen_modules,
        )

        total_loss = 0.0
        n_steps = 0
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader, start=1):
            if self.steps_per_epoch is not None and step > self.steps_per_epoch:
                break

            batch = _move_batch_to_device(batch, self.device)
            with self._autocast():
                pred, target = self.stage.forward_batch(self.model, batch)
                loss = self.loss_fn(pred, target)
                loss = loss / self.accumulation_steps

            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % self.accumulation_steps == 0:
                if self.grad_clip is not None:
                    if self._scaler is not None:
                        self._scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.grad_clip,
                    )
                if self._scaler is not None:
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            step_loss = float(loss.detach().item()) * self.accumulation_steps
            total_loss += step_loss
            n_steps += 1
            self._call_callbacks(
                "on_batch_end",
                self,
                epoch=epoch,
                step=step,
                loss=step_loss,
                phase="train",
            )

        # Flush leftover accumulated grads if epoch ended mid-accumulation
        if n_steps > 0 and (n_steps % self.accumulation_steps) != 0:
            if self.grad_clip is not None:
                if self._scaler is not None:
                    self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip,
                )
            if self._scaler is not None:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return total_loss / max(n_steps, 1)

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> float:
        if self.val_loader is None:
            return float("nan")

        self.model.eval()
        total_loss = 0.0
        n_steps = 0

        for step, batch in enumerate(self.val_loader, start=1):
            if self.steps_per_epoch is not None and step > self.steps_per_epoch:
                break
            batch = _move_batch_to_device(batch, self.device)
            with self._autocast():
                pred, target = self.stage.forward_batch(self.model, batch)
                loss = self.loss_fn(pred, target)
            step_loss = float(loss.detach().item())
            total_loss += step_loss
            n_steps += 1
            self._call_callbacks(
                "on_batch_end",
                self,
                epoch=epoch,
                step=step,
                loss=step_loss,
                phase="val",
            )

        return total_loss / max(n_steps, 1)

    def fit(self) -> dict[str, Any]:
        """Run the training loop for ``epochs`` (or until early stopping).

        Returns:
            History dict with losses and best metric metadata.
        """
        self._call_callbacks("on_train_begin", self)

        for epoch in range(1, self.epochs + 1):
            self._call_callbacks("on_epoch_begin", self, epoch)

            train_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            metrics: dict[str, float] = {"train_loss": train_loss}
            if self.val_loader is not None:
                metrics["val_loss"] = val_loss

            self.history["train_loss"].append(train_loss)
            if self.val_loader is not None:
                self.history["val_loss"].append(val_loss)

            self._call_callbacks("on_epoch_end", self, epoch, metrics)

            monitored = metrics.get(self.monitor)
            if monitored is None:
                # Fall back to train_loss when val is unavailable
                monitored = metrics.get("train_loss")

            if monitored is not None and self.early_stopping is not None:
                self.early_stopping.step(monitored)
                if self.best_metric is None or (
                    (self.monitor_mode == "min" and monitored < self.best_metric)
                    or (self.monitor_mode == "max" and monitored > self.best_metric)
                ):
                    self.best_metric = float(monitored)
                if self.early_stopping.should_stop:
                    logger.info(
                        "Early stopping triggered at epoch %d (best=%s)",
                        epoch,
                        self.early_stopping.best,
                    )
                    break
            elif monitored is not None:
                if self.best_metric is None or (
                    (self.monitor_mode == "min" and monitored < self.best_metric)
                    or (self.monitor_mode == "max" and monitored > self.best_metric)
                ):
                    self.best_metric = float(monitored)

        self._call_callbacks("on_train_end", self)
        return {
            "history": self.history,
            "best_metric": self.best_metric,
            "stage": self.stage.get_spec().name,
        }
