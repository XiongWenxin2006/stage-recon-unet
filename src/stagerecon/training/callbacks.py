"""Lightweight callback hooks for the Trainer loop."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Callback(Protocol):
    """Protocol for trainer callbacks (all methods optional in practice)."""

    def on_train_begin(self, trainer: Any) -> None: ...

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None: ...

    def on_batch_end(
        self,
        trainer: Any,
        *,
        epoch: int,
        step: int,
        loss: float,
        phase: str,
    ) -> None: ...

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: dict[str, float],
    ) -> None: ...

    def on_train_end(self, trainer: Any) -> None: ...


class CallbackBase:
    """No-op base class so subclasses only override what they need."""

    def on_train_begin(self, trainer: Any) -> None:
        return None

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        return None

    def on_batch_end(
        self,
        trainer: Any,
        *,
        epoch: int,
        step: int,
        loss: float,
        phase: str,
    ) -> None:
        return None

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        return None

    def on_train_end(self, trainer: Any) -> None:
        return None


class LoggingCallback(CallbackBase):
    """Log epoch metrics via the standard library logger."""

    def __init__(self, log_every_n_steps: int = 0) -> None:
        self.log_every_n_steps = int(log_every_n_steps)

    def on_train_begin(self, trainer: Any) -> None:
        stage_name = getattr(getattr(trainer, "stage", None), "get_spec", lambda: None)()
        name = getattr(stage_name, "name", None) if stage_name is not None else None
        logger.info(
            "Training begin | device=%s | stage=%s",
            getattr(trainer, "device", "?"),
            name or "?",
        )

    def on_batch_end(
        self,
        trainer: Any,
        *,
        epoch: int,
        step: int,
        loss: float,
        phase: str,
    ) -> None:
        if self.log_every_n_steps > 0 and phase == "train":
            if step % self.log_every_n_steps == 0:
                logger.info(
                    "epoch=%d step=%d phase=%s loss=%.6f",
                    epoch,
                    step,
                    phase,
                    loss,
                )

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        parts = " ".join(f"{k}={v:.6f}" for k, v in sorted(metrics.items()))
        logger.info("epoch=%d %s", epoch, parts)

    def on_train_end(self, trainer: Any) -> None:
        logger.info(
            "Training end | best_metric=%s",
            getattr(trainer, "best_metric", None),
        )


class CheckpointCallback(CallbackBase):
    """Save last / best checkpoints via the trainer's CheckpointManager."""

    def __init__(
        self,
        *,
        monitor: str = "val_loss",
        mode: str = "min",
        save_last: bool = True,
        filename_best: str | None = None,
        filename_last: str = "last.pt",
    ) -> None:
        self.monitor = monitor
        mode = str(mode).lower().strip()
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.save_last = save_last
        self.filename_best = filename_best
        self.filename_last = filename_last
        self.best: float | None = None

    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best
        return value > self.best

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        ckpt = getattr(trainer, "checkpoint_manager", None)
        if ckpt is None:
            return

        if self.save_last:
            trainer.save_checkpoint(self.filename_last, epoch=epoch, is_best=False)

        if self.monitor not in metrics:
            return

        value = float(metrics[self.monitor])
        if self._is_better(value):
            self.best = value
            trainer.best_metric = value
            best_name = self.filename_best
            if best_name is None:
                spec = trainer.stage.get_spec()
                best_name = spec.checkpoint_output or "best.pt"
            trainer.save_checkpoint(best_name, epoch=epoch, is_best=True)
            logger.info(
                "New best %s=%.6f -> saved %s",
                self.monitor,
                value,
                best_name,
            )
