"""Abstract base class for staged ModularUNet training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from stagerecon.training.checkpoint_manager import CheckpointManager
from stagerecon.training.parameter_controller import ParameterController
from stagerecon.training.stage_spec import StageSpec


def extract_batch_input(batch: Any) -> torch.Tensor:
    """Extract the network input tensor from a batch."""
    if isinstance(batch, dict):
        for key in ("image", "input", "x", "data"):
            if key in batch:
                return batch[key]
        raise KeyError(
            "Batch dict must contain one of: image, input, x, data. "
            f"Got keys: {sorted(batch.keys())}"
        )
    if isinstance(batch, (list, tuple)):
        if not batch:
            raise ValueError("Empty batch sequence.")
        return batch[0]
    if isinstance(batch, torch.Tensor):
        return batch
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def extract_batch_target(batch: Any, forward_mode: str, inputs: torch.Tensor) -> torch.Tensor:
    """Extract the supervision target for the given forward mode."""
    mode = forward_mode.lower()
    if isinstance(batch, dict):
        if mode == "segmentation":
            for key in ("mask", "label", "target", "seg", "segmentation"):
                if key in batch:
                    return batch[key]
            raise KeyError(
                "Segmentation batch must contain one of: mask, label, target, "
                f"seg, segmentation. Got keys: {sorted(batch.keys())}"
            )
        for key in ("target", "reconstruction_target", "image", "input"):
            if key in batch:
                return batch[key]
        return inputs
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            return batch[1]
        return inputs
    # Tensor-only batch: autoencoder-style reconstruction target = input
    return inputs


class BaseStage(ABC):
    """Thin stage wrapper around a :class:`StageSpec`.

    Stages do **not** build incomplete models. They operate on a full
    :class:`~stagerecon.models.composed.modular_unet.ModularUNet` and only
    control initialization + freeze sets + forward mode.
    """

    def __init__(
        self,
        spec: StageSpec,
        *,
        checkpoint_manager: CheckpointManager | None = None,
        parameter_controller: ParameterController | None = None,
    ) -> None:
        self.spec = spec
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.parameter_controller = parameter_controller or ParameterController()

    def get_spec(self) -> StageSpec:
        """Return this stage's specification."""
        return self.spec

    def prepare(self, model: nn.Module) -> nn.Module:
        """Initialize modules from checkpoints, then apply freeze / trainable sets.

        Intended call order in an experiment runner::

            model = build_model(cfg)          # random init by construction
            stage.prepare(model)              # module-wise load + freeze
            optimizer = build_optimizer(
                ParameterController.get_trainable_param_groups(model), cfg
            )
            trainer.fit(...)
        """
        if self.spec.module_initialization:
            self.checkpoint_manager.initialize_modules(
                model, self.spec.module_initialization
            )
        self.parameter_controller.apply_trainable_frozen(
            model,
            trainable=self.spec.trainable_modules,
            frozen=self.spec.frozen_modules,
        )
        self.parameter_controller.set_train_eval_modes(
            model,
            trainable=self.spec.trainable_modules,
            frozen=self.spec.frozen_modules,
        )
        return model

    def forward_batch(
        self,
        model: nn.Module,
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ``model(x, mode=spec.forward_mode)`` and return ``(pred, target)``."""
        inputs = extract_batch_input(batch)
        output = model(inputs, mode=self.spec.forward_mode)
        if hasattr(output, "prediction"):
            pred = output.prediction
        elif isinstance(output, torch.Tensor):
            pred = output
        else:
            raise TypeError(
                f"Model output must provide .prediction or be a Tensor; got {type(output)!r}"
            )
        target = extract_batch_target(batch, self.spec.forward_mode, inputs)
        return pred, target

    @abstractmethod
    def stage_kind(self) -> str:
        """Short identifier for factory / logging (e.g. ``encoder_bottleneck``)."""
