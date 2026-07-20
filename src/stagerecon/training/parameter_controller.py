"""Freeze / unfreeze helpers and trainable-parameter utilities."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator, Sequence

import torch
import torch.nn as nn

from stagerecon.training.module_access import get_model_module

logger = logging.getLogger(__name__)


class ParameterController:
    """Control which ModularUNet modules are trainable during a stage."""

    @staticmethod
    def freeze_modules(model: nn.Module, names: Sequence[str]) -> None:
        """Set ``requires_grad=False`` for all parameters in named modules."""
        for name in names:
            module = get_model_module(model, name)
            for param in module.parameters():
                param.requires_grad = False
            logger.info("Froze module '%s'", name)

    @staticmethod
    def unfreeze_modules(model: nn.Module, names: Sequence[str]) -> None:
        """Set ``requires_grad=True`` for all parameters in named modules."""
        for name in names:
            module = get_model_module(model, name)
            for param in module.parameters():
                param.requires_grad = True
            logger.info("Unfroze module '%s'", name)

    @classmethod
    def apply_trainable_frozen(
        cls,
        model: nn.Module,
        trainable: Sequence[str],
        frozen: Sequence[str],
    ) -> None:
        """Apply trainable / frozen sets with overlap validation.

        First freezes ``frozen`` modules, then unfreezes ``trainable`` modules.
        Modules not listed in either set are left unchanged.
        """
        trainable_set = {str(n) for n in trainable}
        frozen_set = {str(n) for n in frozen}
        overlap = trainable_set & frozen_set
        if overlap:
            raise ValueError(
                "trainable_modules and frozen_modules must not overlap; "
                f"found: {sorted(overlap)}"
            )
        if frozen_set:
            cls.freeze_modules(model, sorted(frozen_set))
        if trainable_set:
            cls.unfreeze_modules(model, sorted(trainable_set))

    @staticmethod
    def set_train_eval_modes(
        model: nn.Module,
        trainable: Sequence[str],
        frozen: Sequence[str] | None = None,
    ) -> None:
        """Put trainable modules in train mode and frozen modules in eval.

        BatchNorm / Dropout in frozen modules stay in eval to avoid updating
        running statistics when those modules should not learn.
        """
        frozen = list(frozen or [])
        for name in frozen:
            get_model_module(model, name).eval()
        for name in trainable:
            get_model_module(model, name).train()

    @staticmethod
    def iter_trainable_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
        """Yield parameters with ``requires_grad=True``."""
        for param in model.parameters():
            if param.requires_grad:
                yield param

    @classmethod
    def get_trainable_param_groups(
        cls,
        model: nn.Module,
        *,
        lr: float | None = None,
        weight_decay: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return a single optimizer param group for trainable parameters."""
        params = list(cls.iter_trainable_parameters(model))
        if not params:
            raise RuntimeError("No trainable parameters found on the model.")
        group: dict[str, Any] = {"params": params}
        if lr is not None:
            group["lr"] = lr
        if weight_decay is not None:
            group["weight_decay"] = weight_decay
        return [group]

    @staticmethod
    def validate_optimizer_excludes_frozen(optimizer: torch.optim.Optimizer) -> None:
        """Raise if the optimizer contains any frozen (``requires_grad=False``) params."""
        bad: list[str] = []
        for group_idx, group in enumerate(optimizer.param_groups):
            for p_idx, param in enumerate(group["params"]):
                if not param.requires_grad:
                    bad.append(f"param_groups[{group_idx}][{p_idx}]")
        if bad:
            raise RuntimeError(
                "Optimizer includes frozen parameters (requires_grad=False): "
                + ", ".join(bad)
            )

    @classmethod
    def validate_trainable_can_receive_grad(
        cls,
        model: nn.Module,
        trainable: Sequence[str],
        frozen: Sequence[str] | None = None,
        *,
        forward_fn: Any | None = None,
        sample_input: torch.Tensor | None = None,
    ) -> None:
        """Optional sanity check that trainable modules receive gradients.

        If ``forward_fn`` (or ``sample_input`` with ``model``) is provided, runs
        a tiny forward+backward and asserts:

        - every listed trainable module has at least one parameter with grad
        - every listed frozen module has no parameter grads
        """
        frozen = list(frozen or [])
        trainable = list(trainable)

        if forward_fn is None and sample_input is None:
            # Structural check only: requires_grad flags match the lists.
            for name in trainable:
                module = get_model_module(model, name)
                if not any(p.requires_grad for p in module.parameters()):
                    raise RuntimeError(
                        f"Trainable module '{name}' has no requires_grad=True parameters."
                    )
            for name in frozen:
                module = get_model_module(model, name)
                if any(p.requires_grad for p in module.parameters()):
                    raise RuntimeError(
                        f"Frozen module '{name}' still has requires_grad=True parameters."
                    )
            return

        model.zero_grad(set_to_none=True)
        if forward_fn is not None:
            output = forward_fn(model)
        else:
            assert sample_input is not None
            output = model(sample_input)
        if hasattr(output, "prediction"):
            loss = output.prediction.float().mean()
        elif isinstance(output, torch.Tensor):
            loss = output.float().mean()
        else:
            raise TypeError(
                "Forward output must be a Tensor or object with .prediction"
            )
        loss.backward()

        for name in trainable:
            module = get_model_module(model, name)
            has_grad = any(
                p.grad is not None and torch.any(p.grad != 0)
                for p in module.parameters()
                if p.requires_grad
            )
            # Allow modules with no parameters (rare); otherwise require grad.
            params = list(module.parameters())
            if params and not has_grad:
                # Soft check: at least some param has a grad tensor allocated.
                has_any = any(p.grad is not None for p in params if p.requires_grad)
                if not has_any:
                    raise RuntimeError(
                        f"Trainable module '{name}' did not receive gradients."
                    )

        for name in frozen:
            module = get_model_module(model, name)
            leaked = [
                pname
                for pname, p in module.named_parameters()
                if p.grad is not None and torch.any(p.grad != 0)
            ]
            if leaked:
                raise RuntimeError(
                    f"Frozen module '{name}' unexpectedly received non-zero "
                    f"gradients on: {leaked[:5]}"
                )

        model.zero_grad(set_to_none=True)
