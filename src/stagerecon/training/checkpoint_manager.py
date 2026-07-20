"""Module-wise checkpoint save / load for staged ModularUNet training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from stagerecon.training.module_access import (
    KNOWN_MODULES,
    get_model_module,
    iter_known_modules,
)
from stagerecon.training.stage_spec import ModuleInitializationSpec

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Save and load ModularUNet weights on a per-module basis.

    Checkpoints store::

        {
            "modules": {module_name: state_dict, ...},
            "stage": ...,
            "epoch": ...,
            "optimizer": ...,
            "scheduler": ...,
            "best_metric": ...,
            "config": ...,
            "seed": ...,
        }

    Stage initialization must go through :meth:`initialize_modules` /
    :meth:`load_modules` — never by blindly calling
    ``model.load_state_dict(...)`` on a full monolithic state dict.
    """

    def __init__(self, save_dir: str | Path | None = None) -> None:
        self.save_dir = Path(save_dir) if save_dir is not None else Path(".")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve ``path`` relative to ``save_dir`` when not absolute."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.save_dir / p

    def collect_module_state_dicts(self, model: nn.Module) -> dict[str, dict[str, Any]]:
        """Collect ``state_dict`` for every known module present on ``model``."""
        modules: dict[str, dict[str, Any]] = {}
        for name, module in iter_known_modules(model):
            modules[name] = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        return modules

    def save(
        self,
        model: nn.Module,
        path: str | Path,
        *,
        stage: str | None = None,
        epoch: int | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        best_metric: float | None = None,
        config: Any | None = None,
        seed: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save a module-wise checkpoint (plus optional training state)."""
        out_path = self.resolve_path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "modules": self.collect_module_state_dicts(model),
            "stage": stage,
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
            "seed": seed,
            "known_modules": list(KNOWN_MODULES),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            payload["scheduler"] = scheduler.state_dict()
        if extra:
            payload["extra"] = dict(extra)

        torch.save(payload, out_path)
        logger.info(
            "Saved module-wise checkpoint to %s (modules=%s)",
            out_path,
            sorted(payload["modules"].keys()),
        )
        return out_path

    @staticmethod
    def _read_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict):
            raise TypeError(
                f"Expected checkpoint dict at {path}, got {type(obj)!r}. "
                "Stage init requires module-wise checkpoints."
            )
        return obj

    @classmethod
    def extract_modules_dict(cls, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
        """Return the per-module state dict mapping from a checkpoint payload."""
        if "modules" in checkpoint and isinstance(checkpoint["modules"], Mapping):
            return dict(checkpoint["modules"])
        # Reject monolithic model state_dict for stage initialization.
        if "model" in checkpoint or "state_dict" in checkpoint:
            raise ValueError(
                "Checkpoint contains a full model state_dict ('model'/'state_dict') "
                "but no 'modules' mapping. Refusing to load blindly for stage init. "
                "Use module-wise checkpoints produced by CheckpointManager.save()."
            )
        # Allow a bare {module_name: state_dict} mapping.
        if all(isinstance(v, Mapping) for v in checkpoint.values()):
            keys = set(checkpoint.keys())
            if keys & set(KNOWN_MODULES):
                return dict(checkpoint)
        raise ValueError(
            "Unrecognized checkpoint format: expected a top-level 'modules' dict "
            "mapping module names to state_dicts."
        )

    def load_modules(
        self,
        model: nn.Module,
        checkpoint_path: str | Path,
        module_names: Sequence[str],
        *,
        strict: bool = True,
        source_module_map: Mapping[str, str] | None = None,
    ) -> None:
        """Load selected modules from a module-wise checkpoint.

        Args:
            model: Target ModularUNet (or compatible).
            checkpoint_path: Path to checkpoint file.
            module_names: Target module names on ``model`` to load into.
            strict: Forwarded to ``load_state_dict``.
            source_module_map: Optional map ``target_name -> source_name`` for
                reading a differently named module from the checkpoint.
        """
        path = self.resolve_path(checkpoint_path)
        payload = self._read_checkpoint(path)
        modules_dict = self.extract_modules_dict(payload)
        source_module_map = dict(source_module_map or {})

        for target_name in module_names:
            source_name = source_module_map.get(target_name, target_name)
            if source_name not in modules_dict:
                raise KeyError(
                    f"Module '{source_name}' missing in checkpoint {path}. "
                    f"Available: {sorted(modules_dict.keys())}"
                )
            module = get_model_module(model, target_name)
            state = modules_dict[source_name]
            module.load_state_dict(state, strict=strict)
            logger.info(
                "Loaded %s from %s",
                target_name if source_name == target_name else f"{target_name}<-{source_name}",
                path.name,
            )

    def initialize_modules(
        self,
        model: nn.Module,
        initialization: Mapping[str, ModuleInitializationSpec | Mapping[str, Any]],
    ) -> None:
        """Initialize modules according to per-module specs.

        - ``source="random"``: skip (keep existing / randomly initialized weights)
        - ``source="checkpoint"``: load that module only from the given path

        Raises:
            KeyError: If a requested module is missing from the checkpoint.
            ValueError: If a checkpoint-backed spec lacks a path.
        """
        for module_name, spec_raw in initialization.items():
            if isinstance(spec_raw, ModuleInitializationSpec):
                spec = spec_raw
            else:
                spec = ModuleInitializationSpec.from_config(spec_raw)

            if spec.source == "random":
                logger.info(
                    "Keeping random / existing initialization for module '%s'",
                    module_name,
                )
                continue

            if not spec.checkpoint_path:
                raise ValueError(
                    f"Checkpoint initialization for '{module_name}' requires "
                    "checkpoint_path."
                )

            source_map = None
            if spec.source_module and spec.source_module != module_name:
                source_map = {module_name: spec.source_module}

            self.load_modules(
                model,
                spec.checkpoint_path,
                module_names=[module_name],
                strict=spec.strict,
                source_module_map=source_map,
            )
            src = spec.source_module or module_name
            logger.info(
                "Loaded %s from %s",
                module_name if src == module_name else f"{module_name} (from {src})",
                Path(spec.checkpoint_path).name,
            )

    def load_training_state(
        self,
        checkpoint_path: str | Path,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> dict[str, Any]:
        """Load optimizer / scheduler / metadata for resume (not module weights).

        Returns the full checkpoint payload so callers can read ``epoch``,
        ``best_metric``, etc.
        """
        path = self.resolve_path(checkpoint_path)
        payload = self._read_checkpoint(path)

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
            logger.info("Restored optimizer state from %s", path)
        if (
            scheduler is not None
            and "scheduler" in payload
            and hasattr(scheduler, "load_state_dict")
        ):
            scheduler.load_state_dict(payload["scheduler"])
            logger.info("Restored scheduler state from %s", path)

        return payload
