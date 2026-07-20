"""Stage specification dataclasses for staged ModularUNet training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    """Convert OmegaConf / Mapping configs to a plain dict."""
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
    raise TypeError(f"Unsupported config type: {type(cfg)!r}")


@dataclass
class ModuleInitializationSpec:
    """How a single named module should be initialized for a stage.

    Attributes:
        source: ``"random"`` keeps the model's current (typically random)
            weights; ``"checkpoint"`` loads weights from ``checkpoint_path``.
        checkpoint_path: Path to a module-wise checkpoint when ``source`` is
            ``"checkpoint"``.
        source_module: Optional name of the module inside the checkpoint to
            load from (defaults to the target module name).
        strict: Whether ``load_state_dict`` should be strict.
    """

    source: str  # "random" | "checkpoint"
    checkpoint_path: str | None = None
    source_module: str | None = None
    strict: bool = True

    def __post_init__(self) -> None:
        source = str(self.source).lower().strip()
        if source not in {"random", "checkpoint"}:
            raise ValueError(
                f"Invalid initialization source '{self.source}'. "
                "Expected 'random' or 'checkpoint'."
            )
        self.source = source
        if self.source == "checkpoint" and not self.checkpoint_path:
            raise ValueError(
                "ModuleInitializationSpec with source='checkpoint' requires "
                "checkpoint_path."
            )

    @classmethod
    def from_config(cls, cfg: Any) -> ModuleInitializationSpec:
        """Build from a dict / OmegaConf node."""
        if isinstance(cfg, ModuleInitializationSpec):
            return cfg
        data = _to_plain_dict(cfg)
        return cls(
            source=str(data.get("source", "random")),
            checkpoint_path=data.get("checkpoint_path"),
            source_module=data.get("source_module"),
            strict=bool(data.get("strict", True)),
        )


@dataclass
class StageSpec:
    """Declarative specification for one training stage.

    All pretraining stages use a full ModularUNet forward. Differences between
    stages are captured by ``module_initialization`` and the
    trainable / frozen module sets.
    """

    name: str
    forward_mode: str  # bottleneck_reconstruction | reconstruction | segmentation
    module_initialization: dict[str, ModuleInitializationSpec] = field(
        default_factory=dict
    )
    trainable_modules: list[str] = field(default_factory=list)
    frozen_modules: list[str] = field(default_factory=list)
    checkpoint_output: str = ""
    loss_name: str = "mse"
    checkpoint_input: str | None = None  # legacy optional

    def __post_init__(self) -> None:
        mode = str(self.forward_mode).lower().strip()
        allowed = {
            "bottleneck_reconstruction",
            "reconstruction",
            "segmentation",
        }
        if mode not in allowed:
            raise ValueError(
                f"Invalid forward_mode '{self.forward_mode}'. "
                f"Expected one of {sorted(allowed)}."
            )
        self.forward_mode = mode
        self.trainable_modules = [str(m) for m in self.trainable_modules]
        self.frozen_modules = [str(m) for m in self.frozen_modules]

        overlap = set(self.trainable_modules) & set(self.frozen_modules)
        if overlap:
            raise ValueError(
                f"StageSpec '{self.name}' has overlapping trainable/frozen "
                f"modules: {sorted(overlap)}"
            )

    @classmethod
    def from_config(cls, cfg: Any) -> StageSpec:
        """Build a :class:`StageSpec` from a dict / OmegaConf configuration.

        Expected keys::

            name, forward_mode, module_initialization, trainable_modules,
            frozen_modules, checkpoint_output, loss_name, checkpoint_input
        """
        if isinstance(cfg, StageSpec):
            return cfg

        root = _to_plain_dict(cfg)
        # Allow nesting under "stage"
        if "stage" in root and isinstance(root["stage"], (Mapping, dict)):
            data = _to_plain_dict(root["stage"])
        else:
            data = root

        init_cfg = data.get("module_initialization", {}) or {}
        init_plain = _to_plain_dict(init_cfg)
        module_initialization: dict[str, ModuleInitializationSpec] = {}
        for module_name, mod_cfg in init_plain.items():
            module_initialization[str(module_name)] = ModuleInitializationSpec.from_config(
                mod_cfg
            )

        trainable = list(data.get("trainable_modules", []) or [])
        frozen = list(data.get("frozen_modules", []) or [])

        return cls(
            name=str(data.get("name", "unnamed_stage")),
            forward_mode=str(data.get("forward_mode", "reconstruction")),
            module_initialization=module_initialization,
            trainable_modules=[str(m) for m in trainable],
            frozen_modules=[str(m) for m in frozen],
            checkpoint_output=str(data.get("checkpoint_output", "")),
            loss_name=str(data.get("loss_name", "mse")),
            checkpoint_input=data.get("checkpoint_input"),
        )


def default_stage1_spec(checkpoint_output: str = "stage1_best.pt") -> StageSpec:
    """Default Stage-1: full UNet, all modules random, reconstruction forward."""
    modules = [
        "encoder",
        "bottleneck",
        "decoder",
        "reconstruction_head",
    ]
    return StageSpec(
        name="stage1_encoder_bottleneck",
        forward_mode="reconstruction",
        module_initialization={
            m: ModuleInitializationSpec(source="random") for m in modules
        },
        trainable_modules=list(modules),
        frozen_modules=[],
        checkpoint_output=checkpoint_output,
        loss_name="mse",
    )


def default_stage2_spec(
    stage1_checkpoint: str,
    checkpoint_output: str = "stage2_best.pt",
) -> StageSpec:
    """Default Stage-2: load bottleneck from stage1; freeze encoder."""
    return StageSpec(
        name="stage2_bottleneck_decoder",
        forward_mode="reconstruction",
        module_initialization={
            "encoder": ModuleInitializationSpec(source="random"),
            "bottleneck": ModuleInitializationSpec(
                source="checkpoint",
                checkpoint_path=stage1_checkpoint,
                source_module="bottleneck",
            ),
            "decoder": ModuleInitializationSpec(source="random"),
            "reconstruction_head": ModuleInitializationSpec(source="random"),
        },
        trainable_modules=["bottleneck", "decoder", "reconstruction_head"],
        frozen_modules=["encoder"],
        checkpoint_output=checkpoint_output,
        loss_name="mse",
        checkpoint_input=stage1_checkpoint,
    )


def default_stage3_spec(
    stage1_checkpoint: str,
    stage2_checkpoint: str,
    checkpoint_output: str = "stage3_best.pt",
) -> StageSpec:
    """Default Stage-3: encoder from s1, bottleneck+decoder from s2."""
    return StageSpec(
        name="stage3_full_reconstruction",
        forward_mode="reconstruction",
        module_initialization={
            "encoder": ModuleInitializationSpec(
                source="checkpoint",
                checkpoint_path=stage1_checkpoint,
                source_module="encoder",
            ),
            "bottleneck": ModuleInitializationSpec(
                source="checkpoint",
                checkpoint_path=stage2_checkpoint,
                source_module="bottleneck",
            ),
            "decoder": ModuleInitializationSpec(
                source="checkpoint",
                checkpoint_path=stage2_checkpoint,
                source_module="decoder",
            ),
            "reconstruction_head": ModuleInitializationSpec(source="random"),
        },
        trainable_modules=[
            "encoder",
            "bottleneck",
            "decoder",
            "reconstruction_head",
        ],
        frozen_modules=[],
        checkpoint_output=checkpoint_output,
        loss_name="mse",
    )


def default_downstream_spec(
    stage3_checkpoint: str,
    checkpoint_output: str = "downstream_best.pt",
) -> StageSpec:
    """Default downstream: backbone from stage3, random segmentation head."""
    return StageSpec(
        name="downstream_segmentation",
        forward_mode="segmentation",
        module_initialization={
            "encoder": ModuleInitializationSpec(
                source="checkpoint",
                checkpoint_path=stage3_checkpoint,
                source_module="encoder",
            ),
            "bottleneck": ModuleInitializationSpec(
                source="checkpoint",
                checkpoint_path=stage3_checkpoint,
                source_module="bottleneck",
            ),
            "decoder": ModuleInitializationSpec(
                source="checkpoint",
                checkpoint_path=stage3_checkpoint,
                source_module="decoder",
            ),
            "segmentation_head": ModuleInitializationSpec(source="random"),
        },
        trainable_modules=[
            "encoder",
            "bottleneck",
            "decoder",
            "segmentation_head",
        ],
        frozen_modules=[],
        checkpoint_output=checkpoint_output,
        loss_name="ce",
        checkpoint_input=stage3_checkpoint,
    )
