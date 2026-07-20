"""Staged training package for ModularUNet pretraining and fine-tuning."""

from stagerecon.training.callbacks import (
    CallbackBase,
    CheckpointCallback,
    LoggingCallback,
)
from stagerecon.training.checkpoint_manager import CheckpointManager
from stagerecon.training.early_stopping import EarlyStopping
from stagerecon.training.module_access import KNOWN_MODULES, get_model_module
from stagerecon.training.optimizer_factory import build_optimizer
from stagerecon.training.parameter_controller import ParameterController
from stagerecon.training.scheduler_factory import build_scheduler
from stagerecon.training.stage_factory import build_stage
from stagerecon.training.stage_spec import (
    ModuleInitializationSpec,
    StageSpec,
    default_downstream_spec,
    default_stage1_spec,
    default_stage2_spec,
    default_stage3_spec,
)
from stagerecon.training.stages import (
    BaseStage,
    BottleneckDecoderStage,
    DownstreamSegmentationStage,
    EncoderBottleneckStage,
    FullReconstructionStage,
)
from stagerecon.training.trainer import Trainer

__all__ = [
    "BaseStage",
    "BottleneckDecoderStage",
    "CallbackBase",
    "CheckpointCallback",
    "CheckpointManager",
    "DownstreamSegmentationStage",
    "EarlyStopping",
    "EncoderBottleneckStage",
    "FullReconstructionStage",
    "KNOWN_MODULES",
    "LoggingCallback",
    "ModuleInitializationSpec",
    "ParameterController",
    "StageSpec",
    "Trainer",
    "build_optimizer",
    "build_scheduler",
    "build_stage",
    "default_downstream_spec",
    "default_stage1_spec",
    "default_stage2_spec",
    "default_stage3_spec",
    "get_model_module",
]
