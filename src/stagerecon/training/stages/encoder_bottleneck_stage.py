"""Stage 1: encoder + bottleneck focused pretraining on full reconstruction."""

from __future__ import annotations

from typing import Any

from stagerecon.training.checkpoint_manager import CheckpointManager
from stagerecon.training.parameter_controller import ParameterController
from stagerecon.training.stage_spec import StageSpec, default_stage1_spec
from stagerecon.training.stages.base_stage import BaseStage


class EncoderBottleneckStage(BaseStage):
    """Stage-1 wrapper: full ModularUNet, typically all-random init.

    Trainable / frozen sets come from :class:`StageSpec` (config-driven).
    Forward always uses the full reconstruction path when
    ``forward_mode='reconstruction'``.
    """

    def __init__(
        self,
        spec: StageSpec | None = None,
        *,
        checkpoint_manager: CheckpointManager | None = None,
        parameter_controller: ParameterController | None = None,
        **spec_overrides: Any,
    ) -> None:
        if spec is None:
            spec = default_stage1_spec()
            if spec_overrides:
                spec = StageSpec.from_config({**spec.__dict__, **spec_overrides})
        super().__init__(
            spec,
            checkpoint_manager=checkpoint_manager,
            parameter_controller=parameter_controller,
        )

    def stage_kind(self) -> str:
        return "encoder_bottleneck"
