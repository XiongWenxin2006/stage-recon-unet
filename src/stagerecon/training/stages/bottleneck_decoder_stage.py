"""Stage 2: bottleneck + decoder focused pretraining on full reconstruction."""

from __future__ import annotations

from typing import Any

from stagerecon.training.checkpoint_manager import CheckpointManager
from stagerecon.training.parameter_controller import ParameterController
from stagerecon.training.stage_spec import StageSpec, default_stage2_spec
from stagerecon.training.stages.base_stage import BaseStage


class BottleneckDecoderStage(BaseStage):
    """Stage-2 wrapper: load bottleneck from stage1; typically freeze encoder.

    Uses the full ModularUNet reconstruction forward. Encoder is usually frozen
    so it receives no gradients (``frozen_modules=['encoder']``).
    """

    def __init__(
        self,
        spec: StageSpec | None = None,
        *,
        stage1_checkpoint: str | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        parameter_controller: ParameterController | None = None,
        **spec_overrides: Any,
    ) -> None:
        if spec is None:
            if not stage1_checkpoint:
                raise ValueError(
                    "BottleneckDecoderStage requires `spec` or `stage1_checkpoint`."
                )
            spec = default_stage2_spec(stage1_checkpoint)
            if spec_overrides:
                spec = StageSpec.from_config({**spec.__dict__, **spec_overrides})
        super().__init__(
            spec,
            checkpoint_manager=checkpoint_manager,
            parameter_controller=parameter_controller,
        )

    def stage_kind(self) -> str:
        return "bottleneck_decoder"
