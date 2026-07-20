"""Downstream segmentation fine-tuning stage."""

from __future__ import annotations

from typing import Any

from stagerecon.training.checkpoint_manager import CheckpointManager
from stagerecon.training.parameter_controller import ParameterController
from stagerecon.training.stage_spec import StageSpec, default_downstream_spec
from stagerecon.training.stages.base_stage import BaseStage


class DownstreamSegmentationStage(BaseStage):
    """Downstream wrapper: backbone from stage3, random segmentation head.

    ``forward_mode`` is ``segmentation``.
    """

    def __init__(
        self,
        spec: StageSpec | None = None,
        *,
        stage3_checkpoint: str | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        parameter_controller: ParameterController | None = None,
        **spec_overrides: Any,
    ) -> None:
        if spec is None:
            if not stage3_checkpoint:
                raise ValueError(
                    "DownstreamSegmentationStage requires `spec` or "
                    "`stage3_checkpoint`."
                )
            spec = default_downstream_spec(stage3_checkpoint)
            if spec_overrides:
                spec = StageSpec.from_config({**spec.__dict__, **spec_overrides})
        super().__init__(
            spec,
            checkpoint_manager=checkpoint_manager,
            parameter_controller=parameter_controller,
        )

    def stage_kind(self) -> str:
        return "downstream_segmentation"
