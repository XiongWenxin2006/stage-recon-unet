"""Stage 3: assemble encoder/bottleneck/decoder for full reconstruction."""

from __future__ import annotations

from typing import Any

from stagerecon.training.checkpoint_manager import CheckpointManager
from stagerecon.training.parameter_controller import ParameterController
from stagerecon.training.stage_spec import StageSpec, default_stage3_spec
from stagerecon.training.stages.base_stage import BaseStage


class FullReconstructionStage(BaseStage):
    """Stage-3 wrapper: encoder from s1, bottleneck+decoder from s2.

    Reconstruction head is typically randomly initialized; all listed modules
    participate in the full reconstruction forward.
    """

    def __init__(
        self,
        spec: StageSpec | None = None,
        *,
        stage1_checkpoint: str | None = None,
        stage2_checkpoint: str | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        parameter_controller: ParameterController | None = None,
        **spec_overrides: Any,
    ) -> None:
        if spec is None:
            if not stage1_checkpoint or not stage2_checkpoint:
                raise ValueError(
                    "FullReconstructionStage requires `spec` or both "
                    "`stage1_checkpoint` and `stage2_checkpoint`."
                )
            spec = default_stage3_spec(stage1_checkpoint, stage2_checkpoint)
            if spec_overrides:
                spec = StageSpec.from_config({**spec.__dict__, **spec_overrides})
        super().__init__(
            spec,
            checkpoint_manager=checkpoint_manager,
            parameter_controller=parameter_controller,
        )

    def stage_kind(self) -> str:
        return "full_reconstruction"
