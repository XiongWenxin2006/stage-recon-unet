"""Training stage wrappers for staged ModularUNet pretraining / fine-tuning."""

from stagerecon.training.stages.base_stage import (
    BaseStage,
    extract_batch_input,
    extract_batch_target,
)
from stagerecon.training.stages.bottleneck_decoder_stage import BottleneckDecoderStage
from stagerecon.training.stages.downstream_segmentation_stage import (
    DownstreamSegmentationStage,
)
from stagerecon.training.stages.encoder_bottleneck_stage import EncoderBottleneckStage
from stagerecon.training.stages.full_reconstruction_stage import FullReconstructionStage

__all__ = [
    "BaseStage",
    "BottleneckDecoderStage",
    "DownstreamSegmentationStage",
    "EncoderBottleneckStage",
    "FullReconstructionStage",
    "extract_batch_input",
    "extract_batch_target",
]
