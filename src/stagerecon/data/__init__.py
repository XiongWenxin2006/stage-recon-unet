"""StageRecon data package: datasets, transforms, streaming, and dataloaders."""

from stagerecon.data.dataloader_factory import build_dataloader
from stagerecon.data.datasets import (
    LocalReconstructionDataset,
    LocalSegmentationDataset,
    SyntheticDataset,
    SyntheticReconstructionDataset,
    build_dataset,
    build_synthetic_dataset,
    list_datasets,
)
from stagerecon.data.sample_types import ReconstructionSample, SegmentationSample
from stagerecon.data.streaming import (
    build_shard_urls,
    build_webdataset,
    expand_brace_urls,
    rewrite_remote_url,
)

__all__ = [
    "LocalReconstructionDataset",
    "LocalSegmentationDataset",
    "ReconstructionSample",
    "SegmentationSample",
    "SyntheticDataset",
    "SyntheticReconstructionDataset",
    "build_dataloader",
    "build_dataset",
    "build_shard_urls",
    "build_synthetic_dataset",
    "build_webdataset",
    "expand_brace_urls",
    "list_datasets",
    "rewrite_remote_url",
]
