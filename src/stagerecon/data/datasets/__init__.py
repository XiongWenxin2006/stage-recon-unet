"""Dataset implementations and registry for StageRecon."""

from stagerecon.data.datasets.dataset_registry import (
    build_dataset,
    get_dataset,
    list_datasets,
    register_dataset,
)
from stagerecon.data.datasets.reconstruction_dataset import LocalReconstructionDataset
from stagerecon.data.datasets.segmentation_dataset import LocalSegmentationDataset
from stagerecon.data.datasets.synthetic_dataset import (
    SyntheticDataset,
    SyntheticReconstructionDataset,
    build_synthetic_dataset,
)

__all__ = [
    "LocalReconstructionDataset",
    "LocalSegmentationDataset",
    "SyntheticDataset",
    "SyntheticReconstructionDataset",
    "build_dataset",
    "build_synthetic_dataset",
    "get_dataset",
    "list_datasets",
    "register_dataset",
]
