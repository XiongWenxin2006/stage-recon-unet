"""Optional helpers to export model predictions to disk."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch


class PredictionExporter:
    """Save prediction tensors as NumPy ``.npy`` files.

    Args:
        output_dir: Directory where files are written.
        enabled: If False, :meth:`export` is a no-op.
        prefix: Filename prefix.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        enabled: bool = True,
        prefix: str = "pred",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.prefix = prefix
        self._counter = 0
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        prediction: torch.Tensor | np.ndarray,
        *,
        name: str | None = None,
        index: int | None = None,
    ) -> Path | None:
        """Write a single prediction array to ``.npy``.

        Args:
            prediction: Tensor or ndarray to save (moved to CPU if needed).
            name: Optional explicit stem (without extension). When omitted,
                files are named ``{prefix}_{index:06d}.npy``.
            index: Optional index used when ``name`` is omitted. Defaults to
                an internal monotonic counter.

        Returns:
            Path of the written file, or ``None`` if exporting is disabled.
        """
        if not self.enabled:
            return None

        if isinstance(prediction, torch.Tensor):
            array = prediction.detach().cpu().numpy()
        else:
            array = np.asarray(prediction)

        if name is None:
            if index is None:
                index = self._counter
                self._counter += 1
            name = f"{self.prefix}_{int(index):06d}"

        path = self.output_dir / f"{name}.npy"
        np.save(path, array)
        return path

    def export_batch(
        self,
        predictions: torch.Tensor | np.ndarray,
        *,
        names: list[str] | None = None,
        start_index: int | None = None,
    ) -> list[Path]:
        """Export each item along the batch dimension.

        Args:
            predictions: Batched predictions ``(B, ...)``.
            names: Optional per-sample stems.
            start_index: Starting index when ``names`` is omitted.

        Returns:
            List of written paths (empty when disabled).
        """
        if not self.enabled:
            return []

        if isinstance(predictions, torch.Tensor):
            batch = predictions.detach().cpu()
            length = int(batch.shape[0])
        else:
            batch = np.asarray(predictions)
            length = int(batch.shape[0])

        paths: list[Path] = []
        base = self._counter if start_index is None else int(start_index)
        for i in range(length):
            sample_name = names[i] if names is not None else None
            idx = None if sample_name is not None else base + i
            path = self.export(batch[i], name=sample_name, index=idx)
            if path is not None:
                paths.append(path)
        if names is None and start_index is None:
            self._counter = base + length
        return paths


def save_prediction_npy(
    prediction: torch.Tensor | np.ndarray,
    path: str | Path,
) -> Path:
    """Convenience function to save one prediction array as ``.npy``.

    Args:
        prediction: Tensor or ndarray.
        path: Destination path (``.npy`` appended if missing).

    Returns:
        Resolved output path.
    """
    out = Path(path)
    if out.suffix != ".npy":
        out = out.with_suffix(".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(prediction, torch.Tensor):
        array = prediction.detach().cpu().numpy()
    else:
        array = np.asarray(prediction)
    np.save(out, array)
    return out
