"""Batch evaluation loop for reconstruction and segmentation modes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stagerecon.evaluation.boundary_metrics import hausdorff_distance_95, nanmean
from stagerecon.evaluation.prediction_exporter import PredictionExporter
from stagerecon.evaluation.reconstruction_metrics import mae, mse, psnr
from stagerecon.evaluation.segmentation_metrics import (
    compute_binary_segmentation_metrics,
)


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        converted = [_move_to_device(v, device) for v in batch]
        return type(batch)(converted)
    return batch


def _extract_pair(
    batch: Any,
    *,
    input_key: str = "image",
    target_key: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract ``(input, target)`` from common batch layouts."""
    if isinstance(batch, Mapping):
        x = batch.get(input_key, batch.get("x", batch.get("input")))
        if target_key is not None:
            y = batch.get(target_key)
        else:
            y = batch.get(
                "target",
                batch.get("label", batch.get("mask", batch.get("y", x))),
            )
        if x is None or y is None:
            raise KeyError(
                f"Batch mapping missing input/target keys. "
                f"Got keys={list(batch.keys())}"
            )
        return x, y

    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Tuple/list batch must provide at least (input, target).")
        return batch[0], batch[1]

    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def _unwrap_prediction(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "prediction"):
        return output.prediction
    if isinstance(output, Mapping) and "prediction" in output:
        return output["prediction"]
    raise TypeError(
        f"Cannot extract prediction tensor from model output of type {type(output)!r}"
    )


class Evaluator:
    """Run a model over a dataloader and aggregate recon / seg metrics.

    Args:
        model: PyTorch module. Prefer modules that accept
            ``forward(x, mode=...)`` (e.g. ``ModularUNet``).
        device: Torch device.
        mode: ``"reconstruction"``, ``"bottleneck_reconstruction"``, or
            ``"segmentation"``. Shorthand ``"recon"`` / ``"seg"`` accepted.
        exporter: Optional :class:`PredictionExporter` for saving ``.npy``
            predictions.
        compute_hd95: If True (segmentation mode), also compute HD95 per
            sample (can be slow).
        hd95_spacing: Optional spacing for HD95.
        input_key: Batch dict key for the network input.
        target_key: Optional batch dict key for the target. Defaults to
            common aliases / the input itself for reconstruction.
    """

    RECON_MODES = frozenset(
        {"reconstruction", "recon", "bottleneck_reconstruction", "btn_recon"}
    )
    SEG_MODES = frozenset({"segmentation", "seg"})

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = "cpu",
        mode: str = "segmentation",
        *,
        exporter: PredictionExporter | None = None,
        compute_hd95: bool = False,
        hd95_spacing: Sequence[float] | None = None,
        input_key: str = "image",
        target_key: str | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.mode = mode.lower()
        self.exporter = exporter
        self.compute_hd95 = compute_hd95
        self.hd95_spacing = hd95_spacing
        self.input_key = input_key
        self.target_key = target_key

        if self.mode not in self.RECON_MODES | self.SEG_MODES:
            raise ValueError(
                f"Unsupported evaluation mode '{mode}'. "
                f"Expected one of {sorted(self.RECON_MODES | self.SEG_MODES)}."
            )

    @property
    def is_segmentation(self) -> bool:
        return self.mode in self.SEG_MODES

    @property
    def forward_mode(self) -> str:
        if self.mode in {"recon", "reconstruction"}:
            return "reconstruction"
        if self.mode in {"btn_recon", "bottleneck_reconstruction"}:
            return "bottleneck_reconstruction"
        return "segmentation"

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            output = self.model(x, mode=self.forward_mode)
        except TypeError:
            # Models that do not accept a mode kwarg
            output = self.model(x)
        return _unwrap_prediction(output)

    def _recon_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, float]:
        return {
            "mse": float(mse(pred, target).item()),
            "mae": float(mae(pred, target).item()),
            "psnr": float(psnr(pred, target).item()),
        }

    def _seg_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, float]:
        metrics = compute_binary_segmentation_metrics(pred, target)
        if self.compute_hd95:
            # Per-sample HD95, then nanmean
            b = pred.shape[0]
            hd_vals: list[float] = []
            pred_np = torch.sigmoid(pred).detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            for i in range(b):
                p_mask = (pred_np[i] >= 0.5).astype(np.uint8)
                t_mask = (target_np[i] >= 0.5).astype(np.uint8)
                # Squeeze channel dim if present
                if p_mask.ndim == 3 and p_mask.shape[0] == 1:
                    p_mask = p_mask[0]
                if t_mask.ndim == 3 and t_mask.shape[0] == 1:
                    t_mask = t_mask[0]
                if p_mask.ndim == 4 and p_mask.shape[0] == 1:
                    p_mask = p_mask[0]
                if t_mask.ndim == 4 and t_mask.shape[0] == 1:
                    t_mask = t_mask[0]
                hd_vals.append(
                    hausdorff_distance_95(
                        p_mask, t_mask, spacing=self.hd95_spacing
                    )
                )
            metrics["hd95"] = nanmean(hd_vals)
        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader | Iterable[Any],
        *,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Evaluate the model and return mean metrics over the dataset.

        Args:
            dataloader: Iterable of batches.
            max_batches: Optional cap on the number of batches.

        Returns:
            Dict of metric name → mean value across batches (batch-weighted
            for count-like confusion totals; simple mean for ratios).
        """
        self.model.eval()
        self.model.to(self.device)

        sums: dict[str, float] = defaultdict(float)
        weight_sum = 0.0
        sample_index = 0

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = _move_to_device(batch, self.device)
            x, y = _extract_pair(
                batch, input_key=self.input_key, target_key=self.target_key
            )
            pred = self._forward(x)

            batch_size = int(x.shape[0]) if isinstance(x, torch.Tensor) else 1

            if self.is_segmentation:
                metrics = self._seg_metrics(pred, y)
            else:
                metrics = self._recon_metrics(pred, y)

            for key, value in metrics.items():
                if key in {"tp", "fp", "tn", "fn"}:
                    sums[key] += float(value)
                else:
                    sums[key] += float(value) * batch_size
            weight_sum += batch_size

            if self.exporter is not None:
                self.exporter.export_batch(pred, start_index=sample_index)
            sample_index += batch_size

        if weight_sum <= 0:
            return {}

        aggregated: dict[str, float] = {}
        for key, total in sums.items():
            if key in {"tp", "fp", "tn", "fn"}:
                aggregated[key] = float(total)
            else:
                aggregated[key] = float(total) / weight_sum

        # Recompute ratio metrics from pooled confusion counts when available
        if self.is_segmentation and all(k in aggregated for k in ("tp", "fp", "tn", "fn")):
            from stagerecon.evaluation.segmentation_metrics import safe_divide

            tp, fp, tn, fn = (
                aggregated["tp"],
                aggregated["fp"],
                aggregated["tn"],
                aggregated["fn"],
            )
            aggregated["accuracy"] = safe_divide(tp + tn, tp + tn + fp + fn)
            aggregated["sensitivity"] = safe_divide(tp, tp + fn)
            aggregated["recall"] = aggregated["sensitivity"]
            aggregated["specificity"] = safe_divide(tn, tn + fp)
            aggregated["precision"] = safe_divide(tp, tp + fp)
            aggregated["dice"] = safe_divide(2 * tp, 2 * tp + fp + fn)
            aggregated["f1"] = aggregated["dice"]
            aggregated["foreground_iou"] = safe_divide(tp, tp + fp + fn)
            aggregated["background_iou"] = safe_divide(tn, tn + fp + fn)
            aggregated["mIoU"] = 0.5 * (
                aggregated["foreground_iou"] + aggregated["background_iou"]
            )

        return aggregated
