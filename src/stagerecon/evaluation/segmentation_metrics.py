"""Binary segmentation metrics with a fixed ``labels=[0, 1]`` confusion matrix."""

from __future__ import annotations

import numpy as np
import torch


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide with a fallback when the denominator is (near) zero.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when ``abs(denominator) < 1e-12``.

    Returns:
        ``numerator / denominator`` or ``default``.
    """
    if abs(denominator) < 1e-12:
        return float(default)
    return float(numerator) / float(denominator)


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def logits_to_binary_mask(
    pred: torch.Tensor | np.ndarray,
    *,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> np.ndarray:
    """Convert logits or probabilities to a binary ``{0, 1}`` mask.

    Thresholding policy
    -------------------
    * When ``from_logits=True`` (default), ``pred`` is passed through a
      sigmoid and compared to ``threshold`` (default **0.5** on probabilities).
      This is preferred over thresholding raw logits at ``0.0``, though both
      are equivalent when ``threshold=0.5`` because ``sigmoid(0)=0.5``.
    * When ``from_logits=False``, ``pred`` is treated as probabilities (or
      already-thresholded values) and compared directly to ``threshold``.

    Args:
        pred: Logits or probabilities of any shape.
        threshold: Probability threshold after sigmoid (default 0.5).
        from_logits: Whether to apply sigmoid first.

    Returns:
        ``np.ndarray`` of dtype ``int64`` with values in ``{0, 1}``.
    """
    arr = _to_numpy(pred).astype(np.float64, copy=False)
    if from_logits:
        # Stable sigmoid
        probs = np.where(
            arr >= 0,
            1.0 / (1.0 + np.exp(-arr)),
            np.exp(arr) / (1.0 + np.exp(arr)),
        )
    else:
        probs = arr
    return (probs >= threshold).astype(np.int64)


def _confusion_counts(pred_bin: np.ndarray, target_bin: np.ndarray) -> tuple[int, int, int, int]:
    """Return TP, FP, TN, FN using fixed labels ``[0, 1]``.

    Empty / all-background / all-foreground cases are handled naturally:
    unused classes simply contribute zero counts.
    """
    pred_flat = pred_bin.reshape(-1).astype(np.int64, copy=False)
    target_flat = target_bin.reshape(-1).astype(np.int64, copy=False)

    # Clip to {0, 1} so unexpected labels do not break the fixed matrix
    pred_flat = np.clip(pred_flat, 0, 1)
    target_flat = np.clip(target_flat, 0, 1)

    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(target_flat, pred_flat, labels=[0, 1])
        # cm[i, j] = true label i, predicted label j
        tn = int(cm[0, 0])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])
        tp = int(cm[1, 1])
        return tp, fp, tn, fn
    except Exception:
        tp = int(np.sum((pred_flat == 1) & (target_flat == 1)))
        fp = int(np.sum((pred_flat == 1) & (target_flat == 0)))
        tn = int(np.sum((pred_flat == 0) & (target_flat == 0)))
        fn = int(np.sum((pred_flat == 0) & (target_flat == 1)))
        return tp, fp, tn, fn


def compute_binary_segmentation_metrics(
    pred_logits: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> dict[str, float]:
    """Compute binary segmentation metrics from logits and a ground-truth mask.

    Metrics (all floats):

    * ``accuracy``
    * ``sensitivity`` / ``recall`` (aliases; both present)
    * ``specificity``
    * ``precision``
    * ``dice`` / ``f1`` (aliases; both present)
    * ``foreground_iou`` – ``TP / (TP + FP + FN)``
    * ``background_iou`` – ``TN / (TN + FP + FN)``
    * ``mIoU`` – mean of foreground and background IoU
    * ``tp``, ``fp``, ``tn``, ``fn`` – raw confusion counts

    Predictions are converted to binary masks by applying sigmoid (when
    ``from_logits=True``) and thresholding probabilities at ``threshold``
    (default **0.5**). See :func:`logits_to_binary_mask`.

    Edge cases (all-background / all-foreground predictions or targets) are
    handled via a fixed ``labels=[0, 1]`` confusion matrix so missing classes
    contribute zeros rather than raising errors. Ratios use :func:`safe_divide`
    and default to ``0.0`` when undefined.

    Args:
        pred_logits: Model logits (or probabilities if ``from_logits=False``).
        target: Binary ground-truth mask.
        threshold: Probability threshold after sigmoid.
        from_logits: Whether ``pred_logits`` are raw logits.

    Returns:
        Dictionary of metric name → float value.
    """
    pred_bin = logits_to_binary_mask(
        pred_logits, threshold=threshold, from_logits=from_logits
    )
    target_bin = _to_numpy(target)
    # Accept soft targets by thresholding at 0.5
    if target_bin.dtype.kind == "f":
        target_bin = (target_bin >= 0.5).astype(np.int64)
    else:
        target_bin = target_bin.astype(np.int64, copy=False)
    target_bin = np.clip(target_bin, 0, 1)

    tp, fp, tn, fn = _confusion_counts(pred_bin, target_bin)

    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    foreground_iou = safe_divide(tp, tp + fp + fn)
    background_iou = safe_divide(tn, tn + fp + fn)
    miou = 0.5 * (foreground_iou + background_iou)

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "dice": dice,
        "f1": dice,
        "foreground_iou": foreground_iou,
        "background_iou": background_iou,
        "mIoU": miou,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }
