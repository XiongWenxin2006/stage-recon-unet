"""Binary segmentation metric edge cases and foreground_iou vs mIoU."""

from __future__ import annotations

import numpy as np
import torch

from stagerecon.evaluation.segmentation_metrics import (
    compute_binary_segmentation_metrics,
)


def _metrics(pred, target, *, from_logits: bool = False):
    return compute_binary_segmentation_metrics(
        pred, target, from_logits=from_logits, threshold=0.5
    )


def test_perfect_match():
    target = np.array([[0, 1], [1, 0]], dtype=np.int64)
    pred = target.copy()
    m = _metrics(pred, target)
    assert m["foreground_iou"] == 1.0
    assert m["background_iou"] == 1.0
    assert m["mIoU"] == 1.0
    assert m["dice"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["tp"] == 2.0 and m["fp"] == 0.0 and m["fn"] == 0.0 and m["tn"] == 2.0


def test_completely_opposite():
    target = np.array([[0, 1], [1, 0]], dtype=np.int64)
    pred = 1 - target
    m = _metrics(pred, target)
    assert m["foreground_iou"] == 0.0
    assert m["dice"] == 0.0
    assert m["tp"] == 0.0
    assert m["fp"] == 2.0
    assert m["fn"] == 2.0


def test_all_background():
    target = np.zeros((4, 4), dtype=np.int64)
    pred = np.zeros((4, 4), dtype=np.int64)
    m = _metrics(pred, target)
    assert m["tp"] == 0.0 and m["fp"] == 0.0 and m["fn"] == 0.0
    assert m["tn"] == 16.0
    assert m["foreground_iou"] == 0.0  # undefined FG → safe_divide default 0
    assert m["background_iou"] == 1.0
    assert m["mIoU"] == 0.5


def test_all_foreground():
    target = np.ones((3, 3), dtype=np.int64)
    pred = np.ones((3, 3), dtype=np.int64)
    m = _metrics(pred, target)
    assert m["foreground_iou"] == 1.0
    assert m["background_iou"] == 0.0
    assert m["mIoU"] == 0.5
    assert m["dice"] == 1.0


def test_pred_empty_label_nonempty():
    target = np.zeros((4, 4), dtype=np.int64)
    target[1:3, 1:3] = 1
    pred = np.zeros_like(target)
    m = _metrics(pred, target)
    assert m["tp"] == 0.0
    assert m["fn"] == 4.0
    assert m["fp"] == 0.0
    assert m["foreground_iou"] == 0.0
    assert m["sensitivity"] == 0.0


def test_pred_nonempty_label_empty():
    target = np.zeros((4, 4), dtype=np.int64)
    pred = np.zeros((4, 4), dtype=np.int64)
    pred[0, 0] = 1
    m = _metrics(pred, target)
    assert m["tp"] == 0.0
    assert m["fp"] == 1.0
    assert m["fn"] == 0.0
    assert m["foreground_iou"] == 0.0
    assert m["precision"] == 0.0


def test_both_empty():
    target = np.zeros((2, 2), dtype=np.int64)
    pred = np.zeros((2, 2), dtype=np.int64)
    m = _metrics(pred, target)
    assert m["tp"] == 0.0 and m["fp"] == 0.0 and m["fn"] == 0.0
    assert m["tn"] == 4.0
    assert m["foreground_iou"] == 0.0
    assert m["background_iou"] == 1.0


def test_foreground_iou_vs_miou_distinction():
    """Crafted case: high background IoU, low foreground IoU → mIoU ≠ foreground_iou."""
    # 10x10 grid: mostly background. One FG pixel in label; pred misses it and
    # predicts a different FG pixel → TP=0, FP=1, FN=1, TN=98.
    target = np.zeros((10, 10), dtype=np.int64)
    pred = np.zeros((10, 10), dtype=np.int64)
    target[0, 0] = 1
    pred[0, 1] = 1
    m = _metrics(pred, target)

    fg = m["foreground_iou"]
    bg = m["background_iou"]
    miou = m["mIoU"]
    assert fg == 0.0
    assert bg == 98.0 / (98.0 + 1.0 + 1.0)  # TN / (TN+FP+FN)
    assert miou == 0.5 * (fg + bg)
    assert miou != fg
    assert miou > fg


def test_logits_path_perfect_match():
    target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    # Large positive/negative logits → confident correct classes after sigmoid.
    logits = torch.tensor([[-10.0, 10.0], [10.0, -10.0]])
    m = compute_binary_segmentation_metrics(logits, target, from_logits=True)
    assert m["foreground_iou"] == 1.0
    assert m["mIoU"] == 1.0
