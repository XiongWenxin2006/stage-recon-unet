"""Hausdorff Distance 95 (HD95) geometric and empty-mask tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from stagerecon.evaluation.boundary_metrics import hausdorff_distance_95


def _disk(h: int, w: int, cy: int, cx: int, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2).astype(np.uint8)


def _boundary(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    binary = mask.astype(bool)
    structure = ndi.generate_binary_structure(binary.ndim, 1)
    eroded = ndi.binary_erosion(binary, structure=structure, border_value=0)
    return np.logical_xor(binary, eroded)


def _reference_surface_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """Independent bidirectional surface HD95 via boundary KDTree queries."""
    pred_pts = np.argwhere(_boundary(pred)).astype(np.float64)
    target_pts = np.argwhere(_boundary(target)).astype(np.float64)
    assert pred_pts.size > 0 and target_pts.size > 0
    d_pt, _ = cKDTree(target_pts).query(pred_pts, k=1)
    d_tp, _ = cKDTree(pred_pts).query(target_pts, k=1)
    return float(np.percentile(np.concatenate([d_pt, d_tp]), 95))


def _forbidden_full_pairwise_percentile(pred: np.ndarray, target: np.ndarray) -> float:
    """Incorrect approach: flatten the full boundary pairwise distance matrix."""
    pred_pts = np.argwhere(_boundary(pred)).astype(np.float64)
    target_pts = np.argwhere(_boundary(target)).astype(np.float64)
    # Full MxN pairwise matrix — must NOT be how HD95 is defined.
    diff = pred_pts[:, None, :] - target_pts[None, :, :]
    dists = np.sqrt((diff**2).sum(axis=-1)).ravel()
    return float(np.percentile(dists, 95))


def test_two_offset_disks_hd95_finite_positive():
    # Centers 24 px apart, radius 8 → surface gap ≈ 8 > 0.
    pred = _disk(64, 64, cy=24, cx=20, radius=8)
    target = _disk(64, 64, cy=24, cx=44, radius=8)
    hd = hausdorff_distance_95(pred, target)
    assert np.isfinite(hd)
    assert hd > 0.0


def test_both_empty_returns_zero():
    pred = np.zeros((16, 16), dtype=np.uint8)
    target = np.zeros((16, 16), dtype=np.uint8)
    assert hausdorff_distance_95(pred, target) == 0.0


def test_one_empty_returns_nan():
    pred = np.zeros((16, 16), dtype=np.uint8)
    target = _disk(16, 16, cy=8, cx=8, radius=3)
    assert np.isnan(hausdorff_distance_95(pred, target))
    assert np.isnan(hausdorff_distance_95(target, pred))


def test_bidirectional_nearest_surface_not_all_pairs_flatten():
    """HD95 uses nearest-surface distances, not a flattened full pairwise matrix."""
    pred = _disk(80, 80, cy=30, cx=30, radius=12)
    target = _disk(80, 80, cy=30, cx=55, radius=12)

    hd_surface = hausdorff_distance_95(pred, target, method="edt")
    hd_ref = _reference_surface_hd95(pred, target)
    hd_pairwise_flat = _forbidden_full_pairwise_percentile(pred, target)

    assert np.isfinite(hd_surface)
    assert hd_surface > 0.0
    # Implementation must match bidirectional nearest-surface reference.
    assert hd_surface == pytest.approx(hd_ref, rel=0, abs=1e-5)
    # Flattening the full MxN pairwise matrix yields a different statistic.
    assert hd_surface != pytest.approx(hd_pairwise_flat, rel=0, abs=1e-3)

    # Bidirectional HD95 is symmetric for the concatenated-percentile definition.
    thin = np.zeros((40, 40), dtype=np.uint8)
    thin[20, 5:15] = 1
    blob = _disk(40, 40, cy=20, cx=30, radius=4)
    hd_ab = hausdorff_distance_95(thin, blob)
    hd_ba = hausdorff_distance_95(blob, thin)
    assert hd_ab == pytest.approx(hd_ba, rel=0, abs=1e-9)

    # Boundary point counts are far smaller than FG volumes.
    assert _boundary(pred).sum() < (pred > 0).sum()
    assert _boundary(target).sum() < (target > 0).sum()


def test_kdtree_method_agrees_roughly_with_edt():
    pred = _disk(48, 48, cy=20, cx=18, radius=6)
    target = _disk(48, 48, cy=20, cx=30, radius=6)
    hd_edt = hausdorff_distance_95(pred, target, method="edt")
    hd_kd = hausdorff_distance_95(pred, target, method="kdtree")
    assert hd_edt == pytest.approx(hd_kd, rel=0, abs=1e-5)
