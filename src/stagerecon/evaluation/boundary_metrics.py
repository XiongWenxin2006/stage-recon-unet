"""Boundary-based segmentation metrics (Hausdorff Distance 95)."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def nanmean(values: Iterable[float], *, empty: float = float("nan")) -> float:
    """Mean that ignores NaNs.

    Args:
        values: Iterable of floats (may include NaN).
        empty: Value returned when there are no finite values.

    Returns:
        NaN-aware mean, or ``empty`` if no finite entries exist.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float(empty)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return float(empty)
    return float(np.mean(arr[mask]))


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.dtype == np.bool_:
        return arr
    return arr > 0


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask of surface voxels/pixels (erosion XOR original)."""
    binary = _as_bool_mask(mask)
    if not np.any(binary):
        return binary
    structure = ndi.generate_binary_structure(binary.ndim, 1)
    eroded = ndi.binary_erosion(binary, structure=structure, border_value=0)
    return np.logical_xor(binary, eroded)


def _surface_distances(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] | None,
) -> np.ndarray:
    """Directed surface distances from ``pred`` boundary to ``target`` boundary.

    Uses Euclidean distance transform on the complement of the target
    boundary so each predicted boundary point receives its nearest-surface
    distance. This avoids an all-pairs flatten + percentile over the full
    volume.
    """
    pred_b = _boundary_mask(pred)
    target_b = _boundary_mask(target)

    if not np.any(pred_b):
        return np.zeros(0, dtype=np.float64)

    # If target has no boundary but has interior (tiny object), treat the
    # object itself as the surface reference.
    if not np.any(target_b):
        if np.any(_as_bool_mask(target)):
            target_b = _as_bool_mask(target)
        else:
            return np.zeros(0, dtype=np.float64)

    spacing_t = None if spacing is None else tuple(float(s) for s in spacing)

    # Distance to nearest target boundary voxel; EDT of ~boundary.
    distances = ndi.distance_transform_edt(~target_b, sampling=spacing_t)
    return distances[pred_b].astype(np.float64, copy=False)


def _surface_distances_kdtree(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] | None,
) -> np.ndarray:
    """Directed distances via cKDTree on boundary coordinates (fallback)."""
    pred_b = _boundary_mask(pred)
    target_b = _boundary_mask(target)
    if not np.any(pred_b):
        return np.zeros(0, dtype=np.float64)
    if not np.any(target_b):
        if np.any(_as_bool_mask(target)):
            target_b = _as_bool_mask(target)
        else:
            return np.zeros(0, dtype=np.float64)

    pred_pts = np.argwhere(pred_b).astype(np.float64)
    target_pts = np.argwhere(target_b).astype(np.float64)
    if spacing is not None:
        scale = np.asarray(list(spacing), dtype=np.float64)
        pred_pts = pred_pts * scale
        target_pts = target_pts * scale

    tree = cKDTree(target_pts)
    dists, _ = tree.query(pred_pts, k=1)
    return np.atleast_1d(dists).astype(np.float64, copy=False)


def hausdorff_distance_95(
    pred: np.ndarray | object,
    target: np.ndarray | object,
    spacing: Sequence[float] | None = None,
    empty_value: float = float("nan"),
    *,
    method: str = "edt",
) -> float:
    """Compute the bidirectional 95th-percentile Hausdorff distance (HD95).

    For each direction (pred→target and target→pred), nearest surface
    distances are computed on **boundary points only** using either:

    * ``method="edt"`` (default): :func:`scipy.ndimage.distance_transform_edt`
    * ``method="kdtree"``: :class:`scipy.spatial.cKDTree` on boundary coords

    HD95 is the 95th percentile of the concatenated bidirectional distances.

    Empty-mask handling
    -------------------
    * Both masks empty → ``0.0``
    * Exactly one mask empty → ``empty_value`` (default ``NaN``)

    Dimensionality
    --------------
    * **2D** fully supported (``H×W`` or ``1×H×W`` squeezed).
    * **3D** supported for volumetric masks (``D×H×W``).

    Args:
        pred: Predicted binary mask (array-like).
        target: Ground-truth binary mask (array-like).
        spacing: Optional per-axis spacing (e.g. ``(sy, sx)`` or
            ``(sz, sy, sx)``) passed to the distance transform / point scaling.
        empty_value: Value returned when exactly one mask is empty.
        method: ``"edt"`` or ``"kdtree"``.

    Returns:
        HD95 as a Python ``float`` (may be NaN).
    """
    pred_arr = _as_bool_mask(np.asarray(pred))
    target_arr = _as_bool_mask(np.asarray(target))

    # Squeeze leading singleton channel dims commonly used in tensors
    while pred_arr.ndim > 3 and pred_arr.shape[0] == 1:
        pred_arr = np.squeeze(pred_arr, axis=0)
    while target_arr.ndim > 3 and target_arr.shape[0] == 1:
        target_arr = np.squeeze(target_arr, axis=0)

    if pred_arr.shape != target_arr.shape:
        raise ValueError(
            f"pred and target shapes must match; got {pred_arr.shape} vs {target_arr.shape}"
        )
    if pred_arr.ndim not in (2, 3):
        raise ValueError(
            f"HD95 supports 2D or 3D masks; got ndim={pred_arr.ndim}. "
            "For batched inputs, call per-sample."
        )

    if spacing is not None and len(spacing) != pred_arr.ndim:
        raise ValueError(
            f"spacing length ({len(spacing)}) must match mask ndim ({pred_arr.ndim})"
        )

    pred_empty = not np.any(pred_arr)
    target_empty = not np.any(target_arr)

    if pred_empty and target_empty:
        return 0.0
    if pred_empty or target_empty:
        return float(empty_value)

    dist_fn = _surface_distances if method == "edt" else _surface_distances_kdtree
    if method not in {"edt", "kdtree"}:
        raise ValueError(f"Unknown method {method!r}; expected 'edt' or 'kdtree'.")

    d_pt = dist_fn(pred_arr, target_arr, spacing)
    d_tp = dist_fn(target_arr, pred_arr, spacing)
    all_d = np.concatenate([d_pt, d_tp]) if d_pt.size or d_tp.size else np.zeros(0)

    if all_d.size == 0:
        return 0.0
    return float(np.percentile(all_d, 95))


def aggregate_hd95(
    values: Iterable[float],
    *,
    empty: float = float("nan"),
) -> float:
    """Aggregate per-sample HD95 scores with :func:`nanmean`."""
    return nanmean(values, empty=empty)
