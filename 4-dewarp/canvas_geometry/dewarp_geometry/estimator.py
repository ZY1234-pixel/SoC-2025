"""Model-independent 3D arc-length aspect-ratio estimator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


Layout = Literal["auto", "hwc", "chw"]


@dataclass(frozen=True)
class AspectEstimate:
    aspect_ratio: float
    width_arc_length: float
    height_arc_length: float
    row_cv: float
    column_cv: float
    opposite_edge_error: float
    invalid_segment_fraction: float
    confidence: float
    valid: bool
    method: str = "robust_3d_arc_length_v1"
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["warnings"] = list(self.warnings)
        return result


def _as_numpy(grid: Any) -> np.ndarray:
    if hasattr(grid, "detach"):
        grid = grid.detach().cpu().numpy()
    return np.asarray(grid, dtype=np.float64)


def _normalise_layout(grid: Any, layout: Layout) -> np.ndarray:
    array = _as_numpy(grid)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("A batched grid must have batch size 1")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"grid3d must be 3-D or 1x3-D, got shape {array.shape}")

    if layout == "auto":
        first_is_xyz = array.shape[0] == 3
        last_is_xyz = array.shape[-1] == 3
        if first_is_xyz == last_is_xyz:
            raise ValueError(
                f"Cannot infer coordinate axis from shape {array.shape}; pass layout='chw' or 'hwc'"
            )
        layout = "chw" if first_is_xyz else "hwc"
    if layout == "chw":
        if array.shape[0] != 3:
            raise ValueError(f"CHW grid must have shape 3xHxW, got {array.shape}")
        array = np.moveaxis(array, 0, -1)
    elif layout == "hwc":
        if array.shape[-1] != 3:
            raise ValueError(f"HWC grid must have shape HxWx3, got {array.shape}")
    else:
        raise ValueError(f"Unknown layout: {layout}")
    if min(array.shape[:2]) < 2:
        raise ValueError("grid3d needs at least 2x2 vertices")
    return array


def _robust_location(values: np.ndarray, trim_ratio: float) -> float:
    finite = np.sort(values[np.isfinite(values)])
    if finite.size == 0:
        return float("nan")
    trim = int(np.floor(finite.size * trim_ratio))
    if trim and finite.size - 2 * trim >= 1:
        finite = finite[trim:-trim]
    return float(np.median(finite))


def _cv(values: np.ndarray, centre: float, eps: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not np.isfinite(centre) or centre <= eps:
        return float("inf")
    mad = np.median(np.abs(finite - np.median(finite)))
    return float(1.4826 * mad / centre)


def estimate_aspect_from_grid(
    grid3d: Any,
    *,
    layout: Layout = "auto",
    valid_mask: Any | None = None,
    trim_ratio: float = 0.10,
    min_segment_length: float = 1e-8,
    max_invalid_fraction: float = 0.05,
) -> AspectEstimate:
    """Estimate intrinsic page W/H by integrating the predicted 3D grid.

    Grid rows run across page width and grid columns run across page height.
    Uniform rotation, translation and scale do not change the returned ratio.
    ``confidence`` is a diagnostic score, not a calibrated probability.
    """
    if not 0.0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5)")
    grid = _normalise_layout(grid3d, layout)
    vertex_valid = np.isfinite(grid).all(axis=-1)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != grid.shape[:2]:
            raise ValueError(f"valid_mask shape {mask.shape} != grid shape {grid.shape[:2]}")
        vertex_valid &= mask

    horizontal = np.linalg.norm(np.diff(grid, axis=1), axis=-1)
    vertical = np.linalg.norm(np.diff(grid, axis=0), axis=-1)
    h_valid = vertex_valid[:, 1:] & vertex_valid[:, :-1] & (horizontal > min_segment_length)
    v_valid = vertex_valid[1:] & vertex_valid[:-1] & (vertical > min_segment_length)
    horizontal = np.where(h_valid, horizontal, np.nan)
    vertical = np.where(v_valid, vertical, np.nan)

    row_lengths = np.nansum(horizontal, axis=1)
    column_lengths = np.nansum(vertical, axis=0)
    row_coverage = np.mean(h_valid, axis=1)
    column_coverage = np.mean(v_valid, axis=0)
    row_lengths[row_coverage < 0.95] = np.nan
    column_lengths[column_coverage < 0.95] = np.nan

    width = _robust_location(row_lengths, trim_ratio)
    height = _robust_location(column_lengths, trim_ratio)
    eps = min_segment_length
    ratio = width / height if np.isfinite(width) and np.isfinite(height) and height > eps else np.nan
    row_cv = _cv(row_lengths, width, eps)
    column_cv = _cv(column_lengths, height, eps)

    edge_values = np.array([row_lengths[0], row_lengths[-1], column_lengths[0], column_lengths[-1]])
    if np.isfinite(edge_values).all() and width > eps and height > eps:
        edge_error = max(
            abs(row_lengths[0] - row_lengths[-1]) / width,
            abs(column_lengths[0] - column_lengths[-1]) / height,
        )
    else:
        edge_error = float("inf")

    total_segments = h_valid.size + v_valid.size
    valid_segments = int(h_valid.sum() + v_valid.sum())
    invalid_fraction = 1.0 - valid_segments / total_segments
    warnings: list[str] = []
    if invalid_fraction > 0:
        warnings.append("invalid_or_degenerate_grid_segments")
    if row_cv > 0.10:
        warnings.append("high_row_length_dispersion")
    if column_cv > 0.10:
        warnings.append("high_column_length_dispersion")
    if edge_error > 0.15:
        warnings.append("opposite_edges_inconsistent")

    valid = bool(
        np.isfinite(ratio)
        and ratio > 0
        and invalid_fraction <= max_invalid_fraction
        and np.isfinite(row_cv)
        and np.isfinite(column_cv)
    )
    if not valid:
        warnings.append("estimate_invalid")

    # Smooth diagnostic score. This is intentionally conservative and uncalibrated.
    penalties = [row_cv, column_cv, edge_error, invalid_fraction * 4.0]
    finite_penalties = [p for p in penalties if np.isfinite(p)]
    confidence = float(np.exp(-4.0 * max(finite_penalties, default=float("inf")))) if valid else 0.0
    return AspectEstimate(
        aspect_ratio=float(ratio),
        width_arc_length=float(width),
        height_arc_length=float(height),
        row_cv=float(row_cv),
        column_cv=float(column_cv),
        opposite_edge_error=float(edge_error),
        invalid_segment_fraction=float(invalid_fraction),
        confidence=confidence,
        valid=valid,
        warnings=tuple(dict.fromkeys(warnings)),
    )
