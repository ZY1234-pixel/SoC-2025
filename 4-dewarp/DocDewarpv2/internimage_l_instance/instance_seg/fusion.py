"""Conservative local fusion of coarse and refined boundaries."""

from typing import Any, Dict, Tuple
import cv2
import numpy as np
from .geometry import local_edge_support


def fuse_candidate_with_coarse_mask(
    candidate: np.ndarray,
    coarse: np.ndarray,
    image_bgr: np.ndarray,
    analysis_scale: float,
    max_local_component_ratio: float,
    max_local_p95_shift: float,
    min_coarse_edge_advantage: float,
    contour_component_max_ratio: float,
    contour_min_p95_shift: float,
    contour_min_length_ratio: float,
    contour_min_edge_continuity_gap: float,
    contour_coarse_edge_advantage: float,
    contour_edge_magnitude: float,
    long_edge_component_max_ratio: float,
    long_edge_min_p95_shift: float,
    long_edge_min_elongation: float,
    long_edge_coarse_edge_advantage: float,
    long_edge_min_coarse_continuity: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Keep broad ROI corrections but suppress image-supported local mistakes.

    A useful second pass may remove a large document-like background, so a low
    global IoU is not automatically bad.  Conversely, a small but deep notch
    can be visually wrong while global IoU remains near one.  Small, deep
    difference components are therefore reverted only when the coarse boundary
    also has clearly stronger image-edge support than the ROI boundary.  This
    extra evidence preserves a useful ROI correction that moves an erroneous
    coarse boundary onto a visible paper edge.
    """
    candidate_bool = candidate.astype(bool, copy=False)
    coarse_bool = coarse.astype(bool, copy=False)
    if candidate_bool.shape != coarse_bool.shape:
        raise ValueError(
            f"Candidate/coarse shapes differ: {candidate_bool.shape} vs "
            f"{coarse_bool.shape}"
        )
    if image_bgr.shape[:2] != coarse_bool.shape:
        raise ValueError(
            f"Image/mask shapes differ: {image_bgr.shape[:2]} vs {coarse_bool.shape}"
        )
    fused = candidate_bool.copy()
    coarse_area = int(coarse_bool.sum())
    if coarse_area == 0:
        return fused, {
            "local_fusion_eligible_component_count": 0,
            "contour_fusion_eligible_component_count": 0,
            "contour_reverted_component_count": 0,
            "long_edge_fusion_eligible_component_count": 0,
            "long_edge_reverted_component_count": 0,
            "edge_preserved_component_count": 0,
            "reverted_component_count": 0,
            "reverted_pixel_count": 0,
            "largest_reverted_component_ratio": 0.0,
            "largest_reverted_p95_shift_input_px": 0.0,
            "largest_reverted_coarse_edge_advantage": 0.0,
            "largest_reverted_boundary_length_ratio": 0.0,
            "largest_reverted_edge_continuity_gap": 0.0,
            "largest_reverted_component_elongation": 0.0,
            "largest_reverted_coarse_edge_continuity": 0.0,
        }

    added = np.logical_and(candidate_bool, ~coarse_bool)
    removed = np.logical_and(coarse_bool, ~candidate_bool)
    outside_distance = cv2.distanceTransform(
        (~coarse_bool).astype(np.uint8), cv2.DIST_L2, 5
    )
    inside_distance = cv2.distanceTransform(
        coarse_bool.astype(np.uint8), cv2.DIST_L2, 5
    )
    boundary_kernel = np.ones((3, 3), dtype=np.uint8)
    coarse_boundary = (
        cv2.morphologyEx(
            coarse_bool.astype(np.uint8), cv2.MORPH_GRADIENT, boundary_kernel
        )
        > 0
    )
    candidate_boundary = (
        cv2.morphologyEx(
            candidate_bool.astype(np.uint8), cv2.MORPH_GRADIENT, boundary_kernel
        )
        > 0
    )
    eligible_count = 0
    contour_eligible_count = 0
    contour_reverted_count = 0
    long_edge_eligible_count = 0
    long_edge_reverted_count = 0
    edge_preserved_count = 0
    reverted_count = 0
    reverted_pixels = 0
    largest_ratio = 0.0
    largest_p95_shift = 0.0
    largest_edge_advantage = 0.0
    largest_boundary_length_ratio = 0.0
    largest_edge_continuity_gap = 0.0
    largest_component_elongation = 0.0
    largest_coarse_edge_continuity = 0.0

    for change_mask, distance_map, restore_value, change_kind in (
        (added, outside_distance, False, "added"),
        (removed, inside_distance, True, "removed"),
    ):
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            change_mask.astype(np.uint8), connectivity=8
        )
        for component_id in range(1, component_count):
            x, y, width, height, area = [int(value) for value in stats[component_id]]
            component_ratio = float(area / coarse_area)
            component_elongation = float(
                max(width, height) / max(1, min(width, height))
            )
            standard_geometry_candidate = component_ratio < max_local_component_ratio
            contour_geometry_candidate = (
                change_kind == "removed"
                and component_ratio < contour_component_max_ratio
            )
            long_edge_geometry_candidate = bool(
                change_kind == "removed"
                and component_ratio < long_edge_component_max_ratio
                and component_elongation >= long_edge_min_elongation
            )
            if not (
                standard_geometry_candidate
                or contour_geometry_candidate
                or long_edge_geometry_candidate
            ):
                continue
            label_region = labels[y : y + height, x : x + width]
            component_region = label_region == component_id
            shifts = distance_map[y : y + height, x : x + width][
                component_region
            ] * float(analysis_scale)
            if shifts.size == 0:
                continue
            p95_shift = float(np.percentile(shifts, 95))
            standard_candidate = (
                standard_geometry_candidate and p95_shift > max_local_p95_shift
            )
            contour_candidate = (
                contour_geometry_candidate and p95_shift > contour_min_p95_shift
            )
            long_edge_candidate = (
                long_edge_geometry_candidate and p95_shift > long_edge_min_p95_shift
            )
            if (
                not standard_candidate
                and not contour_candidate
                and not long_edge_candidate
            ):
                continue
            if standard_candidate:
                eligible_count += 1
            if contour_candidate:
                contour_eligible_count += 1
            if long_edge_candidate:
                long_edge_eligible_count += 1

            padding = 3
            padded_x1 = max(0, x - padding)
            padded_y1 = max(0, y - padding)
            padded_x2 = min(coarse_bool.shape[1], x + width + padding)
            padded_y2 = min(coarse_bool.shape[0], y + height + padding)
            component_padded = (
                labels[padded_y1:padded_y2, padded_x1:padded_x2] == component_id
            )
            local_support = (
                cv2.dilate(
                    component_padded.astype(np.uint8),
                    np.ones((5, 5), dtype=np.uint8),
                )
                > 0
            )
            edge_support = local_edge_support(
                image_bgr, (padded_x1, padded_y1, padded_x2, padded_y2)
            )
            coarse_values = edge_support[
                coarse_boundary[padded_y1:padded_y2, padded_x1:padded_x2]
                & local_support
            ]
            candidate_values = edge_support[
                candidate_boundary[padded_y1:padded_y2, padded_x1:padded_x2]
                & local_support
            ]
            if not coarse_values.size or not candidate_values.size:
                edge_preserved_count += 1
                continue
            coarse_edge_score = float(np.median(coarse_values))
            candidate_edge_score = float(np.median(candidate_values))
            edge_advantage = float(
                (coarse_edge_score + 1e-6) / (candidate_edge_score + 1e-6)
            )
            standard_revert = bool(
                standard_candidate
                and edge_advantage >= min_coarse_edge_advantage
                and coarse_edge_score >= candidate_edge_score + 5.0
            )
            candidate_to_coarse_length_ratio = float(
                candidate_values.size / max(1, coarse_values.size)
            )
            coarse_edge_continuity = float(
                np.mean(coarse_values >= contour_edge_magnitude)
            )
            candidate_edge_continuity = float(
                np.mean(candidate_values >= contour_edge_magnitude)
            )
            edge_continuity_gap = coarse_edge_continuity - candidate_edge_continuity
            contour_revert = bool(
                contour_candidate
                and candidate_to_coarse_length_ratio >= contour_min_length_ratio
                and edge_continuity_gap >= contour_min_edge_continuity_gap
                and edge_advantage >= contour_coarse_edge_advantage
                and coarse_edge_score >= candidate_edge_score + 5.0
            )
            long_edge_revert = bool(
                long_edge_candidate
                and coarse_edge_continuity >= long_edge_min_coarse_continuity
                and edge_advantage >= long_edge_coarse_edge_advantage
                and coarse_edge_score >= candidate_edge_score + 5.0
            )
            if not standard_revert and not contour_revert and not long_edge_revert:
                edge_preserved_count += 1
                continue

            fused_region = fused[y : y + height, x : x + width]
            fused_region[component_region] = restore_value
            reverted_count += 1
            if contour_revert:
                contour_reverted_count += 1
            if long_edge_revert:
                long_edge_reverted_count += 1
            reverted_pixels += area
            largest_ratio = max(largest_ratio, component_ratio)
            largest_p95_shift = max(largest_p95_shift, p95_shift)
            largest_edge_advantage = max(largest_edge_advantage, edge_advantage)
            largest_boundary_length_ratio = max(
                largest_boundary_length_ratio, candidate_to_coarse_length_ratio
            )
            largest_edge_continuity_gap = max(
                largest_edge_continuity_gap, edge_continuity_gap
            )
            largest_component_elongation = max(
                largest_component_elongation, component_elongation
            )
            largest_coarse_edge_continuity = max(
                largest_coarse_edge_continuity, coarse_edge_continuity
            )

    return fused, {
        "local_fusion_eligible_component_count": eligible_count,
        "contour_fusion_eligible_component_count": contour_eligible_count,
        "contour_reverted_component_count": contour_reverted_count,
        "long_edge_fusion_eligible_component_count": long_edge_eligible_count,
        "long_edge_reverted_component_count": long_edge_reverted_count,
        "edge_preserved_component_count": edge_preserved_count,
        "reverted_component_count": reverted_count,
        "reverted_pixel_count": reverted_pixels,
        "largest_reverted_component_ratio": largest_ratio,
        "largest_reverted_p95_shift_input_px": largest_p95_shift,
        "largest_reverted_coarse_edge_advantage": largest_edge_advantage,
        "largest_reverted_boundary_length_ratio": largest_boundary_length_ratio,
        "largest_reverted_edge_continuity_gap": largest_edge_continuity_gap,
        "largest_reverted_component_elongation": largest_component_elongation,
        "largest_reverted_coarse_edge_continuity": largest_coarse_edge_continuity,
    }
