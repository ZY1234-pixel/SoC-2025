"""Mask geometry, crop boundaries and candidate matching."""

from typing import Dict, Optional, Sequence, Tuple
import cv2
import math
import numpy as np


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return an exclusive XYXY box tightly enclosing a boolean mask."""
    x, y, width, height = cv2.boundingRect(
        np.ascontiguousarray(mask.astype(np.uint8, copy=False))
    )
    if width <= 0 or height <= 0:
        return None
    return x, y, x + width, y + height


def expanded_crop_box(
    bbox: Sequence[float],
    image_shape: Tuple[int, int],
    margin_ratio: float,
    minimum_margin: int = 16,
) -> Tuple[int, int, int, int]:
    """Expand an XYXY box, clip it to the image, and keep exclusive maxima."""
    height, width = image_shape
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    margin_x = max(float(minimum_margin), box_width * float(margin_ratio))
    margin_y = max(float(minimum_margin), box_height * float(margin_ratio))
    crop_x1 = max(0, int(math.floor(x1 - margin_x)))
    crop_y1 = max(0, int(math.floor(y1 - margin_y)))
    crop_x2 = min(width, int(math.ceil(x2 + margin_x)))
    crop_y2 = min(height, int(math.ceil(y2 + margin_y)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError(
            f"Invalid expanded crop {(crop_x1, crop_y1, crop_x2, crop_y2)} "
            f"from bbox {tuple(bbox)} in image {(width, height)}"
        )
    return crop_x1, crop_y1, crop_x2, crop_y2


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_bool = first.astype(bool, copy=False)
    second_bool = second.astype(bool, copy=False)
    union = np.logical_or(first_bool, second_bool).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(first_bool, second_bool).sum() / union)


def candidate_metrics(candidate: np.ndarray, coarse: np.ndarray) -> Dict[str, float]:
    candidate_bool = candidate.astype(bool, copy=False)
    coarse_bool = coarse.astype(bool, copy=False)
    candidate_area = int(candidate_bool.sum())
    coarse_area = int(coarse_bool.sum())
    intersection = int(np.logical_and(candidate_bool, coarse_bool).sum())
    union = candidate_area + coarse_area - intersection
    return {
        "iou": float(intersection / union) if union else 0.0,
        "overlap_min": (
            float(intersection / min(candidate_area, coarse_area))
            if candidate_area and coarse_area
            else 0.0
        ),
        "area_ratio": float(candidate_area / coarse_area) if coarse_area else 0.0,
    }


def boundary_change_metrics(
    candidate: np.ndarray,
    coarse: np.ndarray,
    analysis_scale: float,
) -> Dict[str, float]:
    """Measure local mask displacement in first-pass input pixels.

    Global IoU can hide a deep local notch on a multi-megapixel document.  Both
    masks are therefore resampled at the first-pass image scale and distances
    are measured only where the second pass changes the first-pass mask.
    """
    if candidate.shape != coarse.shape:
        raise ValueError(
            f"Candidate/coarse shapes differ: {candidate.shape} vs {coarse.shape}"
        )
    height, width = coarse.shape
    resized_width = max(1, int(round(width * float(analysis_scale))))
    resized_height = max(1, int(round(height * float(analysis_scale))))
    coarse_small = (
        cv2.resize(
            coarse.astype(np.uint8),
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )
    candidate_small = (
        cv2.resize(
            candidate.astype(np.uint8),
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )
    added = np.logical_and(candidate_small, ~coarse_small)
    removed = np.logical_and(coarse_small, ~candidate_small)
    changed = np.logical_or(added, removed)
    coarse_area = int(coarse_small.sum())
    if not bool(changed.any()):
        return {
            "max_boundary_shift_input_px": 0.0,
            "p95_boundary_shift_input_px": 0.0,
            "changed_pixel_ratio": 0.0,
        }

    outside_distance = cv2.distanceTransform(
        (~coarse_small).astype(np.uint8), cv2.DIST_L2, 5
    )
    inside_distance = cv2.distanceTransform(
        coarse_small.astype(np.uint8), cv2.DIST_L2, 5
    )
    distances = np.concatenate((outside_distance[added], inside_distance[removed]))
    return {
        "max_boundary_shift_input_px": float(distances.max()),
        "p95_boundary_shift_input_px": float(np.percentile(distances, 95)),
        "changed_pixel_ratio": (
            float(changed.sum() / coarse_area) if coarse_area else 0.0
        ),
    }


def local_edge_support(
    image_bgr: np.ndarray, box: Tuple[int, int, int, int]
) -> np.ndarray:
    """Calculate LAB edge evidence only around one changed mask component."""
    x1, y1, x2, y2 = box
    image_height, image_width = image_bgr.shape[:2]
    # Sobel needs one neighbouring pixel and the 5x5 dilation needs two more.
    context = 3
    source_x1 = max(0, x1 - context)
    source_y1 = max(0, y1 - context)
    source_x2 = min(image_width, x2 + context)
    source_y2 = min(image_height, y2 + context)
    image_lab = cv2.cvtColor(
        image_bgr[source_y1:source_y2, source_x1:source_x2], cv2.COLOR_BGR2LAB
    )
    support = np.zeros(image_lab.shape[:2], dtype=np.float32)
    for channel in cv2.split(image_lab):
        gradient_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        support = np.maximum(support, cv2.magnitude(gradient_x, gradient_y))
    support = cv2.dilate(support, np.ones((5, 5), dtype=np.uint8))
    return support[
        y1 - source_y1 : y2 - source_y1,
        x1 - source_x1 : x2 - source_x1,
    ]


def select_roi_candidate(
    candidate_masks: Sequence[np.ndarray],
    candidate_scores: Sequence[float],
    coarse_crop: np.ndarray,
) -> Tuple[Optional[int], Dict[str, float]]:
    """Match a second-pass instance to the projected first-pass mask."""
    best_index: Optional[int] = None
    best_metrics: Dict[str, float] = {}
    best_match_score = -1.0
    for index, (candidate, confidence) in enumerate(
        zip(candidate_masks, candidate_scores)
    ):
        metrics = candidate_metrics(candidate, coarse_crop)
        match_score = (
            0.70 * metrics["iou"]
            + 0.20 * metrics["overlap_min"]
            + 0.10 * float(confidence)
        )
        if match_score > best_match_score:
            best_index = index
            best_match_score = match_score
            best_metrics = {
                **metrics,
                "candidate_score": float(confidence),
                "match_score": float(match_score),
            }
    return best_index, best_metrics


def touches_internal_crop_edge(
    mask: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    border: int = 2,
) -> bool:
    """Detect a likely-too-tight ROI while allowing true source-image borders."""
    if not bool(mask.any()):
        return False
    x1, y1, x2, y2 = crop_box
    image_height, image_width = image_shape
    border = max(1, int(border))
    if y1 > 0 and bool(mask[:border, :].any()):
        return True
    if y2 < image_height and bool(mask[-border:, :].any()):
        return True
    if x1 > 0 and bool(mask[:, :border].any()):
        return True
    if x2 < image_width and bool(mask[:, -border:].any()):
        return True
    return False
