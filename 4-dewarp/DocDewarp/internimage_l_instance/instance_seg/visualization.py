"""Instance overlays and ROI diagnostic images."""

from typing import Optional, Sequence, Tuple
import cv2
import numpy as np


COLORS = [(230, 80, 80), (80, 210, 120), (80, 140, 240), (235, 190, 65), (180, 90, 225)]


def draw_roi_debug(
    crop_bgr: np.ndarray,
    coarse_crop: np.ndarray,
    candidate: Optional[np.ndarray],
    label: str,
    max_side: int = 1400,
) -> np.ndarray:
    """Draw coarse (blue) and second-pass (green) boundaries for inspection."""
    debug = crop_bgr.copy()
    coarse_contours, _ = cv2.findContours(
        coarse_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(debug, coarse_contours, -1, (255, 100, 40), 3)
    if candidate is not None:
        candidate_contours, _ = cv2.findContours(
            candidate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(debug, candidate_contours, -1, (60, 220, 80), 3)
    cv2.putText(
        debug,
        label,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        debug,
        label,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    height, width = debug.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        debug = cv2.resize(
            debug,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return debug


def render_mask(
    overlay: np.ndarray,
    mask: np.ndarray,
    bbox: Sequence[float],
    score: float,
    instance_id: int,
    color: Tuple[int, int, int],
) -> None:
    color_array = np.asarray(color, dtype=np.float32)
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * color_array).astype(np.uint8)
    x1, y1, x2, y2 = [float(value) for value in bbox]
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, color, 3)
    cv2.putText(
        overlay,
        f"doc {instance_id}: {score:.3f}",
        (round(x1), max(18, round(y1) - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
