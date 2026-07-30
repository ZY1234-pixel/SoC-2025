"""Infer scale-independent stroke weight from text crops."""

from __future__ import annotations

from statistics import median
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image


def infer_text_stroke_ratio(crops: Iterable[Image.Image]) -> Optional[float]:
    ratios = [ratio for crop in crops if (ratio := _stroke_ratio(crop)) is not None]
    return median(ratios) if ratios else None


def _stroke_ratio(image: Image.Image) -> Optional[float]:
    pixels = np.asarray(image.convert("RGB"))
    if pixels.ndim != 3 or min(pixels.shape[:2]) < 5:
        return None
    flat = pixels.reshape(-1, 3)
    bins = (flat.astype(np.uint16) // 16).astype(np.uint16)
    codes = bins[:, 0] * 256 + bins[:, 1] * 16 + bins[:, 2]
    background_code = int(np.bincount(codes, minlength=4096).argmax())
    background = np.median(flat[codes == background_code], axis=0)
    foreground = np.linalg.norm(pixels.astype(np.float32) - background, axis=2) >= 40.0
    if foreground.mean() < 0.003 or foreground.mean() > 0.85:
        return None

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    cleaned = np.zeros(foreground.shape, dtype=np.uint8)
    minimum_area = max(2, round(foreground.size * 0.00001))
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == label] = 1
    active_rows = np.flatnonzero(cleaned.any(axis=1))
    if len(active_rows) < 5 or np.count_nonzero(cleaned) < 16:
        return None

    ink_height = int(active_rows[-1] - active_rows[0] + 1)
    distances = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)[cleaned > 0]
    return float(np.percentile(distances, 60) * 2.0 / max(ink_height, 1))
