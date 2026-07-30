"""Infer editable text and table colors from recognition crops."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class CropStylePrediction:
    text_color: str
    background_color: Optional[str]
    confidence: float


def infer_crop_style(image_base64: Optional[str]) -> Optional[CropStylePrediction]:
    image = _decode_image(image_base64)
    return _infer_pixels_style(image) if image is not None else None


def infer_background_extent(
    image_rgb: np.ndarray,
    bbox: tuple[float, float, float, float],
    color: str,
) -> Optional[tuple[float, float, float, float]]:
    """Return the connected page-color region overlapping a recognized text box."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or len(color) != 7:
        return None
    try:
        target = np.array([int(color[index : index + 2], 16) for index in (1, 3, 5)], dtype=np.int16)
    except ValueError:
        return None
    border = np.concatenate((image_rgb[0], image_rgb[-1], image_rgb[:, 0], image_rgb[:, -1]))
    page_color = np.median(border, axis=0)
    tolerance = min(48.0, max(12.0, np.linalg.norm(target - page_color) * 0.45))
    mask = (np.linalg.norm(image_rgb.astype(np.int16) - target, axis=2) <= tolerance).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    height, width = mask.shape
    x1, y1, x2, y2 = (
        max(0, min(width, round(bbox[0]))),
        max(0, min(height, round(bbox[1]))),
        max(0, min(width, round(bbox[2]))),
        max(0, min(height, round(bbox[3]))),
    )
    if x2 <= x1 or y2 <= y1:
        return None
    overlaps = np.bincount(labels[y1:y2, x1:x2].ravel(), minlength=count)
    overlaps[0] = 0
    label = int(overlaps.argmax())
    if overlaps[label] < max((x2 - x1) * (y2 - y1) * 0.08, 16):
        return None
    left, top, region_width, region_height, _area = stats[label]
    box_height = y2 - y1
    if region_height > max(box_height * 3, box_height + 12):
        top, region_height = y1, box_height
    return float(left), float(top), float(left + region_width), float(top + region_height)


def infer_table_row_fills(image_base64: Optional[str], row_count: int) -> tuple[tuple[int, str, str], ...]:
    image = _decode_image(image_base64)
    if image is None or row_count <= 0:
        return ()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturated = (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 40)
    colored_lines = np.flatnonzero(saturated.mean(axis=1) >= 0.45)
    if not len(colored_lines):
        return ()
    runs = np.split(colored_lines, np.where(np.diff(colored_lines) > 1)[0] + 1)
    fills = {}
    for run in runs:
        if len(run) < 2:
            continue
        top, bottom = int(run[0]), int(run[-1]) + 1
        band = image[top:bottom]
        band_saturated = saturated[top:bottom]
        style = _infer_pixels_style(band)
        if style is None:
            continue
        row = min(int(((top + bottom) / 2) / image.shape[0] * row_count), row_count - 1)
        fills[row] = (_hex_bgr(np.median(band[band_saturated], axis=0)), style.text_color)
    return tuple((row, *colors) for row, colors in sorted(fills.items()))


def infer_table_rule_style(image_base64: Optional[str]) -> Optional[str]:
    image = _decode_image(image_base64)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(image.shape[1] // 8, 2), 1)),
    )
    vertical = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(image.shape[0] // 4, 2))),
    )
    horizontal_rules = int(np.count_nonzero((horizontal > 0).mean(axis=1) >= 0.40))
    vertical_rules = int(np.count_nonzero((vertical > 0).mean(axis=0) >= 0.40))
    if vertical_rules:
        return "grid"
    return "horizontal" if horizontal_rules >= 2 else "borderless"


def _infer_pixels_style(image: np.ndarray) -> Optional[CropStylePrediction]:
    pixels = image.reshape(-1, 3)
    bins = (pixels.astype(np.uint16) // 16).astype(np.uint16)
    codes = bins[:, 0] * 256 + bins[:, 1] * 16 + bins[:, 2]
    background_code = int(np.bincount(codes, minlength=4096).argmax())
    background_mask = codes == background_code
    background = np.median(pixels[background_mask], axis=0)
    distances = np.linalg.norm(pixels.astype(np.float32) - background, axis=1)
    foreground = pixels[distances >= 40.0]
    if len(foreground) < 16:
        return None
    foreground_distances = np.linalg.norm(foreground.astype(np.float32) - background, axis=1)
    core = foreground[foreground_distances >= np.percentile(foreground_distances, 80)]
    text_color = _hex_bgr(np.median(core, axis=0))
    background_color = None
    if float(background_mask.mean()) >= 0.25 and np.linalg.norm(background - 255.0) >= 32.0:
        background_color = _hex_bgr(background)
    confidence = min(1.0, float(background_mask.mean()) + len(foreground) / max(len(pixels), 1))
    return CropStylePrediction(text_color, background_color, confidence)


def _decode_image(value: Optional[str]) -> Optional[np.ndarray]:
    if not value:
        return None
    try:
        payload = value.split(",", 1)[1] if "," in value[:80] else value
        data = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (ValueError, TypeError):
        return None


def _hex_bgr(color: np.ndarray) -> str:
    blue, green, red = (int(np.clip(round(float(value)), 0, 255)) for value in color)
    return f"#{red:02X}{green:02X}{blue:02X}"
