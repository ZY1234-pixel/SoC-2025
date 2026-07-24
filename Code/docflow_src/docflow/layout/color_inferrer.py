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
