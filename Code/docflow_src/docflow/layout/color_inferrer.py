"""Text foreground color inference from page pixels."""

from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np

from docflow.model.base import BlockType
from docflow.model.blocks.text_block import TextBlock, TextLine
from docflow.model.page import Page
from docflow.schema.models import BlockStyle, TextLineStyle


@dataclass(frozen=True)
class ColorPrediction:
    color: str
    confidence: float
    foreground_pixels: int


@dataclass(frozen=True)
class CropStylePrediction:
    text_color: str
    background_color: Optional[str]
    confidence: float


_TEXT_TYPES = {
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.REFERENCE,
    BlockType.ABSTRACT,
    BlockType.FOOTNOTE,
    BlockType.FIGURE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.FORMULA_CAPTION,
}


def infer_text_colors(
    page: Page,
    blocks: list,
    *,
    min_foreground_pixels: int = 24,
    min_confidence: float = 0.58,
) -> dict:
    """Infer foreground text colors and write them into line/block styles.

    Black text is treated as the default and is not explicitly written.
    Non-default colors are promoted from line styles to block style when most
    text lines agree.
    """
    image = _load_page_image(page)
    if image is None:
        return {"enabled": True, "available": False, "reason": "page_image_missing", "lines": 0, "blocks": 0}

    line_count = 0
    colored_lines = 0
    colored_blocks = 0

    for block in blocks:
        if not isinstance(block, TextBlock) or block.block_type not in _TEXT_TYPES:
            continue
        block_colors: list[str] = []
        for line in block.lines or []:
            line_count += 1
            pred = infer_line_color(image, line, min_foreground_pixels=min_foreground_pixels)
            if pred is None or pred.confidence < min_confidence:
                continue
            if pred.color == "#000000":
                continue
            if line.style is None:
                line.style = TextLineStyle()
            if line.style.color is None:
                line.style.color = pred.color
            block_colors.append(pred.color)
            colored_lines += 1

        if block_colors:
            color, count = Counter(block_colors).most_common(1)[0]
            if count / max(len(block.lines or []), 1) >= 0.60:
                if block.style is None:
                    block.style = BlockStyle()
                if block.style.color is None:
                    block.style.color = color
                    colored_blocks += 1
                for line in block.lines or []:
                    if line.style is not None and line.style.color == color:
                        line.style.color = None

    return {
        "enabled": True,
        "available": True,
        "lines": line_count,
        "colored_lines": colored_lines,
        "colored_blocks": colored_blocks,
    }


def infer_line_color(
    image: np.ndarray,
    line: TextLine,
    *,
    min_foreground_pixels: int = 24,
) -> Optional[ColorPrediction]:
    if line.text_region is None:
        return None
    mask = _polygon_mask(image.shape[:2], line.text_region)
    if mask is None:
        return None
    pixels = image[mask > 0]
    if pixels.size == 0:
        return None
    foreground = _foreground_pixels(pixels)
    if len(foreground) < min_foreground_pixels:
        return None
    return _classify_foreground_color(foreground)


def infer_crop_style(image_base64: Optional[str]) -> Optional[CropStylePrediction]:
    image = _decode_image(image_base64)
    return _infer_pixels_style(image) if image is not None else None


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


def _load_page_image(page: Page) -> Optional[np.ndarray]:
    if page.image_path:
        image = cv2.imread(page.image_path)
        if image is not None:
            return image
    return None


def _polygon_mask(shape: tuple[int, int], polygon: Iterable[Iterable[float]]) -> Optional[np.ndarray]:
    points = []
    for point in polygon:
        try:
            x, y = point
        except Exception:
            continue
        points.append([int(round(float(x))), int(round(float(y)))])
    if len(points) < 3:
        return None
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
    return mask


def _foreground_pixels(pixels_bgr: np.ndarray) -> np.ndarray:
    pixels = pixels_bgr.reshape(-1, 3).astype(np.uint8)
    gray = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).reshape(-1)
    # Text polygons often contain a lot of white paper and anti-aliased edge
    # pixels. Use the darker quantile as ink candidates instead of averaging the
    # whole polygon.
    cutoff = int(np.percentile(gray, 28))
    fg = pixels[gray <= min(235, max(25, cutoff + 18))]
    if len(fg) < 16:
        fg = pixels[gray <= int(np.percentile(gray, 40))]
    return fg


def _classify_foreground_color(foreground_bgr: np.ndarray) -> ColorPrediction:
    hsv = cv2.cvtColor(foreground_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    bgr = foreground_bgr.astype(np.float32)
    gray = cv2.cvtColor(foreground_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).reshape(-1)

    h = hsv[:, 0].astype(np.float32)
    s = hsv[:, 1].astype(np.float32)
    v = hsv[:, 2].astype(np.float32)

    red_mask = ((h <= 15) | (h >= 160)) & (s >= 24) & (v >= 25)
    blue_mask = (h >= 88) & (h <= 132) & (s >= 30) & (v >= 30)
    green_mask = (h >= 35) & (h <= 85) & (s >= 30) & (v >= 30)
    dark_mask = (gray <= 95) & (s <= 80)

    candidates = {
        "#C00000": float(red_mask.mean()),
        "#1F4E79": float(blue_mask.mean()),
        "#008000": float(green_mask.mean()),
        "#000000": float(dark_mask.mean()),
    }
    color, confidence = max(candidates.items(), key=lambda item: item[1])

    if color == "#000000":
        return ColorPrediction(color=color, confidence=confidence, foreground_pixels=len(foreground_bgr))

    # Keep the measured hue, but estimate the representative ink color from the
    # denser/darker part of the selected foreground pixels. A plain median is too
    # easily pulled toward pale anti-aliased edge pixels on scanned documents.
    selected = {
        "#C00000": red_mask,
        "#1F4E79": blue_mask,
        "#008000": green_mask,
    }[color]
    if selected.any():
        rgb = _representative_ink_rgb(bgr[selected])
        hex_color = "#{:02X}{:02X}{:02X}".format(
            int(np.clip(round(rgb[0]), 0, 255)),
            int(np.clip(round(rgb[1]), 0, 255)),
            int(np.clip(round(rgb[2]), 0, 255)),
        )
        return ColorPrediction(color=hex_color, confidence=confidence, foreground_pixels=len(foreground_bgr))
    return ColorPrediction(color=color, confidence=confidence, foreground_pixels=len(foreground_bgr))


def _representative_ink_rgb(selected_bgr: np.ndarray) -> np.ndarray:
    selected = selected_bgr.reshape(-1, 3).astype(np.float32)
    if len(selected) == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if len(selected) < 8:
        return np.median(selected, axis=0)[::-1]

    hsv = cv2.cvtColor(selected.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    gray = cv2.cvtColor(selected.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).reshape(-1)
    saturation = hsv[:, 1].astype(np.float32)

    sat_cut = float(np.percentile(saturation, 55))
    gray_cut = float(np.percentile(gray, 45))
    core = (saturation >= sat_cut) & (gray <= gray_cut)
    if int(core.sum()) < max(8, len(selected) // 20):
        ink_score = saturation - gray.astype(np.float32) * 0.35
        core = ink_score >= float(np.percentile(ink_score, 75))

    core_bgr = selected[core] if core.any() else selected
    rgb = np.median(core_bgr, axis=0)[::-1]
    return rgb
