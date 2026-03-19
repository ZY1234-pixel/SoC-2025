"""Debug visualization helpers for layout + OCR analysis.

This module provides two overlays:
1) Layout + OCR overlay from Paddle-style raw regions.
2) Sorted-layout overlay from DocFlow post-layout blocks.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_TYPE_PALETTE = [
    (80, 127, 255),   # blue-ish
    (46, 204, 113),   # green
    (241, 196, 15),   # yellow
    (231, 76, 60),    # red
    (155, 89, 182),   # purple
    (26, 188, 156),   # cyan
    (230, 126, 34),   # orange
]

_COL_PALETTE = [
    (220, 80, 50),
    (50, 180, 80),
    (50, 130, 220),
    (180, 50, 180),
    (50, 200, 200),
    (200, 130, 50),
]


def _type_color(type_name: str) -> tuple[int, int, int]:
    idx = abs(hash(type_name)) % len(_TYPE_PALETTE)
    return _TYPE_PALETTE[idx]


def _as_int_bbox(bbox: Sequence[float]) -> tuple[int, int, int, int]:
    if not bbox or len(bbox) != 4:
        return 0, 0, 0, 0
    x1, y1, x2, y2 = bbox
    return int(x1), int(y1), int(x2), int(y2)


def _load_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size, encoding="utf-8")
        except Exception:
            pass
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def _extract_text_result(item) -> tuple[list[list[float]] | None, str]:
    """Normalize OCR line item to (poly, text)."""
    if isinstance(item, dict):
        poly = item.get("text_region")
        text = str(item.get("text", ""))
        return poly, text
    if isinstance(item, (list, tuple)) and len(item) == 2:
        poly = item[0]
        value = item[1]
        if isinstance(value, (list, tuple)) and value:
            text = str(value[0])
        else:
            text = str(value)
        return poly, text
    return None, ""


def draw_layout_ocr(
    image_bgr: np.ndarray,
    regions: List[Dict],
    *,
    font_path: str | None = None,
    show_text_preview: bool = True,
    max_preview_chars: int = 24,
) -> np.ndarray:
    """Draw layout bbox + OCR line polygons on top of page image."""
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(font_path, size=14)
    tiny_font = _load_font(font_path, size=12)

    for idx, region in enumerate(regions):
        rtype = str(region.get("type", "?"))
        score = float(region.get("score", 0.0))
        color = _type_color(rtype)
        x1, y1, x2, y2 = _as_int_bbox(region.get("bbox", []))

        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{idx}:{rtype} {score:.2f}"
        tw, th = _text_size(draw, label, font)
        ly1 = max(0, y1 - th - 4)
        draw.rectangle((x1, ly1, x1 + tw + 6, ly1 + th + 4), fill=color)
        draw.text((x1 + 3, ly1 + 1), label, fill=(255, 255, 255), font=font)

        res = region.get("res")
        if not isinstance(res, list):
            continue
        for line in res:
            poly, txt = _extract_text_result(line)
            if not isinstance(poly, (list, tuple)) or len(poly) < 2:
                continue
            pts = [(int(p[0]), int(p[1])) for p in poly if isinstance(p, (list, tuple)) and len(p) >= 2]
            if len(pts) < 2:
                continue
            draw.line(pts + [pts[0]], fill=(255, 255, 0), width=1)
            if show_text_preview and txt:
                preview = txt[:max_preview_chars]
                px, py = pts[0]
                draw.text((px + 1, max(0, py - 12)), preview, fill=(255, 255, 0), font=tiny_font)

    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def draw_sorted_layout(image_bgr: np.ndarray, blocks: List[Dict]) -> np.ndarray:
    """Draw sorted-layout overlay (color by col_index, mark spanning blocks)."""
    vis = image_bgr.copy()
    for order, block in enumerate(blocks):
        bbox = block.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = _as_int_bbox(bbox)
        col_index = int(block.get("col_index", 0))
        col_count = int(block.get("col_count", 1))
        spanned_cols = block.get("spanned_cols") or [col_index]
        if not isinstance(spanned_cols, list):
            spanned_cols = [col_index]

        color = _COL_PALETTE[col_index % len(_COL_PALETTE)]
        thick = 4 if len(spanned_cols) > 1 else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        if len(spanned_cols) > 1:
            cv2.rectangle(vis, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), (255, 255, 255), 1)

        btype = str(block.get("type", "?"))
        label = f"{order}:{btype[:6]} {col_count}c-{col_index}"
        if len(spanned_cols) > 1:
            label += f"[{','.join(str(v) for v in spanned_cols)}]"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ly = max(y1 - 2, th + 4)
        cv2.rectangle(vis, (x1, ly - th - 3), (x1 + tw + 4, ly + 2), color, -1)
        cv2.putText(
            vis,
            label,
            (x1 + 2, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def extract_sorted_blocks(document, page_index: int = 0) -> List[Dict]:
    """Flatten DocFlow page.zones blocks to list[dict] for visualization."""
    if not document.pages or page_index < 0 or page_index >= len(document.pages):
        return []

    page = document.pages[page_index]
    out: List[Dict] = []
    for zone in page.zones:
        for blk in zone.blocks:
            out.append(
                {
                    "type": blk.block_type.value,
                    "bbox": [blk.bbox.x1, blk.bbox.y1, blk.bbox.x2, blk.bbox.y2],
                    "col_count": int(blk.col_count),
                    "col_index": int(blk.col_index),
                    "spanned_cols": list(blk.spanned_cols) if blk.spanned_cols else [int(blk.col_index)],
                }
            )
    return out
