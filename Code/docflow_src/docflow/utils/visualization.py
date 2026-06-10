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
    (231, 76, 60),    # red
    (80, 127, 255),   # blue
    (46, 204, 113),   # green
    (241, 196, 15),   # yellow
    (155, 89, 182),   # purple
    (26, 188, 156),   # cyan
    (230, 126, 34),   # orange
    (192, 57, 43),    # dark red
    (39, 174, 96),    # dark green
    (211, 84, 0),     # dark orange
    (142, 68, 173),   # dark purple
    (41, 128, 185),   # dark blue
]


def _type_color(type_name: str) -> tuple[int, int, int]:
    """DJB2 字符串哈希 + 12 色调色板，相同类型颜色稳定，碰撞率低。"""
    h = 5381
    for ch in type_name:
        h = ((h * 33) + ord(ch)) & 0xFFFFFFFF
    return _TYPE_PALETTE[h % len(_TYPE_PALETTE)]


# 列索引调色板（BGR 格式，适配 cv2.rectangle）
_COL_PALETTE = [
    (50, 80, 220),    # blue
    (80, 180, 50),    # green
    (220, 130, 50),   # orange
    (180, 50, 180),   # purple
    (200, 200, 50),   # yellow
    (50, 130, 200),   # brown
]

_FLOW_PALETTE_BGR = [
    (40, 85, 235),    # vivid blue
    (38, 178, 68),    # vivid green
    (44, 158, 245),   # vivid orange
    (192, 68, 214),   # vivid violet
    (30, 198, 198),   # vivid cyan
    (68, 74, 238),    # vivid red
    (48, 206, 240),   # bright amber
    (132, 112, 36),   # olive/brown
]


def _as_int_bbox(bbox: Sequence[float]) -> tuple[int, int, int, int]:
    if not bbox or len(bbox) != 4:
        return 0, 0, 0, 0
    x1, y1, x2, y2 = bbox
    return int(x1), int(y1), int(x2), int(y2)


def _flow_color(flow_id: str, fallback_col_index: int = 0) -> tuple[int, int, int]:
    if not flow_id:
        return _COL_PALETTE[fallback_col_index % len(_COL_PALETTE)]
    h = 5381
    for ch in flow_id:
        h = ((h * 33) + ord(ch)) & 0xFFFFFFFF
    return _FLOW_PALETTE_BGR[h % len(_FLOW_PALETTE_BGR)]


def _tint_for_column(color: tuple[int, int, int], col_index: int) -> tuple[int, int, int]:
    if col_index <= 0:
        return color
    # 同一 flow 内不同列采用更强的明暗分层，优先保证肉眼易区分。
    factor = max(0.34, 1.0 - 0.28 * col_index)
    lift = 255 * (1.0 - factor) * 0.22
    return tuple(max(0, min(255, int(channel * factor + lift))) for channel in color)


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


def _overlay_ocr_polys(
    image_bgr: np.ndarray,
    regions: List[Dict] | None,
    *,
    color: tuple[int, int, int] = (0, 255, 255),
    show_text_preview: bool = False,
    font_path: str | None = None,
    max_preview_chars: int = 18,
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    if not regions:
        return image_bgr

    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    tiny_font = _load_font(font_path, size=12)
    dx, dy = offset

    for region in regions:
        res = region.get("res")
        if not isinstance(res, list):
            continue
        for line in res:
            poly, txt = _extract_text_result(line)
            if not isinstance(poly, (list, tuple)) or len(poly) < 2:
                continue
            pts = [
                (int(p[0]) + dx, int(p[1]) + dy)
                for p in poly
                if isinstance(p, (list, tuple)) and len(p) >= 2
            ]
            if len(pts) < 2:
                continue
            draw.line(pts + [pts[0]], fill=color, width=2)
            if show_text_preview and txt:
                preview = txt[:max_preview_chars]
                px, py = pts[0]
                draw.text((px + 1, max(0, py - 12)), preview, fill=color, font=tiny_font)

    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


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

        # 区块边界框：加粗描边
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

        # 区域标签
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


def draw_sorted_layout(
    image_bgr: np.ndarray,
    blocks: List[Dict],
    *,
    ocr_regions: List[Dict] | None = None,
) -> np.ndarray:
    """Draw sorted-layout overlay (color by col_index, mark spanning blocks)."""
    vis, dx, dy = _make_annotation_canvas(image_bgr, top=32, bottom=8, left=8, right=8)
    for order, block in enumerate(blocks):
        bbox = block.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = _as_int_bbox(bbox)
        x1 += dx
        x2 += dx
        y1 += dy
        y2 += dy
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
        flow_id = str(block.get("flow_id", "") or "")
        flow_tag = ""
        if flow_id:
            flow_tag = f" f{flow_id.split('_')[-1]}"
        label = f"{order}:{btype[:6]} {col_count}c-{col_index}{flow_tag}"
        if len(spanned_cols) > 1:
            label += f"[{','.join(str(v) for v in spanned_cols)}]"
        _draw_label_panel(vis, label, anchor_bbox=(x1, y1, x2, y2), color=color, font_scale=0.42)
    return _overlay_ocr_polys(vis, ocr_regions, show_text_preview=False, offset=(dx, dy))


def _fit_center_font_scale(text: str, box_w: int, box_h: int) -> tuple[float, int]:
    """为块中心大号序号估计更激进的自适应字号。"""
    if box_w <= 0 or box_h <= 0:
        return 0.45, 2

    # 让数字尽量填满区块，同时为细长块保留少量边距。
    target_w = max(20, int(box_w * 0.72))
    target_h = max(20, int(box_h * 0.68))

    # 以区块较短边估算一个初始尺度，再向下试探，避免对大块上限过低。
    short_edge = max(1.0, min(float(box_w), float(box_h)))
    scale = max(0.55, min(8.0, short_edge / 46.0))
    thickness = max(2, min(10, int(scale * 1.6)))

    while scale > 0.30:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw <= target_w and th <= target_h:
            return scale, thickness
        scale *= 0.92
        thickness = max(2, min(10, int(scale * 1.6)))

    return 0.30, 2


def _block_area(block: Dict) -> float:
    x1, y1, x2, y2 = _as_int_bbox(block.get("bbox", []))
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _visual_stack_order(blocks: List[Dict]) -> List[tuple[int, Dict]]:
    """Return blocks in a visibility-friendly z-order.

    Large / spanning blocks are rendered first and smaller local blocks later,
    so cross-column titles do not hide inner text boxes.
    """
    indexed = list(enumerate(blocks))
    return sorted(
        indexed,
        key=lambda item: (
            -len(item[1].get("spanned_cols") or []),
            -_block_area(item[1]),
            item[0],
        ),
    )


def _inset_rect(x1: int, y1: int, x2: int, y2: int, inset: int) -> tuple[int, int, int, int]:
    if inset <= 0:
        return x1, y1, x2, y2
    if (x2 - x1) <= inset * 2 or (y2 - y1) <= inset * 2:
        return x1, y1, x2, y2
    return x1 + inset, y1 + inset, x2 - inset, y2 - inset


def _make_annotation_canvas(
    image_bgr: np.ndarray,
    *,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
    fill_value: int = 255,
) -> tuple[np.ndarray, int, int]:
    h, w = image_bgr.shape[:2]
    canvas = np.full((h + top + bottom, w + left + right, 3), fill_value, dtype=image_bgr.dtype)
    canvas[top:top + h, left:left + w] = image_bgr
    return canvas, left, top


def _place_panel(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    panel_w: int,
    panel_h: int,
    canvas_w: int,
    canvas_h: int,
    gap: int = 4,
) -> tuple[int, int]:
    candidates = [
        (x1, y1 - panel_h - gap),
        (x1, y2 + gap),
        (x2 + gap, y1),
        (x1 - panel_w - gap, y1),
        (x1 + 4, y1 + 4),
    ]
    for px, py in candidates:
        if px < 0 or py < 0:
            continue
        if px + panel_w > canvas_w or py + panel_h > canvas_h:
            continue
        return px, py

    clamped_x = min(max(0, x1 + 4), max(0, canvas_w - panel_w))
    clamped_y = min(max(0, y1 + 4), max(0, canvas_h - panel_h))
    return clamped_x, clamped_y


def _draw_label_panel(
    image_bgr: np.ndarray,
    text: str,
    *,
    anchor_bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.42,
    text_thickness: int = 1,
    gap: int = 4,
) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    panel_w = tw + 8
    panel_h = th + baseline + 8
    px, py = _place_panel(
        *anchor_bbox,
        panel_w=panel_w,
        panel_h=panel_h,
        canvas_w=image_bgr.shape[1],
        canvas_h=image_bgr.shape[0],
        gap=gap,
    )
    cv2.rectangle(image_bgr, (px, py), (px + panel_w, py + panel_h), color, -1)
    cv2.putText(
        image_bgr,
        text,
        (px + 4, py + th + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def _draw_order_badge(
    image_bgr: np.ndarray,
    number: str,
    *,
    anchor_bbox: tuple[int, int, int, int],
    block_type: str,
) -> None:
    x1, y1, x2, y2 = anchor_bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    should_externalize = block_type == "title" or box_h <= 72 or box_w <= 96

    scale, num_thickness = _fit_center_font_scale(number, box_w, box_h)
    if should_externalize:
        scale = min(scale, 1.15)
        num_thickness = max(2, min(4, num_thickness))
        (tw, th), baseline = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, scale, num_thickness)
        panel_w = tw + 18
        panel_h = th + baseline + 16
        px, py = _place_panel(
            x1,
            y1,
            x2,
            y2,
            panel_w=panel_w,
            panel_h=panel_h,
            canvas_w=image_bgr.shape[1],
            canvas_h=image_bgr.shape[0],
            gap=6,
        )
        cv2.rectangle(image_bgr, (px, py), (px + panel_w, py + panel_h), (255, 255, 255), -1)
        cv2.rectangle(image_bgr, (px, py), (px + panel_w, py + panel_h), (60, 60, 60), 2)
        tx = px + max(6, (panel_w - tw) // 2)
        ty = py + th + max(4, (panel_h - (th + baseline)) // 2)
        cv2.putText(
            image_bgr,
            number,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (25, 25, 25),
            num_thickness,
            cv2.LINE_AA,
        )
        return

    (tw, th), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, scale, num_thickness)
    cx = x1 + max(0, (box_w - tw) // 2)
    cy = y1 + max(th, (box_h + th) // 2)
    cv2.putText(
        image_bgr,
        number,
        (cx, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        num_thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        number,
        (cx, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (25, 25, 25),
        num_thickness,
        cv2.LINE_AA,
    )


def draw_reading_order_map(
    image_bgr: np.ndarray,
    blocks: List[Dict],
    *,
    title: str | None = None,
    alpha: float = 0.34,
    ocr_regions: List[Dict] | None = None,
) -> np.ndarray:
    """绘制论文风格的阅读顺序图：块填充 + 中央大号序号。"""
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    title_band_h = 56 if title else 0
    top_pad = title_band_h + 12
    side_pad = 12
    bottom_pad = 12
    base, dx, dy = _make_annotation_canvas(
        image_bgr,
        top=top_pad,
        bottom=bottom_pad,
        left=side_pad,
        right=side_pad,
    )
    overlay = base.copy()

    if title:
        cv2.rectangle(base, (0, 0), (base.shape[1], title_band_h + 6), (248, 248, 248), -1)
        cv2.putText(
            base,
            title,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        overlay = base.copy()

    stacked_blocks = _visual_stack_order(blocks)

    for _order_idx, block in stacked_blocks:
        x1, y1, x2, y2 = _as_int_bbox(block.get("bbox", []))
        x1 += dx
        x2 += dx
        y1 += dy
        y2 += dy
        col_index = int(block.get("col_index", 0))
        flow_id = str(block.get("flow_id", "") or "")
        spanned_cols = block.get("spanned_cols") or [col_index]
        if not isinstance(spanned_cols, list):
            spanned_cols = [col_index]

        color = _tint_for_column(_flow_color(flow_id, col_index), col_index)
        fill_inset = 3 if len(spanned_cols) > 1 else 2
        fx1, fy1, fx2, fy2 = _inset_rect(x1, y1, x2, y2, fill_inset)
        cv2.rectangle(overlay, (fx1, fy1), (fx2, fy2), color, -1)
        if len(spanned_cols) > 1:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 6)

    vis = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)

    for _order_idx, block in stacked_blocks:
        x1, y1, x2, y2 = _as_int_bbox(block.get("bbox", []))
        x1 += dx
        x2 += dx
        y1 += dy
        y2 += dy
        col_index = int(block.get("col_index", 0))
        flow_id = str(block.get("flow_id", "") or "")
        spanned_cols = block.get("spanned_cols") or [col_index]
        if not isinstance(spanned_cols, list):
            spanned_cols = [col_index]

        color = _tint_for_column(_flow_color(flow_id, col_index), col_index)
        thick = 5 if len(spanned_cols) > 1 else 3
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), thick + 2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)

    for order, block in enumerate(blocks, start=1):
        x1, y1, x2, y2 = _as_int_bbox(block.get("bbox", []))
        x1 += dx
        x2 += dx
        y1 += dy
        y2 += dy
        col_index = int(block.get("col_index", 0))
        flow_id = str(block.get("flow_id", "") or "")
        spanned_cols = block.get("spanned_cols") or [col_index]
        if not isinstance(spanned_cols, list):
            spanned_cols = [col_index]

        _draw_order_badge(
            vis,
            str(order),
            anchor_bbox=(x1, y1, x2, y2),
            block_type=str(block.get("type", "?")),
        )

        btype = str(block.get("type", "?"))
        flow_tag = f" f{flow_id.split('_')[-1]}" if flow_id else ""
        small = f"{btype} c{col_index}{flow_tag}"
        _draw_label_panel(
            vis,
            small,
            anchor_bbox=(x1, y1, x2, y2),
            color=(72, 72, 72),
            font_scale=0.42,
            text_thickness=1,
        )

    return _overlay_ocr_polys(vis, ocr_regions, show_text_preview=False, offset=(dx, dy))


def draw_reading_order_comparison(
    image_bgr: np.ndarray,
    legacy_blocks: List[Dict],
    xycutpp_blocks: List[Dict],
    *,
    ocr_regions: List[Dict] | None = None,
) -> np.ndarray:
    """并排比较原始/XY-Cut++ 两种读序。"""
    left = draw_reading_order_map(
        image_bgr,
        legacy_blocks,
        title="Legacy Reading Order",
        ocr_regions=ocr_regions,
    )
    right = draw_reading_order_map(
        image_bgr,
        xycutpp_blocks,
        title="XY-Cut++ Reading Order",
        ocr_regions=ocr_regions,
    )
    gap = np.full((left.shape[0], 24, 3), 245, dtype=np.uint8)
    return np.concatenate([left, gap, right], axis=1)


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
                    "id": getattr(blk, "block_id", ""),
                    "type": blk.block_type.value,
                    "bbox": [blk.bbox.x1, blk.bbox.y1, blk.bbox.x2, blk.bbox.y2],
                    "col_count": int(blk.col_count),
                    "col_index": int(blk.col_index),
                    "spanned_cols": list(blk.spanned_cols) if blk.spanned_cols else [int(blk.col_index)],
                    "flow_id": str((getattr(blk, "attributes", None) or {}).get("flow_id", "")),
                }
            )
    return out
