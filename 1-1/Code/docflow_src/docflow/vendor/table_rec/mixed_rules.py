from __future__ import annotations

import base64
import html
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def h(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def pad(self, dx: float, dy: float, width: int, height: int) -> "Rect":
        return Rect(
            max(0.0, self.x0 - dx),
            max(0.0, self.y0 - dy),
            min(float(width), self.x1 + dx),
            min(float(height), self.y1 + dy),
        )

    def intersect(self, other: "Rect") -> "Rect":
        return Rect(
            max(self.x0, other.x0),
            max(self.y0, other.y0),
            min(self.x1, other.x1),
            min(self.y1, other.y1),
        )

    def overlap_area(self, other: "Rect") -> float:
        inter = self.intersect(other)
        return max(0.0, inter.w) * max(0.0, inter.h)

    def contains_point(self, x: float, y: float, tol: float = 0.0) -> bool:
        return self.x0 - tol <= x <= self.x1 + tol and self.y0 - tol <= y <= self.y1 + tol

    def to_int(self) -> list[int]:
        return [int(round(v)) for v in (self.x0, self.y0, self.x1, self.y1)]

    def to_list(self) -> list[float]:
        return [float(self.x0), float(self.y0), float(self.x1), float(self.y1)]


@dataclass
class OcrToken:
    text: str
    bbox: Rect
    score: float = 1.0
    source: str = "external"


@dataclass
class VisualAsset:
    kind: str
    bbox: Rect
    score: float
    color: str | None = None


@dataclass
class Cell:
    row: int
    col: int
    bbox: Rect
    text: str = ""
    tokens: list[OcrToken] = field(default_factory=list)
    assets: list[VisualAsset] = field(default_factory=list)
    rowspan: int = 1
    colspan: int = 1
    role: str = "body"


@dataclass
class TableBlock:
    block_id: int
    bbox: Rect
    rows: list[float]
    cols: list[float]
    cells: list[Cell]
    assets: list[VisualAsset]
    metadata: dict = field(default_factory=dict)


def rect_from_any(box) -> Rect:
    arr = np.asarray(box, dtype="float32").reshape(-1)
    if arr.size >= 8:
        xs = arr[0::2]
        ys = arr[1::2]
        return Rect(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    if arr.size >= 4:
        x0, y0, x1, y1 = arr[:4]
        return Rect(float(x0), float(y0), float(x1), float(y1))
    raise ValueError(f"Unsupported box: {box!r}")


def merge_close_positions(values: Iterable[float], tol: float) -> list[float]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return []
    groups: list[list[float]] = [[vals[0]]]
    for value in vals[1:]:
        if abs(value - np.mean(groups[-1])) <= tol:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(np.median(g)) for g in groups]


def intervals_from_separators(seps: list[float], min_size: float) -> list[tuple[int, float, float]]:
    out = []
    for idx, (a, b) in enumerate(zip(seps[:-1], seps[1:])):
        if b - a >= min_size:
            out.append((idx, float(a), float(b)))
    return out


def local_median_color(rgb: np.ndarray, rect: Rect) -> str:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = rect.pad(0, 0, w, h).to_int()
    patch = rgb[y0:y1, x0:x1]
    if patch.size == 0:
        return "#000000"
    med = np.median(patch.reshape(-1, 3), axis=0).astype(int)
    return f"#{med[0]:02x}{med[1]:02x}{med[2]:02x}"


class MixedImageTextExtractor:
    """Rule-based prototype for product/image/text mixed comparison charts."""

    def __init__(
        self,
        min_line_fraction: float = 0.18,
        max_blocks: int = 8,
        debug: bool = True,
    ):
        self.min_line_fraction = min_line_fraction
        self.max_blocks = max_blocks
        self.debug = debug

    def extract(
        self,
        image_path: str | Path,
        tokens: list[OcrToken] | None = None,
        layout: dict | None = None,
    ) -> dict:
        image_path = Path(image_path)
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tokens = tokens or []
        layout_boxes = normalize_layout_boxes(layout)
        table_bbox = self._estimate_table_bbox_from_layout(rgb, layout_boxes)
        if table_bbox is None:
            table_bbox = self._estimate_table_bbox(rgb, tokens)
        blocks = self._split_blocks_from_layout(rgb, table_bbox, layout_boxes)
        if not blocks:
            blocks = self._split_blocks(rgb, table_bbox, tokens)
        table_blocks = []
        for block_id, block_bbox in enumerate(blocks):
            block_tokens = [t for t in tokens if block_bbox.overlap_area(t.bbox) > 0]
            table_blocks.append(self._extract_block(rgb, block_id, block_bbox, block_tokens))
        result = {
            "image": str(image_path),
            "image_size": {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
            "type": "mixed_image_text_table",
            "table_bbox": table_bbox.to_list(),
            "layout_box_count": len(layout_boxes),
            "blocks": [self._block_to_dict(block) for block in table_blocks],
        }
        return result

    def _estimate_table_bbox_from_layout(
        self, rgb: np.ndarray, layout_boxes: list[dict]
    ) -> Rect | None:
        if not layout_boxes:
            return None
        h, w = rgb.shape[:2]
        table_boxes = [b for b in layout_boxes if b["label"] == "table" and b["score"] >= 0.25]
        if table_boxes:
            best = max(table_boxes, key=lambda b: rect_from_any(b["bbox"]).area * b["score"])
            return rect_from_any(best["bbox"]).pad(4, 4, w, h)

        useful_labels = {"image", "vision_footnote", "text", "aside_text", "paragraph_title"}
        noise_labels = {
            "doc_title",
            "figure_title",
            "footer",
            "footer_image",
            "header",
            "header_image",
            "footnote",
        }
        useful = [
            b
            for b in layout_boxes
            if b["label"] in useful_labels
            and b["label"] not in noise_labels
            and b["score"] >= 0.30
        ]
        if len(useful) < 4:
            return None

        rects = [rect_from_any(b["bbox"]) for b in useful]
        x0 = min(r.x0 for r in rects)
        y0 = min(r.y0 for r in rects)
        x1 = max(r.x1 for r in rects)
        y1 = max(r.y1 for r in rects)
        return Rect(x0, y0, x1, y1).pad(10, 10, w, h)

    def _split_blocks_from_layout(
        self, rgb: np.ndarray, table_bbox: Rect, layout_boxes: list[dict]
    ) -> list[Rect]:
        if not layout_boxes:
            return []
        h, w = rgb.shape[:2]
        useful_labels = {"image", "vision_footnote", "text", "aside_text", "paragraph_title"}
        rects = [
            rect_from_any(b["bbox"])
            for b in layout_boxes
            if b["label"] in useful_labels
            and b["score"] >= 0.30
            and table_bbox.overlap_area(rect_from_any(b["bbox"])) > 0
        ]
        if len(rects) < 8:
            return []

        centers = sorted(r.cy for r in rects)
        gaps = [(b - a, a, b) for a, b in zip(centers[:-1], centers[1:])]
        if not gaps:
            return [table_bbox]
        # Only split on very large whitespace between repeated product-card blocks.
        cut_candidates = [
            (gap, (a + b) / 2.0)
            for gap, a, b in gaps
            if gap > max(52.0, table_bbox.h * 0.10)
            and table_bbox.y0 + table_bbox.h * 0.20 < (a + b) / 2.0 < table_bbox.y1 - table_bbox.h * 0.12
        ]
        if not cut_candidates:
            return [table_bbox]

        cuts = [table_bbox.y0] + [cut for _, cut in cut_candidates[: self.max_blocks - 1]] + [table_bbox.y1]
        cuts = merge_close_positions(cuts, tol=max(20, table_bbox.h * 0.03))
        blocks = []
        for _, y0, y1 in intervals_from_separators(cuts, min_size=max(120, table_bbox.h * 0.22)):
            block = Rect(table_bbox.x0, y0, table_bbox.x1, y1).pad(2, 2, w, h)
            # Keep only blocks that contain layout evidence.
            if sum(1 for r in rects if block.overlap_area(r) > 0) >= 4:
                blocks.append(block)
        return blocks or [table_bbox]

    def _estimate_table_bbox(self, rgb: np.ndarray, tokens: list[OcrToken]) -> Rect:
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        card_bbox = self._detect_light_table_card(rgb, hsv)
        if card_bbox is not None:
            return card_bbox.pad(3, 3, w, h)

        edges = cv2.Canny(gray, 50, 150)
        # Use edges and dark foreground as content evidence. Large saturated
        # backgrounds in product posters should not expand the table bbox.
        non_bg = (edges > 0) | (gray < 130)
        # Remove tiny isolated noise but keep weak product-chart lines.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(non_bg.astype("uint8") * 255, cv2.MORPH_CLOSE, kernel)

        if tokens:
            for token in tokens:
                x0, y0, x1, y1 = token.bbox.to_int()
                cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return Rect(0, 0, w, h)

        x0, x1 = np.percentile(xs, [0.5, 99.5])
        y0, y1 = np.percentile(ys, [0.5, 99.5])

        # Avoid letting a large standalone title dominate the top of product charts.
        row_energy = mask.sum(axis=1) / 255.0
        strong = np.where(row_energy > max(20, w * 0.08))[0]
        if len(strong) > 0:
            gaps = np.diff(strong)
            big_gaps = np.where(gaps > max(24, h * 0.035))[0]
            if len(big_gaps) > 0 and strong[big_gaps[0]] < h * 0.25:
                candidate_top = strong[big_gaps[0] + 1]
                if candidate_top < y1:
                    y0 = max(y0, candidate_top - 6)

        return Rect(float(x0), float(y0), float(x1), float(y1)).pad(8, 8, w, h)

    def _detect_light_table_card(self, rgb: np.ndarray, hsv: np.ndarray) -> Rect | None:
        h, w = rgb.shape[:2]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        # Product charts often place the actual table on a large white card over
        # a gray/blue background. This catches that card and drops title regions.
        light = ((sat < 75) & (val > 205)).astype("uint8") * 255
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(11, w // 45), max(11, h // 70))
        )
        light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats((light > 0).astype("uint8"), 8)
        candidates: list[Rect] = []
        for idx in range(1, count):
            x, y, ww, hh, area = stats[idx]
            rect = Rect(float(x), float(y), float(x + ww), float(y + hh))
            if area < w * h * 0.18:
                continue
            if rect.w < w * 0.55 or rect.h < h * 0.35:
                continue
            if rect.y0 < h * 0.02 and rect.h < h * 0.75:
                continue
            candidates.append(rect)
        if not candidates:
            return None
        # Prefer a large card below a possible title, not the whole screenshot.
        candidates.sort(key=lambda r: (r.area, r.y0), reverse=True)
        best = candidates[0]
        if best.area > w * h * 0.92:
            return None
        return best

    def _split_blocks(self, rgb: np.ndarray, table_bbox: Rect, tokens: list[OcrToken]) -> list[Rect]:
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = table_bbox.to_int()
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return [table_bbox]

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 140)
        content = edges > 0
        for token in tokens:
            if not table_bbox.contains_point(token.bbox.cx, token.bbox.cy, tol=5):
                continue
            tx0, ty0, tx1, ty1 = token.bbox.to_int()
            ty0, ty1 = max(y0, ty0) - y0, min(y1, ty1) - y0
            tx0, tx1 = max(x0, tx0) - x0, min(x1, tx1) - x0
            content[ty0:ty1, tx0:tx1] = True

        row_energy = content.mean(axis=1)
        low = row_energy < max(0.006, np.percentile(row_energy, 22))
        min_gap = max(12, int((y1 - y0) * 0.012))
        gaps = []
        start = None
        for idx, value in enumerate(low):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                if idx - start >= min_gap:
                    gaps.append((start, idx))
                start = None
        if start is not None and len(low) - start >= min_gap:
            gaps.append((start, len(low)))

        cuts = [y0]
        for a, b in gaps:
            cut = y0 + (a + b) / 2.0
            if table_bbox.y0 + table_bbox.h * 0.08 < cut < table_bbox.y1 - table_bbox.h * 0.08:
                cuts.append(cut)
        cuts.append(y1)
        cuts = merge_close_positions(cuts, tol=max(12, table_bbox.h * 0.02))

        blocks = []
        for _, a, b in intervals_from_separators(cuts, min_size=max(90, table_bbox.h * 0.15)):
            block = Rect(table_bbox.x0, a, table_bbox.x1, b)
            blocks.append(block.pad(2, 2, w, h))
        return blocks[: self.max_blocks] or [table_bbox]

    def _extract_block(
        self,
        rgb: np.ndarray,
        block_id: int,
        block_bbox: Rect,
        tokens: list[OcrToken],
    ) -> TableBlock:
        row_seps = self._detect_horizontal_separators(rgb, block_bbox, tokens)
        col_seps = self._detect_vertical_separators(rgb, block_bbox, tokens)
        row_seps = self._supplement_rows_with_tokens(row_seps, block_bbox, tokens)
        col_seps = self._supplement_cols_with_visuals(rgb, col_seps, block_bbox, tokens)

        row_intervals = intervals_from_separators(row_seps, min_size=max(14, block_bbox.h * 0.025))
        col_intervals = intervals_from_separators(col_seps, min_size=max(18, block_bbox.w * 0.025))
        rows = [(idx, y0, y1) for idx, (_, y0, y1) in enumerate(row_intervals)]
        cols = [(idx, x0, x1) for idx, (_, x0, x1) in enumerate(col_intervals)]
        assets = self._detect_visual_assets(rgb, block_bbox, row_seps, col_seps)

        cells = []
        for r_idx, ry0, ry1 in rows:
            for c_idx, cx0, cx1 in cols:
                rect = Rect(cx0, ry0, cx1, ry1)
                cell_tokens = self._tokens_in_rect(tokens, rect)
                cell_assets = [
                    a
                    for a in assets
                    if rect.overlap_area(a.bbox) / max(1.0, a.bbox.area) > 0.45
                ]
                role = self._infer_cell_role(r_idx, c_idx, rect, block_bbox, row_seps, col_seps)
                cells.append(
                    Cell(
                        row=r_idx,
                        col=c_idx,
                        bbox=rect,
                        text=self._join_tokens(cell_tokens),
                        tokens=cell_tokens,
                        assets=cell_assets,
                        role=role,
                    )
                )

        self._infer_spans(cells, tokens, assets, row_seps, col_seps)
        metadata = {
            "row_count": len(rows),
            "col_count": len(cols),
            "token_count": len(tokens),
            "asset_count": len(assets),
            "classification": "mixed_image_text_table",
        }
        return TableBlock(
            block_id=block_id,
            bbox=block_bbox,
            rows=row_seps,
            cols=col_seps,
            cells=cells,
            assets=assets,
            metadata=metadata,
        )

    def _line_masks(self, rgb: np.ndarray, bbox: Rect) -> tuple[np.ndarray, np.ndarray]:
        x0, y0, x1, y1 = bbox.to_int()
        crop = rgb[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        # Adaptive binary catches gray separators on light charts and colored separators.
        bw_dark = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 9
        )
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        sat_edges = cv2.Canny(hsv[:, :, 1], 40, 120)
        mask = cv2.bitwise_or(bw_dark, sat_edges)
        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(16, int(bbox.w * self.min_line_fraction)), 1)
        )
        v_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(16, int(bbox.h * self.min_line_fraction)))
        )
        h_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, v_kernel)
        return h_lines, v_lines

    def _component_centers(
        self,
        mask: np.ndarray,
        offset_x: int,
        offset_y: int,
        min_w: float,
        min_h: float,
        axis: str,
    ) -> list[float]:
        count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype("uint8"), 8)
        positions = []
        for idx in range(1, count):
            x, y, w, h, area = stats[idx]
            if area < 8:
                continue
            if w >= min_w and h >= min_h:
                if axis == "y":
                    positions.append(offset_y + y + h / 2.0)
                else:
                    positions.append(offset_x + x + w / 2.0)
        return positions

    def _detect_horizontal_separators(
        self, rgb: np.ndarray, bbox: Rect, tokens: list[OcrToken]
    ) -> list[float]:
        x0, y0, _, _ = bbox.to_int()
        h_lines, _ = self._line_masks(rgb, bbox)
        candidates = [bbox.y0, bbox.y1]
        candidates += self._component_centers(
            h_lines,
            x0,
            y0,
            min_w=max(30, bbox.w * 0.32),
            min_h=1,
            axis="y",
        )
        # Text row gaps are useful in weak-line product charts.
        if tokens:
            token_ys = sorted(t.bbox.cy for t in tokens if bbox.contains_point(t.bbox.cx, t.bbox.cy))
            if len(token_ys) >= 3:
                clusters = merge_close_positions(token_ys, tol=max(8, bbox.h * 0.012))
                for a, b in zip(clusters[:-1], clusters[1:]):
                    if b - a > max(18, bbox.h * 0.035):
                        candidates.append((a + b) / 2.0)
        return merge_close_positions(candidates, tol=max(4, bbox.h * 0.008))

    def _detect_vertical_separators(
        self, rgb: np.ndarray, bbox: Rect, tokens: list[OcrToken]
    ) -> list[float]:
        x0, y0, _, _ = bbox.to_int()
        _, v_lines = self._line_masks(rgb, bbox)
        candidates = [bbox.x0, bbox.x1]
        candidates += self._component_centers(
            v_lines,
            x0,
            y0,
            min_w=1,
            min_h=max(30, bbox.h * 0.28),
            axis="x",
        )

        # Left attribute rail is a stable anchor in product charts.
        attr_boundary = self._detect_left_attribute_boundary(rgb, bbox)
        if attr_boundary:
            candidates.append(attr_boundary)

        # OCR x gaps can recover weak vertical separators.
        if tokens:
            centers = sorted(t.bbox.cx for t in tokens if bbox.contains_point(t.bbox.cx, t.bbox.cy))
            if len(centers) >= 4:
                clusters = merge_close_positions(centers, tol=max(12, bbox.w * 0.025))
                gaps = [(b - a, a, b) for a, b in zip(clusters[:-1], clusters[1:])]
                for gap, a, b in sorted(gaps, reverse=True)[:8]:
                    if gap > max(40, bbox.w * 0.065):
                        candidates.append((a + b) / 2.0)
        return merge_close_positions(candidates, tol=max(4, bbox.w * 0.007))

    def _detect_left_attribute_boundary(self, rgb: np.ndarray, bbox: Rect) -> float | None:
        x0, y0, x1, y1 = bbox.to_int()
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype("float32")
        val = hsv[:, :, 2].astype("float32")
        # Attribute rails are often blue/saturated or a separated white panel.
        col_score = ((sat > 35) & (val > 80)).mean(axis=0)
        smooth = cv2.blur(col_score.reshape(1, -1), (max(5, crop.shape[1] // 80), 1)).ravel()
        limit = int(crop.shape[1] * 0.28)
        if limit < 20:
            return None
        left = smooth[:limit]
        if left.max() > 0.18:
            xs = np.where(left > max(0.08, left.max() * 0.35))[0]
            if len(xs) > 5:
                boundary = int(xs.max()) + x0
                if bbox.x0 + bbox.w * 0.04 < boundary < bbox.x0 + bbox.w * 0.30:
                    return float(boundary)
        return None

    def _supplement_rows_with_tokens(
        self, row_seps: list[float], bbox: Rect, tokens: list[OcrToken]
    ) -> list[float]:
        if not tokens:
            return row_seps
        candidates = list(row_seps)
        token_heights = [t.bbox.h for t in tokens if t.bbox.h > 2]
        median_h = float(np.median(token_heights)) if token_heights else bbox.h * 0.04
        centers = merge_close_positions(
            [t.bbox.cy for t in tokens if bbox.contains_point(t.bbox.cx, t.bbox.cy)],
            tol=max(8, median_h * 0.85),
        )
        for a, b in zip(centers[:-1], centers[1:]):
            if b - a > max(1.8 * median_h, bbox.h * 0.035):
                candidates.append((a + b) / 2.0)
        return merge_close_positions(candidates, tol=max(4, median_h * 0.4))

    def _supplement_cols_with_visuals(
        self, rgb: np.ndarray, col_seps: list[float], bbox: Rect, tokens: list[OcrToken]
    ) -> list[float]:
        assets = self._detect_visual_assets(rgb, bbox, [bbox.y0, bbox.y1], col_seps)
        header_assets = [
            a
            for a in assets
            if a.kind in {"product_image", "badge"}
            and a.bbox.cy < bbox.y0 + bbox.h * 0.36
            and a.bbox.area > bbox.area * 0.001
        ]
        candidates = list(col_seps)
        if len(header_assets) >= 2:
            centers = sorted(a.bbox.cx for a in header_assets)
            for a, b in zip(centers[:-1], centers[1:]):
                if b - a > bbox.w * 0.06:
                    candidates.append((a + b) / 2.0)
            left = max(bbox.x0, centers[0] - np.median(np.diff(centers)) / 2.0)
            candidates.append(left)
        return merge_close_positions(candidates, tol=max(5, bbox.w * 0.01))

    def _detect_visual_assets(
        self,
        rgb: np.ndarray,
        bbox: Rect,
        row_seps: list[float],
        col_seps: list[float],
    ) -> list[VisualAsset]:
        x0, y0, x1, y1 = bbox.to_int()
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return []
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        sat = hsv[:, :, 1]
        edges = cv2.Canny(gray, 45, 140)
        # Remove long table separators from asset candidates.
        h_lines, v_lines = self._line_masks(rgb, bbox)
        line_mask = cv2.bitwise_or(h_lines, v_lines)
        content = ((sat > 45) | (edges > 0)).astype("uint8") * 255
        content[line_mask > 0] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats((content > 0).astype("uint8"), 8)
        assets = []
        min_area = max(24, bbox.area * 0.00025)
        for idx in range(1, count):
            x, y, w, h, area = stats[idx]
            if area < min_area or w < 5 or h < 5:
                continue
            rect = Rect(x0 + x, y0 + y, x0 + x + w, y0 + y + h)
            if rect.w > bbox.w * 0.92 or rect.h > bbox.h * 0.92:
                continue
            aspect = rect.w / max(1.0, rect.h)
            fill = area / max(1.0, rect.area)
            kind = "icon"
            if rect.cy < bbox.y0 + bbox.h * 0.35 and rect.area > bbox.area * 0.0015:
                kind = "product_image"
            elif aspect > 2.0 and fill > 0.35:
                kind = "badge"
            elif rect.area > bbox.area * 0.012:
                kind = "image_region"
            assets.append(
                VisualAsset(
                    kind=kind,
                    bbox=rect,
                    score=float(min(1.0, fill)),
                    color=local_median_color(rgb, rect),
                )
            )
        return self._dedupe_assets(assets)

    def _dedupe_assets(self, assets: list[VisualAsset]) -> list[VisualAsset]:
        assets = sorted(assets, key=lambda a: a.bbox.area, reverse=True)
        kept: list[VisualAsset] = []
        for asset in assets:
            duplicate = False
            for prev in kept:
                overlap = asset.bbox.overlap_area(prev.bbox)
                if overlap / max(1.0, min(asset.bbox.area, prev.bbox.area)) > 0.75:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(asset)
        return sorted(kept, key=lambda a: (a.bbox.y0, a.bbox.x0))

    def _tokens_in_rect(self, tokens: list[OcrToken], rect: Rect) -> list[OcrToken]:
        out = []
        for token in tokens:
            overlap = rect.overlap_area(token.bbox)
            if rect.contains_point(token.bbox.cx, token.bbox.cy, tol=2) or overlap / max(1.0, token.bbox.area) > 0.45:
                out.append(token)
        return sorted(out, key=lambda t: (t.bbox.y0, t.bbox.x0))

    def _join_tokens(self, tokens: list[OcrToken]) -> str:
        if not tokens:
            return ""
        heights = [t.bbox.h for t in tokens if t.bbox.h > 2]
        tol = max(6.0, (float(np.median(heights)) if heights else 12.0) * 0.72)
        lines: list[list[OcrToken]] = []
        for token in sorted(tokens, key=lambda t: (t.bbox.cy, t.bbox.x0)):
            placed = False
            for line in lines:
                if abs(np.mean([t.bbox.cy for t in line]) - token.bbox.cy) <= tol:
                    line.append(token)
                    placed = True
                    break
            if not placed:
                lines.append([token])
        parts = []
        for line in lines:
            texts = [t.text.strip() for t in sorted(line, key=lambda t: t.bbox.x0) if t.text.strip()]
            if texts:
                parts.append(" ".join(texts))
        return "\n".join(parts)

    def _infer_cell_role(
        self,
        row_idx: int,
        col_idx: int,
        rect: Rect,
        block_bbox: Rect,
        row_seps: list[float],
        col_seps: list[float],
    ) -> str:
        if row_idx == 0 and col_idx == 0:
            return "corner"
        if row_idx == 0 or rect.cy < block_bbox.y0 + block_bbox.h * 0.20:
            return "column_header"
        if col_idx == 0 or rect.cx < block_bbox.x0 + block_bbox.w * 0.18:
            return "row_header"
        return "body"

    def _infer_spans(
        self,
        cells: list[Cell],
        tokens: list[OcrToken],
        assets: list[VisualAsset],
        row_seps: list[float],
        col_seps: list[float],
    ) -> None:
        # Mark visible objects that cross separator bands. The grid is kept intact,
        # but the owner cell records rowspan/colspan for HTML reconstruction.
        evidence = [(t.bbox, "text") for t in tokens] + [(a.bbox, a.kind) for a in assets]
        for bbox, _kind in evidence:
            row_indices = [
                idx
                for idx, (_, y0, y1) in enumerate(intervals_from_separators(row_seps, min_size=1))
                if bbox.overlap_area(Rect(col_seps[0], y0, col_seps[-1], y1)) / max(1.0, bbox.area) > 0.18
            ]
            col_indices = [
                idx
                for idx, (_, x0, x1) in enumerate(intervals_from_separators(col_seps, min_size=1))
                if bbox.overlap_area(Rect(x0, row_seps[0], x1, row_seps[-1])) / max(1.0, bbox.area) > 0.18
            ]
            if len(row_indices) <= 1 and len(col_indices) <= 1:
                continue
            owner = None
            best = -1.0
            for cell in cells:
                score = cell.bbox.overlap_area(bbox)
                if score > best:
                    best = score
                    owner = cell
            if owner is None:
                continue
            if len(row_indices) > 1:
                owner.rowspan = max(owner.rowspan, max(row_indices) - min(row_indices) + 1)
            if len(col_indices) > 1:
                owner.colspan = max(owner.colspan, max(col_indices) - min(col_indices) + 1)

    def _block_to_dict(self, block: TableBlock) -> dict:
        return {
            "block_id": block.block_id,
            "bbox": block.bbox.to_list(),
            "rows": [float(v) for v in block.rows],
            "cols": [float(v) for v in block.cols],
            "metadata": block.metadata,
            "assets": [
                {
                    "kind": a.kind,
                    "bbox": a.bbox.to_list(),
                    "score": a.score,
                    "color": a.color,
                }
                for a in block.assets
            ],
            "cells": [
                {
                    "row": c.row,
                    "col": c.col,
                    "bbox": c.bbox.to_list(),
                    "text": c.text,
                    "rowspan": c.rowspan,
                    "colspan": c.colspan,
                    "role": c.role,
                    "assets": [
                        {
                            "kind": a.kind,
                            "bbox": a.bbox.to_list(),
                            "score": a.score,
                            "color": a.color,
                        }
                        for a in c.assets
                    ],
                    "tokens": [
                        {
                            "text": t.text,
                            "bbox": t.bbox.to_list(),
                            "score": t.score,
                            "source": t.source,
                        }
                        for t in c.tokens
                    ],
                }
                for c in block.cells
            ],
        }


def result_to_html(result: dict, image_path: str | Path | None = None) -> str:
    title = html.escape(Path(result["image"]).name)
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body{font-family:Arial,'Noto Sans CJK SC',sans-serif;margin:24px;color:#161616;background:#fff}",
        ".source{max-width:420px;height:auto;border:1px solid #ddd;margin-bottom:16px}",
        "table{border-collapse:collapse;margin:16px 0 32px;width:100%;table-layout:fixed}",
        "td,th{border:1px solid #bfc7d1;padding:6px 8px;vertical-align:middle;white-space:pre-wrap;font-size:13px}",
        "th{background:#f2f5f8;font-weight:600}",
        ".row-header{background:#eef7ff;font-weight:600}",
        ".asset{font-size:11px;color:#666;margin-top:3px}",
        ".empty{color:#aaa}",
        "</style></head><body>",
        f"<h1>{title}</h1>",
    ]
    if image_path:
        encoded = image_to_data_uri(image_path)
        parts.append(f"<img class='source' src='{encoded}' alt='source image'>")
    for block in result["blocks"]:
        rows = int(block["metadata"].get("row_count") or 0)
        cols = int(block["metadata"].get("col_count") or 0)
        grid: dict[tuple[int, int], dict] = {(c["row"], c["col"]): c for c in block["cells"]}
        occupied: set[tuple[int, int]] = set()
        parts.append(f"<h2>Block {block['block_id']}</h2>")
        parts.append("<table>")
        for r in range(rows):
            parts.append("<tr>")
            for c in range(cols):
                if (r, c) in occupied:
                    continue
                cell = grid.get((r, c))
                if cell is None:
                    parts.append("<td class='empty'></td>")
                    continue
                tag = "th" if cell["role"] in {"column_header", "corner"} else "td"
                klass = "row-header" if cell["role"] == "row_header" else ""
                rs = max(1, int(cell.get("rowspan") or 1))
                cs = max(1, int(cell.get("colspan") or 1))
                for rr in range(r, min(rows, r + rs)):
                    for cc in range(c, min(cols, c + cs)):
                        if (rr, cc) != (r, c):
                            occupied.add((rr, cc))
                attrs = []
                if klass:
                    attrs.append(f"class='{klass}'")
                if rs > 1:
                    attrs.append(f"rowspan='{rs}'")
                if cs > 1:
                    attrs.append(f"colspan='{cs}'")
                text = html.escape(cell.get("text") or "")
                asset_label = ""
                if cell.get("assets"):
                    labels = ", ".join(a["kind"] for a in cell["assets"][:3])
                    asset_label = f"<div class='asset'>[{html.escape(labels)}]</div>"
                parts.append(f"<{tag} {' '.join(attrs)}>{text}{asset_label}</{tag}>")
            parts.append("</tr>")
        parts.append("</table>")
    parts.append("</body></html>")
    return "\n".join(parts)


def image_to_data_uri(image_path: str | Path) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def save_result(
    result: dict,
    image_path: str | Path,
    out_dir: str | Path,
    debug: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    stem = Path(image_path).stem
    sample_dir = out_dir / stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    json_path = sample_dir / "structure.json"
    html_path = sample_dir / "table.html"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(result_to_html(result, image_path), encoding="utf-8")
    paths = {"json": str(json_path), "html": str(html_path)}
    if debug:
        debug_path = sample_dir / "debug.png"
        draw_debug(image_path, result, debug_path)
        paths["debug"] = str(debug_path)
    return paths


def draw_debug(image_path: str | Path, result: dict, out_path: str | Path) -> None:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return
    for block in result["blocks"]:
        rect = Rect(*block["bbox"]).to_int()
        cv2.rectangle(bgr, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)
        for y in block["rows"]:
            cv2.line(bgr, (rect[0], int(round(y))), (rect[2], int(round(y))), (0, 180, 255), 1)
        for x in block["cols"]:
            cv2.line(bgr, (int(round(x)), rect[1]), (int(round(x)), rect[3]), (0, 255, 0), 1)
        for asset in block["assets"]:
            a = Rect(*asset["bbox"]).to_int()
            cv2.rectangle(bgr, (a[0], a[1]), (a[2], a[3]), (255, 180, 0), 1)
        for cell in block["cells"]:
            if cell.get("text"):
                c = Rect(*cell["bbox"]).to_int()
                cv2.putText(
                    bgr,
                    f"{cell['row']},{cell['col']}",
                    (c[0] + 2, c[1] + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)


def load_ocr_json(path: str | Path) -> dict[str, list[OcrToken]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {"*": [normalize_token(item) for item in raw]}
    out: dict[str, list[OcrToken]] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "tokens" in value:
            value = value["tokens"]
        out[str(key)] = [normalize_token(item) for item in value]
    return out


def normalize_token(item: dict) -> OcrToken:
    text = str(item.get("text") or item.get("rec_text") or item.get("label") or "")
    box = item.get("bbox") or item.get("box") or item.get("dt_box") or item.get("points")
    if box is None:
        raise ValueError(f"OCR item has no bbox: {item}")
    score = float(item.get("score") or item.get("confidence") or item.get("rec_score") or 1.0)
    return OcrToken(text=text, bbox=rect_from_any(box), score=score, source=str(item.get("source") or "json"))


def normalize_layout_boxes(layout: dict | None) -> list[dict]:
    if not isinstance(layout, dict):
        return []
    boxes = layout.get("boxes") or layout.get("layout_boxes") or []
    out = []
    for item in boxes:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox") or item.get("box")
        if bbox is None:
            continue
        label = str(item.get("label") or item.get("type") or item.get("category") or "")
        score = float(item.get("score") or item.get("confidence") or 1.0)
        class_id = int(item.get("class_id") or item.get("cls_id") or -1)
        out.append({"label": label, "score": score, "class_id": class_id, "bbox": rect_from_any(bbox).to_list()})
    return out


def load_layout_json(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "boxes" in raw and "image" in raw:
        return {Path(str(raw["image"])).name: raw}
    if isinstance(raw, dict):
        out = {}
        for key, value in raw.items():
            if isinstance(value, dict) and "result" in value:
                value = value["result"]
            if isinstance(value, dict):
                out[str(key)] = value
        return out
    return {}


def load_layout_dir(path: str | Path) -> dict[str, dict]:
    root = Path(path)
    out: dict[str, dict] = {}
    if not root.exists():
        return out
    results_json = root / "results.json"
    if results_json.is_file():
        out.update(load_layout_json(results_json))
    for layout_path in root.glob("*/layout.json"):
        data = json.loads(layout_path.read_text(encoding="utf-8"))
        image_name = Path(str(data.get("image") or layout_path.parent.name)).name
        out[image_name] = data
    return out


def collect_images(paths: Iterable[str | Path]) -> list[Path]:
    images: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            images.extend(p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS)
        elif path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
    return images
