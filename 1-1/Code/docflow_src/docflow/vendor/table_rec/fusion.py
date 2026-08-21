from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont

from docflow.vendor.table_rec.mixed_rules import Rect, rect_from_any
from docflow.vendor.table_rec.structure_postprocess import StructurePostProcessor


LAYOUT_CONTENT_LABELS = {
    "table",
    "image",
    "text",
    "vision_footnote",
    "aside_text",
    "paragraph_title",
    "content",
}
LAYOUT_TITLE_LABELS = {"doc_title", "figure_title", "table_title", "chart_title"}
LAYOUT_NOISE_LABELS = {
    "footer",
    "footer_image",
    "header",
    "header_image",
    "footnote",
    "seal",
}


@dataclass
class LayoutObject:
    label: str
    bbox: Rect
    score: float = 1.0
    class_id: int = -1
    role: str = "content"
    image_path: str | None = None
    text: str = ""


@dataclass
class TableCell:
    row: int
    col: int
    bbox: Rect
    text: str = ""
    rowspan: int = 1
    colspan: int = 1
    source_rowspan: int = 1
    source_colspan: int = 1
    role: str = "body"
    layout_objects: list[LayoutObject] = field(default_factory=list)
    ocr_objects: list[LayoutObject] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class VisualDecision:
    render: bool = True


@dataclass
class FusedTable:
    image: str
    image_size: dict
    status: str
    table_bbox: Rect
    cells: list[TableCell]
    layout_objects: list[LayoutObject]
    row_count: int
    col_count: int
    ocr_objects: list[LayoutObject] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def normalize_layout(layout: dict | None) -> list[LayoutObject]:
    if not isinstance(layout, dict):
        return []
    boxes = layout.get("boxes") or layout.get("layout_boxes") or []
    objects: list[LayoutObject] = []
    for item in boxes:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox") or item.get("box")
        if bbox is None:
            continue
        label = str(item.get("label") or item.get("type") or item.get("category") or "")
        score = float(item.get("score") or item.get("confidence") or 1.0)
        class_id = int(item.get("class_id") or item.get("cls_id") or -1)
        role = layout_role(label)
        objects.append(
            LayoutObject(
                label=label,
                bbox=rect_from_any(bbox),
                score=score,
                class_id=class_id,
                role=role,
            )
        )
    return objects


def layout_role(label: str) -> str:
    if label in LAYOUT_TITLE_LABELS:
        return "title"
    if label in LAYOUT_NOISE_LABELS:
        return "noise"
    if label == "table":
        return "table_region"
    if label == "image":
        return "visual_asset"
    if label in {"vision_footnote", "text", "aside_text", "paragraph_title", "content"}:
        return "text_object"
    return "content"


def normalize_table_result(table_result: dict | None) -> tuple[list[TableCell], dict]:
    if not isinstance(table_result, dict):
        return [], {"status": "missing_table_result"}
    if table_result.get("status") not in {None, "ok"}:
        return [], {"status": table_result.get("status"), "reason": table_result.get("reason")}

    boxes = [rect_from_any(box) for box in table_result.get("bbox") or []]
    logic_points = table_result.get("logic_points") or []
    ocr_objects = normalize_ocr_objects(table_result)
    html_text = str(table_result.get("html") or table_result.get("pred_html") or "")
    html_cells = parse_html_cells(html_text)
    cells: list[TableCell] = []

    if boxes and len(logic_points) == len(boxes):
        for idx, (bbox, logic) in enumerate(zip(boxes, logic_points)):
            if not isinstance(logic, (list, tuple)) or len(logic) < 4:
                continue
            row0, row1, col0, col1 = [safe_int(value, 0) for value in logic[:4]]
            row = min(row0, row1)
            col = min(col0, col1)
            rowspan = abs(row1 - row0) + 1
            colspan = abs(col1 - col0) + 1
            cells.append(
                TableCell(
                    row=row,
                    col=col,
                    bbox=bbox,
                    rowspan=max(1, rowspan),
                    colspan=max(1, colspan),
                    source_rowspan=max(1, rowspan),
                    source_colspan=max(1, colspan),
                    role="column_header" if row == 0 else "body",
                )
            )
        assign_ocr_text_to_cells(cells, ocr_objects)
    elif html_cells:
        for idx, hcell in enumerate(html_cells):
            bbox = boxes[idx] if idx < len(boxes) else Rect(0, 0, 0, 0)
            role = "column_header" if hcell["tag"] == "th" or hcell["row"] == 0 else "body"
            cells.append(
                TableCell(
                    row=hcell["row"],
                    col=hcell["col"],
                    bbox=bbox,
                    text=hcell["text"],
                    rowspan=hcell["rowspan"],
                    colspan=hcell["colspan"],
                    source_rowspan=hcell["rowspan"],
                    source_colspan=hcell["colspan"],
                    role=role,
                )
            )
    else:
        cells = cells_from_boxes(boxes)

    row_count = 0
    col_count = 0
    for cell in cells:
        row_count = max(row_count, cell.row + cell.rowspan)
        col_count = max(col_count, cell.col + cell.colspan)
    return cells, {
        "status": "ok" if cells else "empty_table_cells",
        "html_cell_count": len(html_cells),
        "box_count": len(boxes),
        "logic_count": len(logic_points),
        "ocr_count": len(ocr_objects),
        "_ocr_objects": ocr_objects,
        "row_count": row_count,
        "col_count": col_count,
    }


def normalize_ocr_objects(table_result: dict | None) -> list[LayoutObject]:
    if not isinstance(table_result, dict):
        return []
    out: list[LayoutObject] = []
    for item in table_result.get("ocr_result") or []:
        if not isinstance(item, dict):
            continue
        bbox = item.get("rect") or item.get("bbox") or item.get("box") or item.get("poly")
        text = str(item.get("text") or "").strip()
        if not bbox or not text:
            continue
        out.append(
            LayoutObject(
                label="ocr_text",
                bbox=rect_from_any(bbox),
                score=float(item.get("score") or item.get("confidence") or 1.0),
                role="ocr_text",
                text=text,
            )
        )
    return out


def assign_ocr_text_to_cells(cells: list[TableCell], ocr_objects: list[LayoutObject]) -> None:
    if not cells or not ocr_objects:
        return
    cell_tokens: dict[int, list[LayoutObject]] = {idx: [] for idx, _ in enumerate(cells)}
    for obj in ocr_objects:
        best_idx = -1
        best_score = 0.0
        for idx, cell in enumerate(cells):
            if cell.bbox.area <= 0:
                continue
            overlap = cell.bbox.overlap_area(obj.bbox)
            overlap_ratio = overlap / max(1.0, obj.bbox.area)
            center_inside = cell.bbox.contains_point(obj.bbox.cx, obj.bbox.cy, tol=2)
            score = overlap_ratio + (0.75 if center_inside else 0.0)
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx >= 0 and best_score >= 0.20:
            cell_tokens[best_idx].append(obj)
    for idx, tokens in cell_tokens.items():
        if not tokens:
            continue
        cells[idx].text = join_ocr_lines(tokens)
        cells[idx].ocr_objects = list(tokens)


def join_ocr_lines(tokens: list[LayoutObject]) -> str:
    if not tokens:
        return ""
    tokens = sorted(tokens, key=lambda obj: (obj.bbox.cy, obj.bbox.x0))
    line_tol = max(8.0, median_size([obj.bbox.h for obj in tokens]) * 0.65)
    lines: list[list[LayoutObject]] = []
    for obj in tokens:
        if not lines or abs(obj.bbox.cy - float(np.mean([x.bbox.cy for x in lines[-1]]))) > line_tol:
            lines.append([obj])
        else:
            lines[-1].append(obj)
    return "\n".join(" ".join(x.text for x in sorted(line, key=lambda obj: obj.bbox.x0)) for line in lines)


def parse_html_cells(html_text: str) -> list[dict]:
    if not html_text:
        return []
    soup = BeautifulSoup(html_text, "html.parser")
    rows = soup.find_all("tr")
    occupied: set[tuple[int, int]] = set()
    out: list[dict] = []
    for r, row in enumerate(rows):
        c = 0
        for node in row.find_all(["td", "th"], recursive=False):
            while (r, c) in occupied:
                c += 1
            rowspan = safe_int(node.get("rowspan"), 1)
            colspan = safe_int(node.get("colspan"), 1)
            text = node.get_text("\n", strip=True)
            out.append(
                {
                    "row": r,
                    "col": c,
                    "rowspan": max(1, rowspan),
                    "colspan": max(1, colspan),
                    "text": text,
                    "tag": node.name,
                }
            )
            for rr in range(r, r + max(1, rowspan)):
                for cc in range(c, c + max(1, colspan)):
                    if (rr, cc) != (r, c):
                        occupied.add((rr, cc))
            c += max(1, colspan)
    return out


def safe_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def is_title_like(text: str) -> bool:
    value = "".join(str(text).split())
    if len(value) < 6:
        return False
    keywords = ["指南", "对比", "系列", "选购", "测试", "Benchmark", "Matrix", "平台"]
    return any(keyword in value for keyword in keywords)


def normalize_cell_key(text: str) -> str:
    value = "".join(str(text or "").split())
    return value.strip(":：；;,.，。")


class CellContentResolver:
    """Resolve text and visual evidence inside table cells.

    The resolver only decides whether a visual candidate should be rendered.
    Asset cropping is intentionally handled elsewhere and always preserves the
    detector's full bbox. This keeps content decisions separate from geometry
    so a weak visual heuristic cannot cut off product images or diagrams.
    """

    def resolve_cells(self, cells: list[TableCell]) -> None:
        for cell in cells:
            cell.text = self.clean_cell_text(cell)

    def clean_cell_text(self, cell: TableCell) -> str:
        if not cell.text:
            return cell.text
        lines = [self.clean_ocr_line(line.strip()) for line in cell.text.splitlines()]
        return "\n".join(lines)

    def decide(
        self,
        visual_bbox: Rect,
        cell_bbox: Rect,
        cell_role: str = "body",
        cell_text: str = "",
    ) -> VisualDecision:
        text = str(cell_text or "").strip()
        if not text or visual_bbox.area <= 0 or cell_bbox.area <= 0:
            return VisualDecision(render=True)

        area_ratio = visual_bbox.area / max(1.0, cell_bbox.area)
        height_ratio = visual_bbox.h / max(1.0, cell_bbox.h)

        if cell_role in {"column_header", "corner"}:
            return VisualDecision(render=True)

        if self.is_redundant_small_visual(text, visual_bbox, cell_bbox):
            return VisualDecision(render=False)

        if self.is_text_explained_visual(cell_role, area_ratio, height_ratio):
            return VisualDecision(render=False)

        return VisualDecision(render=True)

    @staticmethod
    def clean_ocr_line(line: str) -> str:
        value = line
        if value.startswith("SSS") and any(ch.isdigit() for ch in value):
            value = value[3:]
        if value.endswith("W") and value[:-1].isdigit() and len(value[:-1]) >= 5:
            value = value[:-2] + "W"
        return value

    @staticmethod
    def is_text_explained_visual(cell_role: str, area_ratio: float, height_ratio: float) -> bool:
        if cell_role in {"column_header", "corner"} and height_ratio < 0.45:
            return True
        if area_ratio > 0.78:
            return True
        return height_ratio > 0.72 and area_ratio > 0.30

    @staticmethod
    def is_redundant_small_visual(text: str, visual_bbox: Rect, cell_bbox: Rect) -> bool:
        compact = "".join(text.split())
        if len(compact) < 2:
            return False
        height_ratio = visual_bbox.h / max(1.0, cell_bbox.h)
        width_ratio = visual_bbox.w / max(1.0, cell_bbox.w)
        area_ratio = visual_bbox.area / max(1.0, cell_bbox.area)
        center_distance = abs(visual_bbox.cx - cell_bbox.cx) / max(1.0, cell_bbox.w)
        if area_ratio >= 0.22 or center_distance > 0.42:
            return False
        if height_ratio <= 0.48 and width_ratio <= 0.72 and len(compact) >= 3:
            return True
        return height_ratio <= 0.35 and len(compact) >= 2

    def decide_dict(self, obj: dict, cell: dict | None = None) -> VisualDecision:
        if obj.get("label") != "image":
            return VisualDecision(render=False)
        if not cell:
            return VisualDecision(render=True)
        return self.decide(
            Rect(*obj.get("bbox", [0, 0, 0, 0])),
            Rect(*cell.get("bbox", [0, 0, 0, 0])),
            str(cell.get("role") or "body"),
            str(cell.get("text") or ""),
        )


def cells_from_boxes(boxes: list[Rect]) -> list[TableCell]:
    if not boxes:
        return []
    row_centers = cluster_positions([b.cy for b in boxes], tolerance=max(8.0, median_size([b.h for b in boxes]) * 0.65))
    col_centers = cluster_positions([b.cx for b in boxes], tolerance=max(10.0, median_size([b.w for b in boxes]) * 0.45))
    cells: list[TableCell] = []
    for box in boxes:
        row = nearest_index(row_centers, box.cy)
        col = nearest_index(col_centers, box.cx)
        cells.append(TableCell(row=row, col=col, bbox=box))
    cells.sort(key=lambda c: (c.row, c.col))
    return cells


def cluster_positions(values: Iterable[float], tolerance: float) -> list[float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return []
    groups = [[vals[0]]]
    for value in vals[1:]:
        if abs(value - float(np.mean(groups[-1]))) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(np.median(g)) for g in groups]


def median_size(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if float(v) > 1]
    return float(np.median(vals)) if vals else 20.0


def nearest_index(values: list[float], value: float) -> int:
    if not values:
        return 0
    return int(min(range(len(values)), key=lambda idx: abs(values[idx] - value)))


class LayoutTableFusion:
    """Fuse layout detection and table recognition outputs.

    The table recognizer remains the primary structural source. Layout boxes are
    used to expand table extent, attach image/text objects, and repair spans.
    """

    def __init__(self, object_score_threshold: float = 0.30):
        self.object_score_threshold = float(object_score_threshold)

    def fuse(
        self,
        image_path: str | Path,
        layout: dict | None,
        table_result: dict | None,
    ) -> dict:
        image_path = Path(image_path)
        image_size = read_image_size(image_path)
        layout_objects = normalize_layout(layout)
        table_cells, table_meta = normalize_table_result(table_result)
        fused = self._fuse_cells(
            str(image_path),
            image_size,
            layout_objects,
            table_cells,
            table_meta,
        )
        return fused_to_dict(fused)

    def _fuse_cells(
        self,
        image: str,
        image_size: dict,
        layout_objects: list[LayoutObject],
        table_cells: list[TableCell],
        table_meta: dict,
    ) -> FusedTable:
        table_bbox = self._fused_table_bbox(image_size, layout_objects, table_cells)
        content_objects = self._select_content_objects(layout_objects, table_bbox)
        if not table_cells:
            table_cells = self._cells_from_layout_objects(content_objects, table_bbox)
            table_meta["fallback"] = "layout_objects_to_grid"

        source_rows = int(table_meta.get("row_count") or 0)
        source_cols = int(table_meta.get("col_count") or 0)
        self._attach_layout_objects(table_cells, content_objects)
        self._repair_roles(table_cells, table_bbox)
        self._remove_title_cells(table_cells, layout_objects, table_bbox, source_cols)
        removed_title_rows = self._drop_removed_title_rows(table_cells)
        if removed_title_rows:
            bounds = [cell.bbox for cell in table_cells if cell.bbox.area > 0]
            bounds.extend(obj.bbox for obj in content_objects if obj.role not in {"title", "noise"})
            table_bbox = union_rect(bounds).pad(4, 4, int(image_size["width"]), int(image_size["height"]))
        self._limit_implausible_spans(table_cells, source_rows, source_cols)
        self._repair_spans(table_cells, content_objects, source_rows, source_cols)
        self._limit_implausible_spans(table_cells, source_rows, source_cols)
        self._repair_repeated_attribute_rows(table_cells)
        structure_diag = StructurePostProcessor().process(
            table_cells,
            content_objects,
            table_bbox,
            image_size,
            table_meta,
        )
        if removed_title_rows:
            structure_diag["title_rows_removed"] = removed_title_rows
        self._fill_empty_cells(table_cells)
        CellContentResolver().resolve_cells(table_cells)

        row_count, col_count = table_shape(table_cells)
        public_table_meta = {k: v for k, v in table_meta.items() if not str(k).startswith("_")}
        diagnostics = {
            "table_recognition": public_table_meta,
            "structure_postprocess": structure_diag,
            "layout_objects_total": len(layout_objects),
            "layout_objects_used": len(content_objects),
            "span_repairs": sum(
                1
                for c in table_cells
                if c.rowspan != c.source_rowspan or c.colspan != c.source_colspan
            ),
            "empty_cells": sum(1 for c in table_cells if not c.text and not c.layout_objects),
        }
        return FusedTable(
            image=image,
            image_size=image_size,
            status="ok" if table_cells else "empty",
            table_bbox=table_bbox,
            cells=sorted(table_cells, key=lambda c: (c.row, c.col)),
            layout_objects=content_objects,
            row_count=row_count,
            col_count=col_count,
            ocr_objects=list(table_meta.get("_ocr_objects", [])),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _drop_removed_title_rows(cells: list[TableCell]) -> int:
        rows = sorted({cell.row for cell in cells})
        removed = {
            row
            for row in rows
            if any(cell.role == "title_removed" for cell in cells if cell.row == row)
            and all(not cell.text and not cell.layout_objects for cell in cells if cell.row == row)
        }
        if not removed:
            return 0
        cells[:] = [cell for cell in cells if cell.row not in removed]
        for cell in cells:
            original_row = cell.row
            cell.row -= sum(row < original_row for row in removed)
            cell.rowspan = max(1, cell.rowspan - sum(original_row < row < original_row + cell.rowspan for row in removed))
        return len(removed)

    def _fused_table_bbox(
        self,
        image_size: dict,
        layout_objects: list[LayoutObject],
        table_cells: list[TableCell],
    ) -> Rect:
        width = int(image_size["width"])
        height = int(image_size["height"])
        candidates: list[Rect] = []
        if table_cells:
            candidates.append(union_rect([c.bbox for c in table_cells if c.bbox.area > 0]))
        table_layouts = [
            obj.bbox
            for obj in layout_objects
            if obj.label == "table" and obj.score >= self.object_score_threshold
        ]
        candidates.extend(table_layouts)
        if candidates:
            base = union_rect(candidates).pad(4, 4, width, height)
        else:
            content = [
                obj.bbox
                for obj in layout_objects
                if obj.role not in {"title", "noise"} and obj.score >= self.object_score_threshold
            ]
            base = union_rect(content).pad(8, 8, width, height) if content else Rect(0, 0, width, height)

        # Expand to include layout objects that are close to or overlap the table.
        expanded = [base]
        for obj in layout_objects:
            if obj.role in {"title", "noise"} or obj.score < self.object_score_threshold:
                continue
            overlap = base.overlap_area(obj.bbox)
            near_x = obj.bbox.x1 >= base.x0 - base.w * 0.08 and obj.bbox.x0 <= base.x1 + base.w * 0.08
            near_y = obj.bbox.y1 >= base.y0 - base.h * 0.12 and obj.bbox.y0 <= base.y1 + base.h * 0.12
            if overlap > 0 or (near_x and near_y):
                expanded.append(obj.bbox)
        return union_rect(expanded).pad(4, 4, width, height)

    def _select_content_objects(
        self, layout_objects: list[LayoutObject], table_bbox: Rect
    ) -> list[LayoutObject]:
        selected = []
        for obj in layout_objects:
            if obj.score < self.object_score_threshold:
                continue
            if obj.role in {"title", "noise"}:
                continue
            overlap = table_bbox.overlap_area(obj.bbox)
            if table_bbox.contains_point(obj.bbox.cx, obj.bbox.cy, tol=8) or overlap / max(1.0, obj.bbox.area) > 0.20:
                selected.append(obj)
        return sorted(selected, key=lambda o: (o.bbox.y0, o.bbox.x0))

    def _cells_from_layout_objects(
        self, objects: list[LayoutObject], table_bbox: Rect
    ) -> list[TableCell]:
        text_like = [obj for obj in objects if obj.role in {"text_object", "visual_asset"}]
        if not text_like:
            return []
        rows = cluster_positions(
            [obj.bbox.cy for obj in text_like],
            tolerance=max(10.0, median_size([obj.bbox.h for obj in text_like]) * 0.75),
        )
        cols = cluster_positions(
            [obj.bbox.cx for obj in text_like],
            tolerance=max(18.0, median_size([obj.bbox.w for obj in text_like]) * 0.60),
        )
        cells = []
        for obj in text_like:
            row = nearest_index(rows, obj.bbox.cy)
            col = nearest_index(cols, obj.bbox.cx)
            cells.append(
                TableCell(
                    row=row,
                    col=col,
                    bbox=obj.bbox,
                    text="",
                    role="body",
                    layout_objects=[obj],
                    confidence=max(0.25, obj.score * 0.6),
                )
            )
        return merge_duplicate_cells(cells)

    def _attach_layout_objects(self, cells: list[TableCell], objects: list[LayoutObject]) -> None:
        if not cells:
            return
        for obj in objects:
            best_cell = None
            best_score = -1.0
            for cell in cells:
                overlap = cell.bbox.overlap_area(obj.bbox)
                center_inside = cell.bbox.contains_point(obj.bbox.cx, obj.bbox.cy, tol=4)
                score = overlap / max(1.0, obj.bbox.area)
                if center_inside:
                    score += 1.0
                if score > best_score:
                    best_score = score
                    best_cell = cell
            if best_cell is not None and best_score > 0.05:
                best_cell.layout_objects.append(obj)

    def _repair_roles(self, cells: list[TableCell], table_bbox: Rect) -> None:
        if not cells:
            return
        min_col = min(c.col for c in cells)
        min_row = min(c.row for c in cells)
        for cell in cells:
            if cell.row == min_row:
                cell.role = "column_header"
            if cell.col == min_col:
                cell.role = "row_header" if cell.row != min_row else "corner"

    def _remove_title_cells(
        self,
        cells: list[TableCell],
        layout_objects: list[LayoutObject],
        table_bbox: Rect,
        source_cols: int = 0,
    ) -> None:
        title_rects = [obj.bbox for obj in layout_objects if obj.role == "title" and obj.score >= 0.35]
        for cell in cells:
            if not cell.text:
                continue
            near_top = cell.row <= 1 or cell.bbox.cy < table_bbox.y0 + table_bbox.h * 0.22
            title_like_text = is_title_like(cell.text)
            wide_or_top_title = (
                near_top
                and title_like_text
                and (
                    cell.colspan >= max(2, int((source_cols or 1) * 0.45))
                    or cell.bbox.w >= table_bbox.w * 0.22
                    or cell.row <= 1
                )
            )
            if wide_or_top_title:
                cell.text = ""
                cell.role = "title_removed"
                cell.confidence = min(cell.confidence, 0.2)
                continue
            for title in title_rects:
                overlap = cell.bbox.overlap_area(title) / max(1.0, min(cell.bbox.area, title.area))
                if near_top and overlap > 0.20:
                    cell.text = ""
                    cell.role = "title_removed"
                    cell.confidence = min(cell.confidence, 0.2)
                    break

    def _limit_implausible_spans(
        self,
        cells: list[TableCell],
        source_rows: int,
        source_cols: int,
    ) -> None:
        if not cells:
            return
        for cell in cells:
            # Product-chart row headers are often visually tall, but if the table
            # recognizer claims they span almost the whole table, it usually means
            # it confused the side rail with a merged table cell.
            if source_rows and cell.rowspan >= max(4, int(source_rows * 0.45)):
                if cell.role in {"row_header", "corner"} or cell.col == min(c.col for c in cells):
                    cell.rowspan = 1
                    cell.confidence = min(cell.confidence, 0.45)
            if source_cols and cell.colspan >= max(4, int(source_cols * 0.70)):
                if cell.role not in {"column_header", "corner"}:
                    cell.colspan = 1
                    cell.confidence = min(cell.confidence, 0.45)

    def _repair_spans(
        self,
        cells: list[TableCell],
        objects: list[LayoutObject],
        max_rows: int = 0,
        max_cols: int = 0,
    ) -> None:
        if not cells:
            return
        row_bands = row_band_rects(cells)
        col_bands = col_band_rects(cells)
        median_cell_w = median_size([c.bbox.w for c in cells])
        median_cell_h = median_size([c.bbox.h for c in cells])
        by_pos = {(c.row, c.col): c for c in cells}
        for obj in objects:
            if obj.role == "table_region":
                continue
            # Small OCR/layout fragments inside one visual cell should not change
            # the structural grid. They only get attached as evidence.
            if obj.bbox.w < median_cell_w * 1.18 and obj.bbox.h < median_cell_h * 1.18:
                continue
            rows = [
                row
                for row, rect in row_bands.items()
                if rect.overlap_area(obj.bbox) / max(1.0, obj.bbox.area) > 0.12
            ]
            cols = [
                col
                for col, rect in col_bands.items()
                if rect.overlap_area(obj.bbox) / max(1.0, obj.bbox.area) > 0.12
            ]
            if len(rows) <= 1 and len(cols) <= 1:
                continue
            min_row = min(rows) if rows else nearest_index(sorted(row_bands), obj.bbox.cy)
            min_col = min(cols) if cols else nearest_index(sorted(col_bands), obj.bbox.cx)
            owner = by_pos.get((min_row, min_col)) or owner_cell(cells, obj.bbox)
            if owner is None:
                continue
            if len(rows) > 1:
                span = max(rows) - min(rows) + 1
                if max_rows:
                    span = min(span, max_rows - owner.row)
                owner.rowspan = max(owner.rowspan, max(1, span))
            if len(cols) > 1:
                span = max(cols) - min(cols) + 1
                if max_cols:
                    span = min(span, max_cols - owner.col)
                owner.colspan = max(owner.colspan, max(1, span))

    def _fill_empty_cells(self, cells: list[TableCell]) -> None:
        if not cells:
            return
        row_count, col_count = table_shape(cells)
        occupied = set()
        for cell in cells:
            for r in range(cell.row, min(row_count, cell.row + max(1, cell.rowspan))):
                for c in range(cell.col, min(col_count, cell.col + max(1, cell.colspan))):
                    occupied.add((r, c))
        rows = row_band_rects(cells)
        cols = col_band_rects(cells)
        for r in range(row_count):
            for c in range(col_count):
                if (r, c) in occupied:
                    continue
                rect = rows.get(r, Rect(0, 0, 0, 0)).intersect(cols.get(c, Rect(0, 0, 0, 0)))
                cells.append(TableCell(row=r, col=c, bbox=rect, confidence=0.1))

    def _repair_repeated_attribute_rows(self, cells: list[TableCell]) -> None:
        by_pos = {(c.row, c.col): c for c in cells}
        row_count, col_count = table_shape(cells)
        for row in range(1, row_count):
            row_header = normalize_cell_key((by_pos.get((row, 0)) or TableCell(row, 0, Rect(0, 0, 0, 0))).text)
            prev_header = normalize_cell_key((by_pos.get((row - 1, 0)) or TableCell(row - 1, 0, Rect(0, 0, 0, 0))).text)
            if not prev_header or len(prev_header) > 24:
                continue
            for col in range(1, col_count):
                cell = by_pos.get((row, col))
                if cell is None or not cell.text:
                    continue
                marker = normalize_cell_key(cell.text)
                if marker != prev_header:
                    continue
                if row_header and row_header == marker:
                    continue
                moves: list[tuple[TableCell, TableCell]] = []
                for move_col in range(col + 1, col_count):
                    src = by_pos.get((row, move_col))
                    dst = by_pos.get((row - 1, move_col - col + 1))
                    if src is None or dst is None or not src.text:
                        continue
                    if dst.text:
                        continue
                    moves.append((src, dst))
                if not moves:
                    continue
                for src, dst in moves:
                    dst.text = src.text
                    dst.layout_objects.extend(src.layout_objects)
                    dst.confidence = min(dst.confidence, src.confidence)
                    src.text = ""
                    src.layout_objects = []
                    src.confidence = min(src.confidence, 0.2)
                cell.text = ""
                cell.layout_objects = []
                cell.confidence = min(cell.confidence, 0.2)


def union_rect(rects: Iterable[Rect]) -> Rect:
    values = [r for r in rects if r and r.area >= 0]
    if not values:
        return Rect(0, 0, 0, 0)
    return Rect(
        min(r.x0 for r in values),
        min(r.y0 for r in values),
        max(r.x1 for r in values),
        max(r.y1 for r in values),
    )


def table_shape(cells: list[TableCell]) -> tuple[int, int]:
    rows = 0
    cols = 0
    for cell in cells:
        rows = max(rows, cell.row + max(1, cell.rowspan))
        cols = max(cols, cell.col + max(1, cell.colspan))
    return rows, cols


def row_band_rects(cells: list[TableCell]) -> dict[int, Rect]:
    out: dict[int, list[Rect]] = {}
    for cell in cells:
        out.setdefault(cell.row, []).append(cell.bbox)
    return {row: union_rect(rects) for row, rects in out.items()}


def col_band_rects(cells: list[TableCell]) -> dict[int, Rect]:
    out: dict[int, list[Rect]] = {}
    for cell in cells:
        out.setdefault(cell.col, []).append(cell.bbox)
    return {col: union_rect(rects) for col, rects in out.items()}


def owner_cell(cells: list[TableCell], bbox: Rect) -> TableCell | None:
    best = None
    best_score = -1.0
    for cell in cells:
        overlap = cell.bbox.overlap_area(bbox)
        center = cell.bbox.contains_point(bbox.cx, bbox.cy, tol=4)
        score = overlap / max(1.0, bbox.area) + (1.0 if center else 0.0)
        if score > best_score:
            best_score = score
            best = cell
    return best if best_score > 0 else None


def merge_duplicate_cells(cells: list[TableCell]) -> list[TableCell]:
    by_key: dict[tuple[int, int], TableCell] = {}
    for cell in cells:
        key = (cell.row, cell.col)
        if key not in by_key:
            by_key[key] = cell
            continue
        prev = by_key[key]
        prev.bbox = union_rect([prev.bbox, cell.bbox])
        prev.layout_objects.extend(cell.layout_objects)
        prev.text = "\n".join(x for x in [prev.text, cell.text] if x)
        prev.confidence = max(prev.confidence, cell.confidence)
    return sorted(by_key.values(), key=lambda c: (c.row, c.col))


def read_image_size(image_path: str | Path) -> dict:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return {"width": 0, "height": 0}
    h, w = bgr.shape[:2]
    return {"width": int(w), "height": int(h)}


def fused_to_dict(fused: FusedTable) -> dict:
    return {
        "image": fused.image,
        "image_size": fused.image_size,
        "status": fused.status,
        "table_bbox": fused.table_bbox.to_list(),
        "row_count": fused.row_count,
        "col_count": fused.col_count,
        "diagnostics": fused.diagnostics,
        "layout_objects": [layout_object_to_dict(obj) for obj in fused.layout_objects],
        "ocr_objects": [layout_object_to_dict(obj) for obj in fused.ocr_objects],
        "cells": [cell_to_dict(cell) for cell in fused.cells],
    }


def cell_to_dict(cell: TableCell) -> dict:
    return {
        "row": cell.row,
        "col": cell.col,
        "bbox": cell.bbox.to_list(),
        "text": cell.text,
        "rowspan": cell.rowspan,
        "colspan": cell.colspan,
        "source_rowspan": cell.source_rowspan,
        "source_colspan": cell.source_colspan,
        "role": cell.role,
        "confidence": cell.confidence,
        "layout_objects": [layout_object_to_dict(obj) for obj in cell.layout_objects],
        "ocr_objects": [layout_object_to_dict(obj) for obj in cell.ocr_objects],
    }


def layout_object_to_dict(obj: LayoutObject) -> dict:
    return {
        "label": obj.label,
        "bbox": obj.bbox.to_list(),
        "score": obj.score,
        "class_id": obj.class_id,
        "role": obj.role,
        "image_path": obj.image_path,
        "text": obj.text,
    }


def save_fused_result(result: dict, image_path: str | Path, out_dir: str | Path) -> dict:
    sample_dir = Path(out_dir) / Path(image_path).stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    crop_layout_images(result, image_path, sample_dir / "assets")
    json_path = sample_dir / "fused.json"
    html_path = sample_dir / "fused.html"
    debug_path = sample_dir / "fusion_debug.png"
    ocr_debug_path = sample_dir / "ocr_debug.png"
    ocr_rebuild_path = sample_dir / "ocr_text_rebuild.png"
    ocr_html_path = sample_dir / "ocr.html"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(fused_to_html(result, image_path), encoding="utf-8")
    draw_fusion_debug(image_path, result, debug_path)
    draw_ocr_debug(image_path, result, ocr_debug_path)
    draw_ocr_text_rebuild(result, ocr_rebuild_path)
    ocr_html_path.write_text(ocr_to_html(result, image_path), encoding="utf-8")
    return {
        "json": str(json_path),
        "html": str(html_path),
        "debug": str(debug_path),
        "ocr_debug": str(ocr_debug_path),
        "ocr_rebuild": str(ocr_rebuild_path),
        "ocr_html": str(ocr_html_path),
    }


def crop_layout_images(result: dict, image_path: str | Path, assets_dir: str | Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    h, w = image.shape[:2]
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    resolver = CellContentResolver()
    image_objects = []
    for cell in result.get("cells") or []:
        for obj in cell.get("layout_objects") or []:
            if obj.get("label") == "image":
                image_objects.append((obj, cell))
    for obj in result.get("layout_objects") or []:
        if obj.get("label") == "image":
            image_objects.append((obj, None))

    seen: dict[tuple[int, int, int, int], tuple[str, dict]] = {}
    for idx, (obj, cell) in enumerate(image_objects):
        rect = Rect(*obj.get("bbox", [0, 0, 0, 0])).pad(2, 2, w, h).to_int()
        decision = resolver.decide_dict(obj, cell)
        key = tuple(rect)
        if key in seen:
            rel, visual_evidence = seen[key]
            obj["image_path"] = rel
            obj["visual_evidence"] = visual_evidence
            obj["visual_kind"] = visual_evidence.get("kind", "unknown")
            obj["render_decision"] = "render" if decision.render else "suppress"
            continue
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            continue
        crop = image[y0:y1, x0:x1]
        visual_evidence = classify_visual_evidence(crop, Rect(*rect), cell)
        visual_evidence["crop_policy"] = "full_detector_bbox"
        visual_evidence["geometry_render_decision"] = "render" if decision.render else "suppress"
        obj["visual_evidence"] = visual_evidence
        obj["visual_kind"] = visual_evidence.get("kind", "unknown")
        obj["render_decision"] = visual_evidence["geometry_render_decision"]
        rel = f"assets/layout_image_{len(seen):03d}.png"
        cv2.imwrite(str(assets_dir.parent / rel), crop)
        seen[key] = (rel, visual_evidence)
        obj["image_path"] = rel


def classify_visual_evidence(crop: np.ndarray, crop_bbox: Rect, cell: dict | None) -> dict:
    ocr_objects = [obj for obj in (cell or {}).get("ocr_objects", []) if str(obj.get("text") or "").strip()]
    crop_ocr = []
    for obj in ocr_objects:
        bbox = Rect(*obj.get("bbox", [0, 0, 0, 0]))
        overlap = crop_bbox.overlap_area(bbox)
        if overlap / max(1.0, bbox.area) >= 0.45 or crop_bbox.contains_point(bbox.cx, bbox.cy, tol=2):
            crop_ocr.append(obj)

    text = join_text_values([str(obj.get("text") or "") for obj in crop_ocr])
    cell_text = str((cell or {}).get("text") or "")
    text_match = text_similarity(text, cell_text)
    text_area_ratio = sum(
        crop_bbox.overlap_area(Rect(*obj.get("bbox", [0, 0, 0, 0])))
        for obj in crop_ocr
    ) / max(1.0, crop_bbox.area)
    pixel = visual_residual_metrics(crop, crop_bbox, crop_ocr)
    residual_visual_ratio = pixel["residual_visual_ratio"]

    if residual_visual_ratio >= 0.055:
        kind = "semantic_visual"
    elif crop_ocr and text_match >= 0.72:
        kind = "text_snapshot"
    elif crop_ocr and text_area_ratio >= 0.28 and text_match >= 0.58:
        kind = "text_snapshot"
    elif not crop_ocr:
        kind = "semantic_visual"
    else:
        kind = "text_snapshot"

    return {
        "kind": kind,
        "crop_ocr_text": text,
        "crop_ocr_count": len(crop_ocr),
        "text_match_score": round(float(text_match), 4),
        "text_area_ratio": round(float(text_area_ratio), 4),
        "residual_visual_ratio": round(float(residual_visual_ratio), 4),
        "colored_ratio": round(float(pixel["colored_ratio"]), 4),
        "component_count": int(pixel["component_count"]),
    }


def visual_residual_metrics(crop: np.ndarray, crop_bbox: Rect, crop_ocr: list[dict]) -> dict:
    if crop.size == 0:
        return {
            "colored_ratio": 0.0,
            "component_count": 0,
            "residual_visual_ratio": 0.0,
        }
    h, w = crop.shape[:2]
    if h < 12 or w < 12:
        return {
            "colored_ratio": 0.0,
            "component_count": 0,
            "residual_visual_ratio": 0.0,
        }
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    colored_ratio = float(np.mean((saturation > 45) & (value > 80)))
    ink = (((gray < 185) | (saturation > 45)) & (value < 250)).astype(np.uint8)
    for obj in crop_ocr:
        bbox = Rect(*obj.get("bbox", [0, 0, 0, 0]))
        local = bbox.intersect(crop_bbox)
        x0 = int(max(0, round(local.x0 - crop_bbox.x0 - 2)))
        y0 = int(max(0, round(local.y0 - crop_bbox.y0 - 2)))
        x1 = int(min(w, round(local.x1 - crop_bbox.x0 + 2)))
        y1 = int(min(h, round(local.y1 - crop_bbox.y0 + 2)))
        if x1 > x0 and y1 > y0:
            ink[y0:y1, x0:x1] = 0

    n_labels, _labels, stats, _centers = cv2.connectedComponentsWithStats(ink, 8)
    components = []
    for idx in range(1, n_labels):
        x, y, cw, ch, area = stats[idx]
        if area >= 3:
            components.append((int(cw), int(ch), int(area)))
    residual_area = int(np.count_nonzero(ink))
    return {
        "colored_ratio": colored_ratio,
        "component_count": len(components),
        "residual_visual_ratio": residual_area / max(1.0, h * w),
    }


def join_text_values(values: list[str]) -> str:
    return "".join("".join(str(value or "").split()) for value in values)


def text_similarity(a: str, b: str) -> float:
    left = join_text_values([a])
    right = join_text_values([b])
    if not left:
        return 0.0
    if left in right:
        return 1.0
    if right and right in left:
        return min(1.0, len(right) / max(1.0, len(left)))
    return SequenceMatcher(None, left, right).ratio()


def fused_to_html(result: dict, image_path: str | Path | None = None) -> str:
    rows = int(result.get("row_count") or 0)
    cols = int(result.get("col_count") or 0)
    grid = {(c["row"], c["col"]): c for c in result.get("cells") or []}
    occupied: set[tuple[int, int]] = set()
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>",
        "body{font-family:Arial,'Noto Sans CJK SC',sans-serif;margin:24px;color:#151515}",
        ".source{max-width:420px;height:auto;border:1px solid #ddd;margin-bottom:16px}",
        "table{border-collapse:collapse;width:100%;table-layout:fixed;margin-top:16px}",
        "td,th{border:1px solid #aeb7c2;padding:6px 8px;vertical-align:middle;white-space:pre-wrap;font-size:13px}",
        "th{background:#f2f5f8;font-weight:700}.row-header{background:#eef7ff;font-weight:700}",
        ".diag{display:none}",
        ".cell-imgs{display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin-bottom:4px}",
        ".cell-imgs img{max-width:68px;max-height:58px;object-fit:contain}",
        "</style></head><body>",
        f"<h1>{html.escape(Path(str(result.get('image') or '')).name)}</h1>",
    ]
    if image_path:
        # Keep the source image as a relative/absolute reference instead of
        # embedding a large base64 blob; cropped cell assets remain local files.
        parts.append(f"<img class='source' src='{html.escape(str(Path(image_path).resolve()), quote=True)}'>")
    diag = result.get("diagnostics") or {}
    parts.append(f"<div class='diag'>{html.escape(json.dumps(diag, ensure_ascii=False))}</div>")
    parts.append("<table>")
    for r in range(rows):
        parts.append("<tr>")
        for c in range(cols):
            if (r, c) in occupied:
                continue
            cell = grid.get((r, c))
            if not cell:
                parts.append("<td></td>")
                continue
            tag = "th" if cell.get("role") in {"column_header", "corner"} else "td"
            klass = "row-header" if cell.get("role") == "row_header" else ""
            rowspan = max(1, int(cell.get("rowspan") or 1))
            colspan = max(1, int(cell.get("colspan") or 1))
            for rr in range(r, min(rows, r + rowspan)):
                for cc in range(c, min(cols, c + colspan)):
                    if (rr, cc) != (r, c):
                        occupied.add((rr, cc))
            attrs = []
            if klass:
                attrs.append(f"class='{klass}'")
            if rowspan > 1:
                attrs.append(f"rowspan='{rowspan}'")
            if colspan > 1:
                attrs.append(f"colspan='{colspan}'")
            text = html.escape(cell.get("text") or "")
            objs = cell.get("layout_objects") or []
            img_html = render_cell_images(objs, cell)
            parts.append(f"<{tag} {' '.join(attrs)}>{img_html}{text}</{tag}>")
        parts.append("</tr>")
    parts.append("</table></body></html>")
    return "\n".join(parts)


def render_cell_images(objs: list[dict], cell: dict | None = None) -> str:
    resolver = CellContentResolver()
    images = [
        obj.get("image_path")
        for obj in objs
        if obj.get("image_path")
        and obj.get("visual_kind") != "text_snapshot"
        and (obj.get("visual_kind") == "semantic_visual" or resolver.decide_dict(obj, cell).render)
    ]
    if not images:
        return ""
    tags = []
    for path in images[:4]:
        src = html.escape(str(path), quote=True)
        tags.append(f"<img src='{src}'>")
    return "<div class='cell-imgs'>" + "".join(tags) + "</div>"


def image_to_data_uri(image_path: str | Path) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def draw_fusion_debug(image_path: str | Path, result: dict, out_path: str | Path) -> None:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return
    table = Rect(*result.get("table_bbox", [0, 0, 0, 0])).to_int()
    cv2.rectangle(bgr, (table[0], table[1]), (table[2], table[3]), (255, 0, 255), 3)
    for cell in result.get("cells") or []:
        rect = Rect(*cell["bbox"]).to_int()
        color = (0, 180, 255) if cell.get("role") in {"column_header", "row_header", "corner"} else (0, 220, 0)
        cv2.rectangle(bgr, (rect[0], rect[1]), (rect[2], rect[3]), color, 1)
        cv2.putText(
            bgr,
            f"{cell['row']},{cell['col']}",
            (rect[0] + 2, rect[1] + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    for obj in result.get("layout_objects") or []:
        rect = Rect(*obj["bbox"]).to_int()
        color = (255, 80, 60) if obj.get("label") == "image" else (255, 160, 0)
        cv2.rectangle(bgr, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)


def draw_ocr_debug(image_path: str | Path, result: dict, out_path: str | Path) -> None:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return
    for idx, obj in enumerate(collect_ocr_objects(result), start=1):
        rect = Rect(*obj["bbox"]).to_int()
        cv2.rectangle(bgr, (rect[0], rect[1]), (rect[2], rect[3]), (40, 80, 255), 2)
        cv2.putText(
            bgr,
            str(idx),
            (rect[0], max(12, rect[1] - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (40, 80, 255),
            1,
            cv2.LINE_AA,
        )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)


def draw_ocr_text_rebuild(result: dict, out_path: str | Path) -> None:
    image_size = result.get("image_size") or {}
    width = int(image_size.get("width") or 0)
    height = int(image_size.get("height") or 0)
    if width <= 0 or height <= 0:
        return
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
    for idx, obj in enumerate(collect_ocr_objects(result), start=1):
        bbox = Rect(*obj["bbox"])
        text = str(obj.get("text") or "")
        if not text:
            continue
        color = ocr_palette(idx)
        rect = [int(round(v)) for v in bbox.to_list()]
        draw.rectangle(rect, outline=color, width=2)
        font_size = max(12, min(28, int(bbox.h * 0.82)))
        font = font_cache.setdefault(font_size, load_cjk_font(font_size))
        x = int(round(bbox.x0 + 2))
        y = int(round(bbox.y0 + max(0, (bbox.h - font_size) * 0.35)))
        draw.text((x, y), text, fill=(30, 30, 30), font=font)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def ocr_to_html(result: dict, image_path: str | Path | None = None) -> str:
    image_size = result.get("image_size") or {}
    width = int(image_size.get("width") or 0)
    height = int(image_size.get("height") or 0)
    ocr_objects = collect_ocr_objects(result)
    image_src = html.escape(str(Path(image_path).resolve()), quote=True) if image_path else ""
    rebuild_src = "ocr_text_rebuild.png"
    debug_src = "ocr_debug.png"
    boxes = []
    rows = []
    for idx, obj in enumerate(ocr_objects, start=1):
        bbox = Rect(*obj["bbox"])
        text = html.escape(str(obj.get("text") or ""))
        score = float(obj.get("score") or 0.0)
        boxes.append(
            "<div class='ocr-box' "
            f"style='left:{bbox.x0}px;top:{bbox.y0}px;width:{bbox.w}px;height:{bbox.h}px'>"
            f"<span>{idx}</span></div>"
        )
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{text}</td>"
            f"<td>{score:.3f}</td>"
            f"<td>{', '.join(f'{v:.1f}' for v in bbox.to_list())}</td>"
            "</tr>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{font-family:Arial,'Noto Sans CJK SC',sans-serif;margin:24px;color:#151515}"
        ".views{display:grid;grid-template-columns:repeat(2,minmax(320px,1fr));gap:18px;align-items:start}"
        ".panel{overflow:auto}"
        ".stage{position:relative;display:inline-block;border:1px solid #c8d0da;line-height:0;overflow:auto;max-width:100%}"
        ".stage img{display:block;max-width:none}"
        ".ocr-box{position:absolute;border:2px solid rgba(255,64,64,.9);box-sizing:border-box;line-height:1}"
        ".ocr-box span{position:absolute;left:-2px;top:-18px;background:#ff4040;color:white;font-size:12px;padding:1px 4px}"
        ".rebuild{border:1px solid #c8d0da;max-width:100%;height:auto}"
        "table{border-collapse:collapse;margin-top:18px;width:100%;max-width:1200px}"
        "td,th{border:1px solid #ccd3dc;padding:5px 8px;font-size:13px;vertical-align:top}"
        "th{background:#f2f5f8;text-align:left}"
        "@media(max-width:900px){.views{grid-template-columns:1fr}}"
        "</style></head><body>"
        f"<h1>{html.escape(Path(str(result.get('image') or '')).name)} OCR</h1>"
        "<div class='views'>"
        "<div class='panel'><h2>OCR Boxes On Source</h2>"
        f"<div class='stage' style='width:{width}px;height:{height}px'>"
        f"<img src='{image_src}' width='{width}' height='{height}'>"
        + "".join(boxes)
        + "</div></div>"
        "<div class='panel'><h2>OCR Text Rebuild</h2>"
        f"<img class='rebuild' src='{rebuild_src}' width='{width}' height='{height}'>"
        f"<p><a href='{debug_src}'>Open OCR debug image</a></p>"
        "</div></div>"
        "<table><tr><th>#</th><th>text</th><th>score</th><th>bbox</th></tr>"
        + "\n".join(rows)
        + "</table></body></html>"
    )


def collect_ocr_objects(result: dict) -> list[dict]:
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    out: list[dict] = []
    for obj in result.get("ocr_objects") or []:
        text = str(obj.get("text") or "").strip()
        if not text:
            continue
        rect = Rect(*obj.get("bbox", [0, 0, 0, 0])).to_int()
        key = (text, tuple(rect))
        if key in seen:
            continue
        seen.add(key)
        out.append(obj)
    for cell in result.get("cells") or []:
        for obj in cell.get("ocr_objects") or []:
            text = str(obj.get("text") or "").strip()
            if not text:
                continue
            rect = Rect(*obj.get("bbox", [0, 0, 0, 0])).to_int()
            key = (text, tuple(rect))
            if key in seen:
                continue
            seen.add(key)
            out.append(obj)
    return sorted(out, key=lambda item: (Rect(*item["bbox"]).y0, Rect(*item["bbox"]).x0))


def load_cjk_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def ocr_palette(idx: int) -> tuple[int, int, int]:
    colors = [
        (219, 77, 109),
        (51, 155, 214),
        (66, 171, 100),
        (222, 159, 54),
        (143, 105, 204),
        (43, 169, 166),
    ]
    return colors[(idx - 1) % len(colors)]


def load_table_results(path: str | Path, method: str | None = None) -> dict[str, dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    if method:
        return raw.get(method, {}) if isinstance(raw.get(method), dict) else {}
    if all(isinstance(v, dict) and ("html" in v or "bbox" in v or "status" in v) for v in raw.values()):
        return raw
    preferred = [
        "PaddleOCR TableRecognitionPipelineV2",
        "RapidAI TableStructureRec v2",
        "SLANet_plus",
        "Docling TableFormer fast",
    ]
    for name in preferred:
        if isinstance(raw.get(name), dict):
            return raw[name]
    for value in raw.values():
        if isinstance(value, dict):
            return value
    return {}
