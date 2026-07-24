"""Create a deterministic, page-constrained reflow plan."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median

from bs4 import BeautifulSoup

from docflow.model.stages import (
    DocumentAnalysis,
    FlowKind,
    FlowSection,
    GridCell,
    PageGeometry,
    PlannedElement,
    Rect,
    ReflowLayoutPlan,
    ReflowPagePlan,
)


class ReflowPlanner:
    def __init__(self, page_long_edge_pt: float = 841.89, word_safety_factor: float = 0.80) -> None:
        self.page_long_edge_pt = float(page_long_edge_pt)
        self.word_safety_factor = float(word_safety_factor)

    def plan(self, analysis: DocumentAnalysis) -> ReflowLayoutPlan:
        role_by_id = {role.role_id: role for role in analysis.roles}
        role_counts = defaultdict(int)
        for page in analysis.pages:
            for element in page.elements:
                if element.role_id and element.role_id.startswith("body_"):
                    role_counts[element.role_id] += 1
        default_body_role = max(role_counts, key=role_counts.get, default=None)
        pages = tuple(self._plan_page(page, role_by_id, default_body_role) for page in analysis.pages)
        return ReflowLayoutPlan(
            pages,
            analysis.roles,
            source_file=analysis.source_file,
            word_safety_factor=self.word_safety_factor,
        )

    def _plan_page(self, page, role_by_id, default_body_role) -> ReflowPagePlan:
        scale = self.page_long_edge_pt / max(page.width_px, page.height_px)
        width_pt = page.width_px * scale
        height_pt = page.height_px * scale
        furniture = [element for element in page.elements if element.kind in {"header", "footer", "page_number"}]
        body = [element for element in page.elements if element not in furniture]
        bounds = self._union(element.bbox for element in body) if body else Rect(0, 0, page.width_px, page.height_px)
        geometry = PageGeometry(
            width_pt,
            height_pt,
            bounds.y1 * scale,
            (page.width_px - bounds.x2) * scale,
            (page.height_px - bounds.y2) * scale,
            bounds.x1 * scale,
        )
        usable_width = geometry.width_pt - geometry.margin_left_pt - geometry.margin_right_pt
        sections, placement = self._build_sections(body, bounds, usable_width)
        container_widths = self._container_widths(body, sections, placement, bounds.width)
        planned = tuple(
            PlannedElement(
                element.element_id,
                element.kind,
                role_id=element.role_id or default_body_role,
                text=element.text,
                payload={
                    **dict(element.payload),
                    "source_bbox": (element.bbox.x1, element.bbox.y1, element.bbox.x2, element.bbox.y2),
                    "width_fraction": min(
                        self._visual_width(element) / container_widths.get(element.element_id, max(bounds.width, 1.0)),
                        1.0,
                    ),
                    "column": placement.get(element.element_id, 0),
                    "alignment": self._alignment(element, bounds),
                    "first_line_indent_pt": self._first_line_indent(element, scale),
                    "page_width_px": page.width_px,
                    "table_font_size_pt": self._table_font_size(element, scale),
                },
            )
            for element in page.elements
        )
        estimated_height = sum(
            max(
                self._section_height(section, planned, role_by_id, usable_width),
                self._source_section_height(section, planned, scale),
            )
            for section in sections
        )
        usable_height = geometry.height_pt - geometry.margin_top_pt - geometry.margin_bottom_pt
        fit_scale = min(1.0, usable_height * self.word_safety_factor / max(estimated_height, 1.0))
        header_ids = tuple(
            element.element_id
            for element in furniture
            if element.kind == "header" or (element.kind == "page_number" and (element.bbox.y1 + element.bbox.y2) / 2 < page.height_px / 2)
        )
        footer_ids = tuple(element.element_id for element in furniture if element.element_id not in header_ids)
        return ReflowPagePlan(
            page.page_index,
            geometry,
            planned,
            sections,
            fit_scale,
            header_ids,
            footer_ids,
        )

    def _build_sections(self, elements, bounds: Rect, usable_width: float):
        sections = []
        placement = {}
        run = []
        anchor_lanes = self._anchor_lanes(elements, bounds.width)
        lane_bounds = [
            (min(item.bbox.x1 for item in lane), max(item.bbox.x2 for item in lane))
            for lane in anchor_lanes
        ]

        def flush() -> None:
            if not run:
                return
            section, columns = self._narrow_section(run, bounds, usable_width, len(sections))
            sections.append(section)
            placement.update(columns)
            run.clear()

        for element in elements:
            overlap_count = sum(
                1
                for left, right in lane_bounds
                if max(0.0, min(element.bbox.x2, right) - max(element.bbox.x1, left))
                / max(right - left, 1.0)
                >= 0.30
            )
            is_spanning = overlap_count >= 2 if len(lane_bounds) >= 2 else element.bbox.width / max(bounds.width, 1.0) >= 0.72
            if is_spanning:
                flush()
                section_id = f"section_{len(sections)}"
                sections.append(FlowSection(section_id, FlowKind.SINGLE, (element.element_id,)))
                placement[element.element_id] = 0
            else:
                run.append(element)
        flush()
        return tuple(sections), placement

    @staticmethod
    def _container_widths(elements, sections, placement, body_width: float):
        by_id = {element.element_id: element for element in elements}
        widths = {element.element_id: max(body_width, 1.0) for element in elements}
        for section in sections:
            if section.kind == FlowKind.SINGLE:
                continue
            by_column = defaultdict(list)
            for identifier in section.element_ids:
                by_column[placement[identifier]].append(by_id[identifier])
            for members in by_column.values():
                lane_width = max(item.bbox.x2 for item in members) - min(item.bbox.x1 for item in members)
                for item in members:
                    widths[item.element_id] = max(lane_width, 1.0)
        return widths

    @staticmethod
    def _visual_width(element) -> float:
        bbox = element.payload.get("primary_bbox")
        return max(float(bbox[2]) - float(bbox[0]), 1.0) if bbox else element.bbox.width

    def _narrow_section(self, elements, bounds: Rect, usable_width: float, section_index: int):
        lanes = self._lanes(elements, bounds.width)
        section_id = f"section_{section_index}"
        if len(lanes) < 2:
            return FlowSection(section_id, FlowKind.SINGLE, tuple(item.element_id for item in elements)), {
                item.element_id: 0 for item in elements
            }

        lane_by_id = {
            item.element_id: lane_index
            for lane_index, lane in enumerate(lanes)
            for item in lane
        }
        lane_sequence = [lane_by_id[item.element_id] for item in elements]
        collapsed = [lane_sequence[0]]
        for lane in lane_sequence[1:]:
            if lane != collapsed[-1]:
                collapsed.append(lane)
        widths = [max(item.bbox.x2 for item in lane) - min(item.bbox.x1 for item in lane) for lane in lanes]
        lane_bounds = [(min(item.bbox.x1 for item in lane), max(item.bbox.x2 for item in lane)) for lane in lanes]
        gaps = [max(lane_bounds[index + 1][0] - lane_bounds[index][1], 0.0) for index in range(len(lanes) - 1)]
        gutter = (median(gaps) / max(bounds.width, 1.0) * usable_width) if gaps else 0.0
        available = max(usable_width - gutter * (len(lanes) - 1), 1.0)
        column_widths = tuple(width / max(sum(widths), 1.0) * available for width in widths)
        element_ids = tuple(item.element_id for item in elements)

        if collapsed == sorted(set(collapsed)) and len(collapsed) == len(lanes):
            return FlowSection(
                section_id,
                FlowKind.SEQUENTIAL_COLUMNS,
                element_ids,
                column_widths_pt=column_widths,
                gutter_pt=gutter,
            ), lane_by_id

        rows = self._grid_rows(elements)
        cells = defaultdict(list)
        for item in elements:
            cells[(rows[item.element_id], lane_by_id[item.element_id])].append(item.element_id)
        grid_cells = tuple(
            GridCell(row, column, tuple(ids))
            for (row, column), ids in sorted(cells.items())
        )
        return FlowSection(
            section_id,
            FlowKind.GRID,
            element_ids,
            column_widths_pt=column_widths,
            gutter_pt=gutter,
            grid_cells=grid_cells,
        ), lane_by_id

    @staticmethod
    def _anchor_lanes(elements, page_width: float):
        anchors = [
            item
            for item in elements
            if item.kind == "paragraph_group"
            and item.bbox.width >= page_width * 0.20
        ]
        if len(anchors) < 2:
            return []
        lanes = []
        for item in sorted(anchors, key=lambda element: (element.bbox.x1, element.bbox.x2)):
            best = None
            best_overlap = 0.0
            for index, lane in enumerate(lanes):
                left = min(member.bbox.x1 for member in lane)
                right = max(member.bbox.x2 for member in lane)
                overlap = max(0.0, min(right, item.bbox.x2) - max(left, item.bbox.x1))
                ratio = overlap / max(min(right - left, item.bbox.width), 1.0)
                if ratio > best_overlap:
                    best, best_overlap = index, ratio
            if best is not None and best_overlap >= 0.35:
                lanes[best].append(item)
            else:
                lanes.append([item])
        lanes.sort(key=lambda lane: min(item.bbox.x1 for item in lane))
        return lanes if 2 <= len(lanes) <= 4 else []

    @classmethod
    def _lanes(cls, elements, page_width: float):
        lanes = cls._anchor_lanes(elements, page_width)
        if not lanes:
            return [list(elements)]
        anchored = {item.element_id for lane in lanes for item in lane}
        for item in elements:
            if item.element_id in anchored:
                continue
            center = (item.bbox.x1 + item.bbox.x2) / 2.0
            lane = min(
                lanes,
                key=lambda members: abs(
                    center
                    - median([(member.bbox.x1 + member.bbox.x2) / 2.0 for member in members])
                ),
            )
            lane.append(item)
        lanes.sort(key=lambda lane: min(item.bbox.x1 for item in lane))
        return lanes

    @staticmethod
    def _grid_rows(elements):
        rows = []
        result = {}
        for item in sorted(elements, key=lambda element: (element.bbox.y1, element.bbox.x1)):
            row = next(
                (index for index, bottom in enumerate(rows) if item.bbox.y1 <= bottom),
                None,
            )
            if row is None:
                row = len(rows)
                rows.append(item.bbox.y2)
            else:
                rows[row] = max(rows[row], item.bbox.y2)
            result[item.element_id] = row
        return result

    def _section_height(self, section, elements, roles, usable_width: float) -> float:
        by_id = {element.element_id: element for element in elements}
        if section.kind == FlowKind.SINGLE:
            return sum(self._element_height(by_id[item], roles, usable_width) for item in section.element_ids)
        widths = section.column_widths_pt
        if section.kind == FlowKind.SEQUENTIAL_COLUMNS:
            totals = [0.0] * len(widths)
            for identifier in section.element_ids:
                element = by_id[identifier]
                column = int(element.payload.get("column", 0))
                totals[column] += self._element_height(element, roles, widths[column])
            return max(totals, default=0.0)
        row_heights = defaultdict(float)
        for cell in section.grid_cells:
            width = widths[cell.column]
            height = sum(self._element_height(by_id[identifier], roles, width) for identifier in cell.element_ids)
            row_heights[cell.row] = max(row_heights[cell.row], height)
        return sum(row_heights.values())

    @staticmethod
    def _source_section_height(section, elements, source_scale: float) -> float:
        by_id = {element.element_id: element for element in elements}
        boxes = [by_id[identifier].payload.get("source_bbox") for identifier in section.element_ids]
        boxes = [box for box in boxes if box]
        if not boxes:
            return 0.0
        return (max(float(box[3]) for box in boxes) - min(float(box[1]) for box in boxes)) * source_scale

    @staticmethod
    def _element_height(element, roles, width: float) -> float:
        role = roles.get(element.role_id)
        if role is not None and element.text:
            units = sum(1.0 if ord(char) >= 0x2E80 else 0.52 for char in element.text)
            lines = max(1, math.ceil(units * role.font_size_pt / max(width, 1.0)))
            return lines * role.font_size_pt * role.line_spacing + role.space_before_pt + role.space_after_pt
        bbox = element.payload.get("primary_bbox") or element.payload.get("source_bbox") or (0, 0, 1, 1)
        aspect = max(float(bbox[3]) - float(bbox[1]), 1.0) / max(float(bbox[2]) - float(bbox[0]), 1.0)
        visual_width = width * float(element.payload.get("width_fraction", 1.0))
        caption_lines = 1 if element.payload.get("caption") else 0
        return visual_width * aspect + caption_lines * 10.0

    @staticmethod
    def _alignment(element, bounds: Rect) -> str:
        if element.kind in {"figure_group", "equation_group", "caption"}:
            return "center"
        if element.kind == "heading":
            element_center = (element.bbox.x1 + element.bbox.x2) / 2.0
            bounds_center = (bounds.x1 + bounds.x2) / 2.0
            if abs(element_center - bounds_center) <= bounds.width * 0.08:
                return "center"
        return "justify" if element.kind == "paragraph_group" and len(element.text) >= 40 else "left"

    @staticmethod
    def _first_line_indent(element, source_scale: float) -> float:
        lefts = element.payload.get("line_lefts_px") or ()
        if len(lefts) < 2:
            return 0.0
        indent = float(lefts[0]) - median(float(value) for value in lefts[1:])
        return max(indent * source_scale, 0.0)

    @staticmethod
    def _table_font_size(element, source_scale: float):
        if element.kind != "table_group" or not element.payload.get("html"):
            return None
        table = BeautifulSoup(str(element.payload["html"]), "html.parser").find("table")
        row_count = len(table.find_all("tr")) if table else 0
        if row_count == 0:
            return None
        return element.bbox.height * source_scale / row_count / 1.45

    @staticmethod
    def _union(rectangles):
        rects = tuple(rectangles)
        return Rect(
            min(rect.x1 for rect in rects),
            min(rect.y1 for rect in rects),
            max(rect.x2 for rect in rects),
            max(rect.y2 for rect in rects),
        )
