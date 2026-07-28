"""Create a deterministic, page-constrained reflow plan."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import replace
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
from docflow.renderer.docx_utils.html_table import estimate_text_units, get_table_cell_placements, get_table_column_weights


_CROSS_ENGINE_PAGE_RESERVE_PT = 12.0


class ReflowPlanner:
    def __init__(self, page_long_edge_pt: float = 841.89, word_safety_factor: float = 0.96) -> None:
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
        edge_candidates = [
            element
            for element in page.elements
            if element.kind == "figure_group"
            and element.bbox.width <= page.width_px * 0.08
            and element.bbox.height >= element.bbox.width * 2.5
            and (element.bbox.x2 <= page.width_px * 0.12 or element.bbox.x1 >= page.width_px * 0.88)
        ]
        edge_visuals = {edge_candidates[0].element_id} if len(edge_candidates) == 1 else set()
        furniture = [
            element
            for element in page.elements
            if element.kind in {"header", "footer", "page_number"} or element.element_id in edge_visuals
        ]
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
        sections, placement = self._promote_margin_notes(sections, body, placement, bounds, scale)
        sections, placement = self._promote_wrapped_media(sections, body, placement, scale)
        sections = tuple(self._with_vertical_tracks(section, body, scale) for section in sections)
        container_frames = self._container_frames(body, sections, placement, bounds)
        container_widths = {identifier: max(right - left, 1.0) for identifier, (left, right) in container_frames.items()}
        horizontal_indents = self._horizontal_indents(body, sections, placement, container_frames, usable_width)
        vertical_spacing = self._vertical_spacing(body, sections, placement, scale)
        alignments = {
            element.element_id: self._alignment(element, container_frames.get(element.element_id, (bounds.x1, bounds.x2)))
            for element in page.elements
        }
        hanging_indents = {
            element.element_id: self._heading_hanging_indent(element, scale)
            if alignments[element.element_id] == "left"
            else 0.0
            for element in page.elements
        }
        body_font_size = role_by_id[default_body_role].font_size_pt if default_body_role in role_by_id else 10.5
        table_font_sizes = {
            element.element_id: self._table_font_size(element, scale, body_font_size)
            for element in page.elements
        }
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
                    "alignment": alignments[element.element_id],
                    "first_line_indent_pt": self._first_line_indent(element, scale) - hanging_indents[element.element_id],
                    "left_indent_pt": horizontal_indents.get(
                        element.element_id,
                        (max(element.bbox.x1 * scale - geometry.margin_left_pt, 0.0), 0.0),
                    )[0] + hanging_indents[element.element_id],
                    "right_indent_pt": horizontal_indents.get(
                        element.element_id,
                        (0.0, max((page.width_px - element.bbox.x2) * scale - geometry.margin_right_pt, 0.0)),
                    )[1],
                    "space_before_pt": vertical_spacing.get(element.element_id, 0.0),
                    "line_height_pt": self._source_line_height(
                        element,
                        role_by_id.get(element.role_id or default_body_role),
                        scale,
                    ),
                    "source_scale": scale,
                    "page_width_px": page.width_px,
                    "page_height_px": page.height_px,
                    "table_font_size_pt": table_font_sizes[element.element_id],
                    "caption_font_size_pt": self._caption_font_size(element, scale, body_font_size),
                    "table_height_pt": (
                        self._layout_bbox(element).height * scale
                        if element.kind == "table_group" and element.payload.get("html")
                        else None
                    ),
                    "table_min_font_size_pt": (
                        min(table_font_sizes[element.element_id], 6.5)
                        if table_font_sizes[element.element_id] is not None
                        else None
                    ),
                },
            )
            for element in page.elements
        )
        usable_height = geometry.height_pt - geometry.margin_top_pt - geometry.margin_bottom_pt
        fit_scale = self._fit_scale(sections, planned, role_by_id, usable_width, usable_height)
        header_ids = tuple(
            element.element_id
            for element in furniture
            if element.kind == "header"
            or (element.kind == "page_number" and (element.bbox.y1 + element.bbox.y2) / 2 < page.height_px / 2)
            or (element.element_id in edge_visuals and (element.bbox.y1 + element.bbox.y2) / 2 < page.height_px / 2)
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
        anchor_lanes = self._anchor_lanes(elements, bounds.width) or self._local_visual_lanes(elements, bounds.width)
        lane_bounds = [
            (median(self._layout_bbox(item).x1 for item in lane), median(self._layout_bbox(item).x2 for item in lane))
            for lane in anchor_lanes
        ]
        spanning = []
        for element in elements:
            layout_bbox = self._layout_bbox(element)
            overlap_count = sum(
                1
                for left, right in lane_bounds
                if max(0.0, min(layout_bbox.x2, right) - max(layout_bbox.x1, left))
                / max(right - left, 1.0)
                >= 0.20
            )
            lane_center = (lane_bounds[0][0] + lane_bounds[-1][1]) / 2.0 if lane_bounds else 0.0
            centered_heading = element.kind == "heading" and abs(
                (layout_bbox.x1 + layout_bbox.x2) / 2.0 - lane_center
            ) <= bounds.width * 0.05
            is_spanning = len(lane_bounds) >= 2 and (overlap_count == len(lane_bounds) or centered_heading)
            if is_spanning:
                spanning.append(element)

        spanning.sort(key=lambda item: ((item.bbox.y1 + item.bbox.y2) / 2.0, item.bbox.x1))
        cuts = [(item.bbox.y1 + item.bbox.y2) / 2.0 for item in spanning]
        bands = [[] for _ in range(len(spanning) + 1)]
        spanning_ids = {item.element_id for item in spanning}
        for element in elements:
            if element.element_id not in spanning_ids:
                center = (element.bbox.y1 + element.bbox.y2) / 2.0
                bands[bisect_right(cuts, center)].append(element)

        for index, band in enumerate(bands):
            if band:
                section, columns = self._narrow_section(
                    band,
                    bounds,
                    usable_width,
                    len(sections),
                    anchor_lanes,
                )
                sections.append(section)
                placement.update(columns)
            if index < len(spanning):
                element = spanning[index]
                section_id = f"section_{len(sections)}"
                sections.append(FlowSection(section_id, FlowKind.SINGLE, (element.element_id,)))
                placement[element.element_id] = 0
        return tuple(sections), placement

    @staticmethod
    def _promote_margin_notes(sections, elements, placement, bounds: Rect, source_scale: float):
        by_id = {element.element_id: element for element in elements}
        main_text = [
            element
            for element in elements
            if element.kind == "paragraph_group" and element.bbox.width >= bounds.width * 0.55
        ]
        if len(main_text) < 2:
            return sections, placement
        main_left = median(element.bbox.x1 for element in main_text)
        main_right = median(element.bbox.x2 for element in main_text)
        main_width = max(main_right - main_left, 1.0)
        section_by_id = {
            identifier: index
            for index, section in enumerate(sections)
            for identifier in section.element_ids
        }
        replacements = {}
        removed = set()
        updated_placement = dict(placement)
        for note in elements:
            if (
                note.kind != "paragraph_group"
                or note.bbox.width > main_width * 0.20
                or not (note.bbox.x2 <= main_left + main_width * 0.03 or note.bbox.x1 >= main_right - main_width * 0.03)
            ):
                continue
            note_section_index = section_by_id[note.element_id]
            note_section = sections[note_section_index]
            if note_section.kind != FlowKind.SINGLE:
                continue
            anchors = [
                element
                for element in main_text
                if max(0.0, min(note.bbox.y2, element.bbox.y2) - max(note.bbox.y1, element.bbox.y1))
                / max(min(note.bbox.height, element.bbox.height), 1.0)
                >= 0.20
                and sections[section_by_id[element.element_id]].kind == FlowKind.SINGLE
            ]
            if not anchors:
                continue
            anchor = max(anchors, key=lambda element: min(note.bbox.y2, element.bbox.y2) - max(note.bbox.y1, element.bbox.y1))
            anchor_section_index = section_by_id[anchor.element_id]
            if anchor_section_index in replacements:
                continue
            anchor_section = sections[anchor_section_index]
            side = "left" if note.bbox.x2 <= main_left + main_width * 0.03 else "right"
            element_ids = (
                anchor_section.element_ids
                if note_section_index == anchor_section_index
                else anchor_section.element_ids + (note.element_id,)
            )
            replacements[anchor_section_index] = FlowSection(
                anchor_section.section_id,
                FlowKind.WRAPPED,
                element_ids,
                gutter_pt=max((main_left - note.bbox.x2 if side == "left" else note.bbox.x1 - main_right) * source_scale, 0.0),
                floating_element_id=note.element_id,
                floating_width_pt=note.bbox.width * source_scale,
                floating_side=side,
                floating_offset_x_pt=max(note.bbox.x1 - bounds.x1, 0.0) * source_scale,
                floating_offset_y_pt=max(
                    note.bbox.y1 - min(by_id[identifier].bbox.y1 for identifier in element_ids if identifier != note.element_id),
                    0.0,
                ) * source_scale,
            )
            updated_placement[note.element_id] = 0
            if note_section_index != anchor_section_index:
                removed.add(note_section_index)
        promoted = tuple(
            replacements.get(index, section)
            for index, section in enumerate(sections)
            if index not in removed
        )
        return promoted, updated_placement

    @staticmethod
    def _promote_wrapped_media(sections, elements, placement, source_scale: float):
        by_id = {element.element_id: element for element in elements}
        promoted = []
        updated_placement = dict(placement)
        index = 0
        while index < len(sections):
            section = sections[index]
            figures = [
                by_id[identifier]
                for identifier in section.element_ids
                if by_id[identifier].kind == "figure_group"
            ]
            text = [
                by_id[identifier]
                for identifier in section.element_ids
                if by_id[identifier].kind == "paragraph_group"
            ]
            if (
                section.kind not in {FlowKind.SEQUENTIAL_COLUMNS, FlowKind.GRID}
                or len(figures) != 1
                or len(text) < 2
                or len(figures) + len(text) != len(section.element_ids)
            ):
                promoted.append(section)
                index += 1
                continue
            figure = figures[0]
            figure_column = placement[figure.element_id]
            side = "left" if figure_column == 0 else "right"
            if figure_column not in {0, len(section.column_widths_pt) - 1}:
                promoted.append(section)
                index += 1
                continue
            element_ids = list(section.element_ids)
            next_index = index + 1
            while next_index < len(sections):
                following = sections[next_index]
                if following.kind != FlowKind.SINGLE or any(
                    by_id[identifier].kind != "paragraph_group" for identifier in following.element_ids
                ):
                    break
                element_ids.extend(following.element_ids)
                next_index += 1
            if not any(
                by_id[identifier].kind == "paragraph_group"
                and by_id[identifier].bbox.y1 >= figure.bbox.y2
                for identifier in element_ids
            ):
                promoted.append(section)
                index += 1
                continue
            text_top = min(
                by_id[identifier].bbox.y1
                for identifier in element_ids
                if identifier != figure.element_id
            )
            promoted.append(
                FlowSection(
                    section.section_id,
                    FlowKind.WRAPPED,
                    tuple(element_ids),
                    gutter_pt=section.gutter_pt,
                    floating_element_id=figure.element_id,
                    floating_width_pt=section.column_widths_pt[figure_column],
                    floating_side=side,
                    floating_offset_x_pt=(figure.bbox.x1 * source_scale if side == "left" else 0.0),
                    floating_offset_y_pt=max(figure.bbox.y1 - text_top, 0.0) * source_scale,
                )
            )
            updated_placement.update((identifier, 0) for identifier in element_ids)
            index = next_index
        return tuple(promoted), updated_placement

    @staticmethod
    def _container_frames(elements, sections, placement, bounds: Rect):
        by_id = {element.element_id: element for element in elements}
        frames = {element.element_id: (bounds.x1, bounds.x2) for element in elements}
        for section in sections:
            if section.kind in {FlowKind.SINGLE, FlowKind.WRAPPED}:
                continue
            spans = {
                identifier: cell.column_span
                for cell in section.grid_cells
                for identifier in cell.element_ids
            }
            by_column = defaultdict(list)
            for identifier in section.element_ids:
                if spans.get(identifier, 1) == 1:
                    by_column[placement[identifier]].append(by_id[identifier])
            column_frames = {}
            for column, members in by_column.items():
                left = min(ReflowPlanner._layout_bbox(item).x1 for item in members)
                right = max(ReflowPlanner._layout_bbox(item).x2 for item in members)
                column_frames[column] = (left, right)
                for item in members:
                    frames[item.element_id] = (left, right)
            for identifier, span in spans.items():
                if span == 1:
                    continue
                column = placement[identifier]
                covered = [column_frames[index] for index in range(column, column + span) if index in column_frames]
                if covered:
                    frames[identifier] = (min(frame[0] for frame in covered), max(frame[1] for frame in covered))
        return frames

    @staticmethod
    def _horizontal_indents(elements, sections, placement, frames, usable_width: float):
        by_id = {element.element_id: element for element in elements}
        target_widths = {element.element_id: usable_width for element in elements}
        section_kinds = {}
        wrapped_figure_ids = set()
        for section in sections:
            section_kinds.update((identifier, section.kind) for identifier in section.element_ids)
            if section.kind == FlowKind.WRAPPED and by_id[section.floating_element_id].kind == "figure_group":
                wrapped_figure_ids.update(section.element_ids)
            if section.kind in {FlowKind.SINGLE, FlowKind.WRAPPED}:
                continue
            spans = {
                identifier: cell.column_span
                for cell in section.grid_cells
                for identifier in cell.element_ids
            }
            for identifier in section.element_ids:
                column = placement[identifier]
                span = spans.get(identifier, 1)
                target_widths[identifier] = (
                    sum(section.column_widths_pt[column : column + span]) + section.gutter_pt * (span - 1)
                )
        result = {}
        for element in elements:
            left, right = frames[element.element_id]
            source_width = max(right - left, 1.0)
            source_lines = element.payload.get("lines") or ()
            centered = abs((element.bbox.x1 + element.bbox.x2 - left - right) / 2.0) <= source_width * 0.04
            width_fraction = element.bbox.width / source_width
            section_kind = section_kinds[element.element_id]
            single_flow_uses_full_width = element.element_id in wrapped_figure_ids or (
                section_kind == FlowKind.SINGLE and (element.kind == "heading" or width_fraction > 0.85)
            )
            if element.kind == "heading":
                result[element.element_id] = (
                    0.0
                    if centered or width_fraction > 0.85
                    else max(element.bbox.x1 - left, 0.0) / source_width * target_widths[element.element_id],
                    0.0,
                )
                continue
            if element.kind != "paragraph_group" or width_fraction < 0.3 or single_flow_uses_full_width:
                result[element.element_id] = (0.0, 0.0)
                continue
            target_width = target_widths[element.element_id]
            result[element.element_id] = (
                max(element.bbox.x1 - left, 0.0) / source_width * target_width,
                max(right - element.bbox.x2, 0.0) / source_width * target_width,
            )
        return result

    @staticmethod
    def _visual_width(element) -> float:
        return ReflowPlanner._layout_bbox(element).width

    @staticmethod
    def _layout_bbox(element) -> Rect:
        bbox = element.payload.get("primary_bbox")
        return Rect.from_sequence(bbox) if bbox else element.bbox

    def _narrow_section(self, elements, bounds: Rect, usable_width: float, section_index: int, fallback_lanes=()):
        if len(elements) == 1:
            section_id = f"section_{section_index}"
            return FlowSection(section_id, FlowKind.SINGLE, (elements[0].element_id,)), {elements[0].element_id: 0}
        lanes = self._anchor_lanes(elements, bounds.width) or list(fallback_lanes)
        for rail in self._side_rail_lanes(elements, bounds):
            rail_center = median((item.bbox.x1 + item.bbox.x2) / 2.0 for item in rail)
            if all(
                abs(rail_center - median((item.bbox.x1 + item.bbox.x2) / 2.0 for item in lane)) > bounds.width * 0.08
                for lane in lanes
            ):
                lanes.append(rail)
        lanes.sort(key=lambda lane: median((item.bbox.x1 + item.bbox.x2) / 2.0 for item in lane))
        section_id = f"section_{section_index}"
        if len(lanes) < 2:
            narrow = [item for item in elements if self._layout_bbox(item).width < bounds.width * 0.60]
            ordered = [item for item in elements if item not in narrow]
            for item in sorted(narrow, key=lambda value: (self._layout_bbox(value).y1, self._layout_bbox(value).x1)):
                top = self._layout_bbox(item).y1
                index = next(
                    (position for position, existing in enumerate(ordered) if self._layout_bbox(existing).y1 > top),
                    len(ordered),
                )
                ordered.insert(index, item)
            return FlowSection(section_id, FlowKind.SINGLE, tuple(item.element_id for item in ordered)), {
                item.element_id: 0 for item in elements
            }

        lane_bounds = [
            (median(self._layout_bbox(item).x1 for item in lane), median(self._layout_bbox(item).x2 for item in lane))
            for lane in lanes
        ]
        placement = {}
        for item in elements:
            layout_bbox = self._layout_bbox(item)
            overlaps = [
                index
                for index, (left, right) in enumerate(lane_bounds)
                if max(0.0, min(layout_bbox.x2, right) - max(layout_bbox.x1, left))
                / max(min(layout_bbox.width, right - left), 1.0)
                >= 0.30
            ]
            if overlaps:
                placement[item.element_id] = min(overlaps)
            else:
                center = (layout_bbox.x1 + layout_bbox.x2) / 2.0
                column = min(
                    range(len(lane_bounds)),
                    key=lambda index: abs(center - sum(lane_bounds[index]) / 2.0),
                )
                placement[item.element_id] = column

        if len(set(placement.values())) < 2:
            return FlowSection(section_id, FlowKind.SINGLE, tuple(item.element_id for item in elements)), {
                item.element_id: 0 for item in elements
            }

        spans = {}
        for item in elements:
            bbox = self._layout_bbox(item)
            overlaps = [
                index
                for index, (left, right) in enumerate(lane_bounds)
                if max(0.0, min(bbox.x2, right) - max(bbox.x1, left)) / max(right - left, 1.0) >= 0.30
            ]
            spans[item.element_id] = max(overlaps) - min(overlaps) + 1 if overlaps else 1
        lane_members = [
            [item for item in elements if placement[item.element_id] == index and spans[item.element_id] == 1]
            for index in range(len(lane_bounds))
        ]
        lane_bounds = [
            (
                min(self._layout_bbox(item).x1 for item in members),
                max(self._layout_bbox(item).x2 for item in members),
            )
            if members
            else lane_bounds[index]
            for index, members in enumerate(lane_members)
        ]

        lane_sequence = [placement[item.element_id] for item in elements]
        collapsed = [lane_sequence[0]]
        for lane in lane_sequence[1:]:
            if lane != collapsed[-1]:
                collapsed.append(lane)
        widths = [right - left for left, right in lane_bounds]
        gaps = [max(lane_bounds[index + 1][0] - lane_bounds[index][1], 0.0) for index in range(len(lanes) - 1)]
        gutter = (median(gaps) / max(bounds.width, 1.0) * usable_width) if gaps else 0.0
        available = max(usable_width - gutter * (len(lanes) - 1), 1.0)
        column_widths = tuple(width / max(sum(widths), 1.0) * available for width in widths)
        element_ids = tuple(item.element_id for item in elements)

        balanced_columns = max(widths) <= min(widths) * 1.5
        if max(spans.values(), default=1) == 1 and collapsed == sorted(set(collapsed)) and len(collapsed) == len(lanes) and balanced_columns:
            return FlowSection(
                section_id,
                FlowKind.SEQUENTIAL_COLUMNS,
                element_ids,
                column_widths_pt=column_widths,
                gutter_pt=gutter,
            ), placement

        rows = (
            self._spanning_grid_rows(elements, placement, spans)
            if max(spans.values(), default=1) > 1
            else self._grid_rows(elements)
        )
        cells = defaultdict(list)
        for item in elements:
            cells[(rows[item.element_id], placement[item.element_id], spans[item.element_id])].append(item.element_id)
        grid_cells = [
            GridCell(row, column, tuple(ids), column_span=column_span)
            for (row, column, column_span), ids in sorted(cells.items())
        ]
        for anchor_row in sorted({cell.row for cell in grid_cells if cell.column_span > 1}):
            covered = {
                column
                for cell in grid_cells
                if cell.row == anchor_row and cell.column_span > 1
                for column in range(cell.column, cell.column + cell.column_span)
            }
            for lower in tuple(
                cell
                for cell in grid_cells
                if cell.row == anchor_row and cell.column_span == 1 and cell.column not in covered
            ):
                upper = next(
                    (
                        cell
                        for cell in grid_cells
                        if cell.column == lower.column
                        and cell.column_span == 1
                        and cell.row + cell.row_span == anchor_row
                    ),
                    None,
                )
                if upper is not None:
                    grid_cells.remove(upper)
                    grid_cells.remove(lower)
                    grid_cells.append(
                        GridCell(
                            upper.row,
                            upper.column,
                            upper.element_ids + lower.element_ids,
                            row_span=upper.row_span + lower.row_span,
                        )
                    )
        return FlowSection(
            section_id,
            FlowKind.GRID,
            element_ids,
            column_widths_pt=column_widths,
            gutter_pt=gutter,
            grid_cells=tuple(sorted(grid_cells, key=lambda cell: (cell.row, cell.column))),
        ), placement

    @staticmethod
    def _side_rail_lanes(elements, bounds: Rect):
        candidates = [
            item
            for item in elements
            if item.kind == "figure_group"
            and item.bbox.width <= bounds.width * 0.08
            and (
                item.bbox.x2 <= bounds.x1 + bounds.width * 0.12
                or item.bbox.x1 >= bounds.x2 - bounds.width * 0.12
            )
        ]
        groups = []
        for item in sorted(candidates, key=lambda element: (element.bbox.x1 + element.bbox.x2) / 2.0):
            center = (item.bbox.x1 + item.bbox.x2) / 2.0
            target = next(
                (
                    group
                    for group in groups
                    if abs(center - median((member.bbox.x1 + member.bbox.x2) / 2.0 for member in group)) <= bounds.width * 0.03
                ),
                None,
            )
            if target is None:
                groups.append([item])
            else:
                target.append(item)
        return [
            group
            for group in groups
            if len(group) >= 2 and sum(item.bbox.height for item in group) >= bounds.height * 0.30
        ]

    @staticmethod
    def _anchor_lanes(elements, page_width: float):
        tolerance = page_width * 0.10
        candidate_sets = [
            [
                item
                for item in elements
                if item.kind == "paragraph_group" and page_width * 0.20 <= item.bbox.width <= page_width * 0.60
            ],
            [
                item
                for item in elements
                if item.kind in {"figure_group", "table_group"}
                and ReflowPlanner._layout_bbox(item).width >= page_width * 0.20
            ],
            list(elements),
        ]
        candidate_lanes = []
        for candidate_index, candidates in enumerate(candidate_sets):
            if len(candidates) < 4:
                candidate_lanes.append([])
                continue
            lanes = []
            for item in sorted(
                candidates,
                key=lambda element: (ReflowPlanner._layout_bbox(element).x1 + ReflowPlanner._layout_bbox(element).x2) / 2.0,
            ):
                bbox = ReflowPlanner._layout_bbox(item)
                center = (bbox.x1 + bbox.x2) / 2.0
                best = min(
                    range(len(lanes)),
                    key=lambda index: abs(
                        center
                        - median(
                            (ReflowPlanner._layout_bbox(member).x1 + ReflowPlanner._layout_bbox(member).x2) / 2.0
                            for member in lanes[index]
                        )
                    ),
                    default=None,
                )
                if best is not None and abs(
                    center
                    - median(
                        (ReflowPlanner._layout_bbox(member).x1 + ReflowPlanner._layout_bbox(member).x2) / 2.0
                        for member in lanes[best]
                    )
                ) <= tolerance:
                    lanes[best].append(item)
                else:
                    lanes.append([item])
            lanes = [lane for lane in lanes if len(lane) >= 2]
            lanes.sort(
                key=lambda lane: median(
                    (ReflowPlanner._layout_bbox(item).x1 + ReflowPlanner._layout_bbox(item).x2) / 2.0
                    for item in lane
                )
            )
            candidate_lanes.append(lanes)
            parallel_pairs = ReflowPlanner._parallel_lane_pairs(lanes)
            if candidate_index < 2 and 2 <= len(lanes) <= 4 and parallel_pairs >= 1:
                return lanes
            if candidate_index == 2 and len(lanes) == 2 and parallel_pairs >= 2:
                return lanes

        text_lanes, visual_lanes = candidate_lanes[:2]
        if 2 <= len(text_lanes) <= 4 and len(text_lanes) == len(visual_lanes):
            text_centers = [
                median((ReflowPlanner._layout_bbox(item).x1 + ReflowPlanner._layout_bbox(item).x2) / 2.0 for item in lane)
                for lane in text_lanes
            ]
            visual_centers = [
                median((ReflowPlanner._layout_bbox(item).x1 + ReflowPlanner._layout_bbox(item).x2) / 2.0 for item in lane)
                for lane in visual_lanes
            ]
            if all(abs(text - visual) <= tolerance for text, visual in zip(text_centers, visual_centers)):
                return text_lanes
        return []

    @staticmethod
    def _local_visual_lanes(elements, page_width: float):
        candidates = [
            item
            for item in elements
            if item.kind in {"figure_group", "table_group"}
            and page_width * 0.15 <= ReflowPlanner._layout_bbox(item).width <= page_width * 0.70
        ]
        pairs = []
        for visual in candidates:
            visual_bbox = ReflowPlanner._layout_bbox(visual)
            for partner in elements:
                if partner.element_id == visual.element_id:
                    continue
                partner_bbox = ReflowPlanner._layout_bbox(partner)
                if not page_width * 0.15 <= partner_bbox.width <= page_width * 0.70:
                    continue
                vertical_overlap = max(0.0, min(visual_bbox.y2, partner_bbox.y2) - max(visual_bbox.y1, partner_bbox.y1))
                overlap_ratio = vertical_overlap / max(min(visual_bbox.height, partner_bbox.height), 1.0)
                horizontal_overlap = max(0.0, min(visual_bbox.x2, partner_bbox.x2) - max(visual_bbox.x1, partner_bbox.x1))
                if overlap_ratio < 0.30 or horizontal_overlap > min(visual_bbox.width, partner_bbox.width) * 0.10:
                    continue
                pairs.append((overlap_ratio, visual_bbox.width + partner_bbox.width, visual, partner))
        if not pairs:
            return []
        _overlap, _width, first, second = max(pairs, key=lambda value: (value[0], value[1]))
        return [[first], [second]] if ReflowPlanner._layout_bbox(first).x1 < ReflowPlanner._layout_bbox(second).x1 else [[second], [first]]

    @staticmethod
    def _parallel_lane_pairs(lanes) -> int:
        return sum(
            1
            for left_index, left_lane in enumerate(lanes)
            for right_lane in lanes[left_index + 1 :]
            for left in left_lane
            for right in right_lane
            if max(0.0, min(left.bbox.y2, right.bbox.y2) - max(left.bbox.y1, right.bbox.y1))
            / max(min(left.bbox.height, right.bbox.height), 1.0)
            >= 0.30
        )

    @staticmethod
    def _grid_rows(elements):
        rows = []
        result = {}
        for item in sorted(
            elements,
            key=lambda element: (ReflowPlanner._layout_bbox(element).y1, ReflowPlanner._layout_bbox(element).x1),
        ):
            bbox = ReflowPlanner._layout_bbox(item)
            row = next(
                (index for index, bottom in enumerate(rows) if bbox.y1 <= bottom),
                None,
            )
            if row is None:
                row = len(rows)
                rows.append(bbox.y2)
            else:
                rows[row] = max(rows[row], bbox.y2)
            result[item.element_id] = row
        return result

    @staticmethod
    def _spanning_grid_rows(elements, placement, spans):
        anchor_groups = []
        anchor_group_by_id = {}
        for item in sorted(
            (element for element in elements if spans[element.element_id] > 1),
            key=lambda element: ReflowPlanner._layout_bbox(element).y1,
        ):
            bbox = ReflowPlanner._layout_bbox(item)
            columns = set(range(placement[item.element_id], placement[item.element_id] + spans[item.element_id]))
            if anchor_groups and bbox.y1 <= anchor_groups[-1][1] and columns.isdisjoint(anchor_groups[-1][2]):
                anchor_groups[-1][1] = max(anchor_groups[-1][1], bbox.y2)
                anchor_groups[-1][2].update(columns)
            else:
                anchor_groups.append([bbox.y1, bbox.y2, columns])
            anchor_group_by_id[item.element_id] = len(anchor_groups) - 1

        row_keys = {}
        for item in elements:
            if item.element_id in anchor_group_by_id:
                row_keys[item.element_id] = anchor_group_by_id[item.element_id] * 2 + 1
                continue
            bbox = ReflowPlanner._layout_bbox(item)
            center = (bbox.y1 + bbox.y2) / 2.0
            overlapping = next(
                (
                    index
                    for index, (top, bottom, columns) in enumerate(anchor_groups)
                    if top <= center <= bottom and placement[item.element_id] not in columns
                ),
                None,
            )
            if overlapping is not None:
                row_keys[item.element_id] = overlapping * 2 + 1
            else:
                row_keys[item.element_id] = sum(bottom < center for _top, bottom, _columns in anchor_groups) * 2
        compact_rows = {key: index for index, key in enumerate(sorted(set(row_keys.values())))}
        return {identifier: compact_rows[key] for identifier, key in row_keys.items()}

    @staticmethod
    def _vertical_spacing(elements, sections, placement, source_scale: float):
        by_id = {element.element_id: element for element in elements}
        structural_kinds = {"heading", "caption", "figure_group", "equation_group", "table_group"}
        spacing = {}
        for section in sections:
            previous_by_column = defaultdict(list)
            for identifier in section.element_ids:
                current = by_id[identifier]
                column = placement.get(identifier, 0)
                predecessors = [
                    previous
                    for previous in previous_by_column[column]
                    if previous.bbox.y2 <= current.bbox.y1
                    and max(0.0, min(previous.bbox.x2, current.bbox.x2) - max(previous.bbox.x1, current.bbox.x1))
                    / max(min(previous.bbox.width, current.bbox.width), 1.0)
                    >= 0.30
                ]
                if predecessors:
                    previous = max(predecessors, key=lambda element: element.bbox.y2)
                    if current.kind in structural_kinds or previous.kind in structural_kinds:
                        spacing[identifier] = (current.bbox.y1 - previous.bbox.y2) * source_scale
                previous_by_column[column].append(current)
            if section.kind == FlowKind.GRID and any(cell.column_span > 1 for cell in section.grid_cells):
                row_by_id = {
                    identifier: cell.row
                    for cell in section.grid_cells
                    for identifier in cell.element_ids
                }
                for identifier in set(section.element_ids) & spacing.keys():
                    row = row_by_id[identifier]
                    if any(previous_row < row for previous_row in row_by_id.values()):
                        spacing[identifier] = 0.0
        section_by_id = {
            identifier: section_index
            for section_index, section in enumerate(sections)
            for identifier in section.element_ids
        }
        section_elements = [
            [by_id[identifier] for identifier in section.element_ids]
            for section in sections
        ]
        for current in elements:
            section_index = section_by_id.get(current.element_id, 0)
            if section_index <= 0:
                continue
            section_top = min(element.bbox.y1 for element in section_elements[section_index])
            if current.bbox.y1 > section_top + max(current.bbox.height * 0.10, 2.0):
                continue
            predecessors = [
                previous
                for previous in section_elements[section_index - 1]
                if previous.bbox.y2 <= current.bbox.y1
            ]
            if predecessors:
                previous = max(predecessors, key=lambda element: element.bbox.y2)
                spacing[current.element_id] = (current.bbox.y1 - previous.bbox.y2) * source_scale
        return spacing

    def _fit_scale(self, sections, elements, roles, usable_width: float, usable_height: float) -> float:
        target = max(usable_height * self.word_safety_factor - _CROSS_ENGINE_PAGE_RESERVE_PT, 1.0)
        section_overhead = sum(section.kind != FlowKind.SINGLE for section in sections) * 0.05

        def content_height(scale: float) -> float:
            return section_overhead + sum(
                self._section_height(section, elements, roles, usable_width, scale) for section in sections
            )

        # Reserve vertical capacity without shrinking pages that already fit inside it.
        upper = 1.0
        if content_height(upper) <= target:
            return upper
        lower = 0.0
        for _ in range(12):
            middle = (lower + upper) / 2.0
            if content_height(middle) <= target:
                lower = middle
            else:
                upper = middle
        return max(lower, 0.001)

    def _section_height(self, section, elements, roles, usable_width: float, fit_scale: float) -> float:
        by_id = {element.element_id: element for element in elements}
        if section.kind == FlowKind.SINGLE:
            return sum(self._element_height(by_id[item], roles, usable_width, fit_scale) for item in section.element_ids)
        if section.kind == FlowKind.WRAPPED:
            floating = by_id[section.floating_element_id]
            floating_bbox = floating.payload.get("source_bbox") or (0, 0, 1, 1)
            wrap_width = max(usable_width - section.floating_width_pt - section.gutter_pt, 1.0)
            text_height = sum(
                self._element_height(
                    by_id[identifier],
                    roles,
                    wrap_width
                    if float((by_id[identifier].payload.get("source_bbox") or (0, 0, 1, 1))[1])
                    < float(floating_bbox[3])
                    else usable_width,
                    fit_scale,
                )
                for identifier in section.element_ids
                if identifier != section.floating_element_id
            )
            aspect = max(float(floating_bbox[3]) - float(floating_bbox[1]), 1.0) / max(
                float(floating_bbox[2]) - float(floating_bbox[0]), 1.0
            )
            floating_height = (
                section.floating_offset_y_pt + section.floating_width_pt * aspect
            ) * fit_scale + self._caption_height(floating, fit_scale)
            source_height = section.row_heights_pt[0] * fit_scale if section.row_heights_pt else 0.0
            return max(text_height, floating_height, source_height)
        widths = section.column_widths_pt
        if section.kind == FlowKind.SEQUENTIAL_COLUMNS:
            totals = [0.0] * len(widths)
            for identifier in section.element_ids:
                element = by_id[identifier]
                column = int(element.payload.get("column", 0))
                totals[column] += self._element_height(element, roles, widths[column], fit_scale)
            source_height = section.row_heights_pt[0] * fit_scale if section.row_heights_pt else 0.0
            return max(max(totals, default=0.0), source_height)
        row_heights = defaultdict(
            float,
            {row: height * fit_scale for row, height in enumerate(section.row_heights_pt)},
        )
        for cell in section.grid_cells:
            width = sum(widths[cell.column : cell.column + cell.column_span]) + section.gutter_pt * (cell.column_span - 1)
            height = sum(self._element_height(by_id[identifier], roles, width, fit_scale) for identifier in cell.element_ids)
            for row in range(cell.row, cell.row + cell.row_span):
                row_heights[row] = max(row_heights[row], height / cell.row_span)
        return sum(row_heights.values())

    def _with_vertical_tracks(self, section, elements, source_scale: float):
        if section.kind == FlowKind.SINGLE:
            return section
        by_id = {element.element_id: element for element in elements}
        boxes = sorted((by_id[identifier].bbox for identifier in section.element_ids), key=lambda box: box.y1)
        gap_limit = median(box.height for box in boxes)
        covered_bottom = boxes[0].y2
        for box in boxes[1:]:
            if box.y1 - covered_bottom > gap_limit:
                return section
            covered_bottom = max(covered_bottom, box.y2)
        if section.kind in {FlowKind.SEQUENTIAL_COLUMNS, FlowKind.WRAPPED}:
            height = (max(box.y2 for box in boxes) - min(box.y1 for box in boxes)) * source_scale
            return replace(section, row_heights_pt=(height,))

        row_count = max(cell.row + cell.row_span for cell in section.grid_cells)
        starts = []
        for row in range(row_count):
            identifiers = [
                identifier
                for cell in section.grid_cells
                if cell.row == row
                for identifier in cell.element_ids
            ]
            starts.append(min(by_id[identifier].bbox.y1 for identifier in identifiers) if identifiers else None)
        bottom = max(box.y2 for box in boxes)
        if any(value is None for value in starts) or any(left >= right for left, right in zip(starts, starts[1:])):
            height = (bottom - min(box.y1 for box in boxes)) * source_scale / row_count
            return replace(section, row_heights_pt=(height,) * row_count)
        boundaries = starts + [bottom]
        return replace(
            section,
            row_heights_pt=tuple((boundaries[index + 1] - boundaries[index]) * source_scale for index in range(row_count)),
        )

    @staticmethod
    def _element_height(element, roles, width: float, fit_scale: float) -> float:
        role = roles.get(element.role_id)
        spacing = float(element.payload.get("space_before_pt", 0.0)) * fit_scale
        if element.kind == "table_group" and element.payload.get("html"):
            soup = BeautifulSoup(str(element.payload["html"]), "html.parser")
            table = soup.find("table")
            rows = table.find_all("tr") if table else []
            weights = get_table_column_weights(table) if table else (1.0,)
            base_size = float(element.payload.get("table_font_size_pt") or (role.font_size_pt if role else 10.5))
            font_size = max(
                round(base_size * fit_scale * 2) / 2.0,
                float(element.payload.get("table_min_font_size_pt", 0.5)),
            )
            height = 0.0
            placements = get_table_cell_placements(table) if table else []
            for row_index, _row in enumerate(rows):
                row_height = font_size * 1.2
                for _source_row, column, _row_span, span, cell in placements:
                    if _source_row != row_index:
                        continue
                    table_width = width * float(element.payload.get("width_fraction", 1.0))
                    cell_width = table_width * sum(weights[column : column + span]) / max(sum(weights), 1.0)
                    units = estimate_text_units(cell.get_text(" ", strip=True))
                    lines = max(1, math.ceil(units * font_size / max(cell_width, 1.0)))
                    row_height = max(row_height, lines * font_size * 1.2)
                height += row_height + 2.0
            height = max(height, float(element.payload.get("table_height_pt") or 0.0) * fit_scale)
            return spacing + height + ReflowPlanner._caption_height(element, fit_scale)
        if role is not None and element.text:
            width = max(
                width
                - float(element.payload.get("left_indent_pt", 0.0))
                - float(element.payload.get("right_indent_pt", 0.0)),
                1.0,
            )
            font_size = max(round(role.font_size_pt * fit_scale * 2) / 2.0, 0.5)
            source_lines = element.payload.get("lines") or ()
            line_height = element.payload.get("line_height_pt")
            cjk_count = sum(1 for char in element.text if ord(char) >= 0x2E80)
            units = estimate_text_units(element.text)
            content_lines = max(1, math.ceil(units * font_size / max(width, 1.0)))
            if source_lines:
                source_bbox = element.payload.get("source_bbox") or (0, 0, 1, 1)
                source_width = (float(source_bbox[2]) - float(source_bbox[0])) * float(element.payload["source_scale"])
                observed_lines = max(1, round(len(source_lines) * source_width / max(width, 1.0) * fit_scale))
                lines = max(observed_lines, content_lines)
            else:
                lines = content_lines
            if source_lines and line_height:
                rendered_line_height = max(float(line_height) * fit_scale, font_size * 1.05)
                return spacing + lines * rendered_line_height + min(font_size * 0.15, 2.0)
            paragraph_boundary = font_size if cjk_count else font_size / 4.0
            return spacing + lines * font_size * role.line_spacing * 1.05 + paragraph_boundary
        bbox = element.payload.get("primary_bbox") or element.payload.get("source_bbox") or (0, 0, 1, 1)
        aspect = max(float(bbox[3]) - float(bbox[1]), 1.0) / max(float(bbox[2]) - float(bbox[0]), 1.0)
        visual_width = width * float(element.payload.get("width_fraction", 1.0)) * fit_scale
        return spacing + visual_width * aspect + ReflowPlanner._caption_height(element, fit_scale)

    @staticmethod
    def _caption_height(element, fit_scale: float) -> float:
        caption = str(element.payload.get("caption") or "")
        if not caption:
            return 0.0
        font_size = float(element.payload.get("caption_font_size_pt") or 10.0)
        return len(caption.splitlines()) * font_size * 1.1 * fit_scale

    @staticmethod
    def _alignment(element, frame) -> str:
        if element.kind in {"equation_group", "caption"}:
            return "center"
        left, right = frame
        width = max(right - left, 1.0)
        if element.kind == "figure_group":
            if element.bbox.x1 <= left + width * 0.08:
                return "left"
            if element.bbox.x2 >= right - width * 0.08:
                return "right"
            return "center"
        element_center = (element.bbox.x1 + element.bbox.x2) / 2.0
        centered = abs(element_center - (left + right) / 2.0) <= width * 0.04
        source_lines = element.payload.get("lines") or ()
        if element.kind == "heading" and len(source_lines) > 1 and element.bbox.x1 <= left + width * 0.05:
            return "left"
        if centered and (element.kind == "heading" or element.bbox.width <= width * 0.75):
            return "center"
        if element.bbox.x2 >= right - width * 0.03 and element.bbox.x1 > left + width * 0.25 and len(source_lines) <= 2:
            return "right"
        return "justify" if element.kind == "paragraph_group" and len(source_lines) >= 3 and len(element.text) >= 40 else "left"

    @staticmethod
    def _first_line_indent(element, source_scale: float) -> float:
        lefts = element.payload.get("line_lefts_px") or ()
        if len(lefts) < 2:
            return 0.0
        indent = float(lefts[0]) - median(float(value) for value in lefts[1:])
        return max(indent * source_scale, 0.0)

    @staticmethod
    def _heading_hanging_indent(element, source_scale: float) -> float:
        lefts = element.payload.get("line_lefts_px") or ()
        if element.kind != "heading" or len(lefts) < 2:
            return 0.0
        return max(median(float(value) for value in lefts[1:]) - float(lefts[0]), 0.0) * source_scale

    @staticmethod
    def _source_line_height(element, role, source_scale: float):
        lines = element.payload.get("lines") or ()
        if not lines or role is None:
            return None
        line_heights = element.payload.get("line_heights_px") or ()
        line_tops = element.payload.get("line_tops_px") or ()
        if len(line_tops) >= 2:
            observed = median(right - left for left, right in zip(line_tops, line_tops[1:])) * source_scale
        else:
            observed = median(line_heights) * source_scale if line_heights else element.bbox.height * source_scale / len(lines)
        return min(max(observed, role.font_size_pt * 1.05), role.font_size_pt * 1.5)

    @staticmethod
    def _table_font_size(element, source_scale: float, body_font_size: float):
        if element.kind != "table_group" or not element.payload.get("html"):
            return None
        table = BeautifulSoup(str(element.payload["html"]), "html.parser").find("table")
        row_count = len(table.find_all("tr")) if table else 0
        if row_count == 0:
            return None
        return min(element.bbox.height * source_scale / row_count / 1.45, body_font_size)

    @staticmethod
    def _caption_font_size(element, source_scale: float, body_font_size: float):
        heights = element.payload.get("caption_line_heights_px") or ()
        if not heights:
            return None
        return min(median(float(height) for height in heights) * source_scale / 1.15, body_font_size)

    @staticmethod
    def _union(rectangles):
        rects = tuple(rectangles)
        return Rect(
            min(rect.x1 for rect in rects),
            min(rect.y1 for rect in rects),
            max(rect.x2 for rect in rects),
            max(rect.y2 for rect in rects),
        )
