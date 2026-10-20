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
from docflow.renderer.docx_utils.html_table import (
    estimate_table_cell_lines,
    fit_table_font_size,
    get_table_cell_placements,
    get_table_column_weights,
    table_cell_available_width,
)
from docflow.planning.text_metrics import resolve_text_layout


class LayoutFitError(RuntimeError):
    """Raised when a page cannot satisfy the readable layout contract."""


class ReflowPlanner:
    GRID_WORD_SAFETY_FACTOR = 0.94
    FORMULA_WORD_SAFETY_FACTOR = 0.98
    MIN_GAP_SCALE = 0.30
    MIN_GAP_PT = 2.0
    MIN_PAGE_SCALE = 0.75
    MIN_BODY_FONT_PT = 7.5
    MIN_HEADING_FONT_PT = 9.0
    MIN_CAPTION_FONT_PT = 7.0
    PAGE_HEIGHT_TOLERANCE_PT = 2.0
    PAGE_RENDER_OVERHEAD_PT = 12.0
    WRAPPED_SECTION_OVERHEAD_PT = 18.0
    def __init__(self, page_long_edge_pt: float = 841.89, word_safety_factor: float = 0.95) -> None:
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
            if (element.kind == "figure_group" or element.text_structure.orientation == "vertical")
            and element.bbox.width <= page.width_px * 0.08
            and element.bbox.height >= element.bbox.width * 2.5
            and (element.bbox.x2 <= page.width_px * 0.12 or element.bbox.x1 >= page.width_px * 0.88)
        ]
        edge_visuals = set()
        for candidates in (
            [element for element in edge_candidates if element.bbox.x2 <= page.width_px * 0.12],
            [element for element in edge_candidates if element.bbox.x1 >= page.width_px * 0.88],
        ):
            if len(candidates) == 1 or (
                candidates
                and all(element.text_structure.orientation == "vertical" for element in candidates)
                and self._union(element.bbox for element in candidates).height >= page.height_px * 0.40
            ):
                edge_visuals.update(element.element_id for element in candidates)
        edge_visuals.update(
            element.element_id
            for element in page.elements
            if element.kind == "figure_group"
            and element.bbox.y1 <= page.height_px * 0.01
            and element.bbox.height <= page.height_px * 0.12
            and (element.bbox.x2 <= page.width_px * 0.55 or element.bbox.x1 >= page.width_px * 0.45)
        )
        edge_text_labels = {
            element.element_id
            for element in page.elements
            if self._is_edge_text_label(element, page)
        }
        furniture = [
            element
            for element in page.elements
            if element.kind in {"header", "footer", "page_number"}
            or element.element_id in edge_visuals
        ]
        embedded_table_text = self._embedded_table_text_ids(page.elements)
        if embedded_table_text:
            page = replace(
                page,
                elements=tuple(
                    element
                    for element in page.elements
                    if element.element_id not in embedded_table_text
                ),
            )
            furniture = [element for element in furniture if element.element_id not in embedded_table_text]
        body = [
            element
            for element in page.elements
            if element not in furniture
        ]
        bounds_body = [element for element in body if element.element_id not in edge_text_labels]
        bounds = (
            self._union(element.bbox for element in bounds_body)
            if bounds_body
            else Rect(0, 0, page.width_px, page.height_px)
        )
        lower_furniture_top = min(
            (element.bbox.y1 for element in furniture if element.bbox.y1 >= bounds.y2),
            default=page.height_px,
        )
        geometry = PageGeometry(
            width_pt,
            height_pt,
            bounds.y1 * scale,
            (page.width_px - bounds.x2) * scale,
            (page.height_px - lower_furniture_top) * scale,
            bounds.x1 * scale,
        )
        usable_width = geometry.width_pt - geometry.margin_left_pt - geometry.margin_right_pt
        sections, placement, span_constraints = self._build_sections(body, bounds, usable_width)
        sections, placement = self._pair_heading_icons(sections, body, placement, scale)
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
        row_alignments = {
            element.element_id: self._row_alignments(
                element,
                container_frames.get(element.element_id, (bounds.x1, bounds.x2)),
            )
            for element in page.elements
        }
        hanging_indents = {
            element.element_id: self._heading_hanging_indent(element, scale)
            if alignments[element.element_id] == "left"
            else 0.0
            for element in page.elements
        }
        option_indents = {
            element.element_id: element.text_structure.hanging_indent_px * scale
            for element in page.elements
        }
        background_indents = {
            element.element_id: max(
                self._content_bbox(element).x1 - self._background_bbox(element).x1,
                0.0,
            )
            * scale
            for element in page.elements
        }
        body_font_size = role_by_id[default_body_role].font_size_pt if default_body_role in role_by_id else 10.5
        table_font_sizes = {
            element.element_id: self._table_font_size(element, scale, body_font_size)
            for element in page.elements
        }
        column_offsets = self._column_offsets(sections, page.elements, placement, scale)
        uniform_list_roles = self._uniform_list_roles(body, role_by_id)

        def target_bbox(element):
            return (
                geometry.margin_left_pt + (element.bbox.x1 - bounds.x1) * scale,
                geometry.margin_top_pt + (element.bbox.y1 - bounds.y1) * scale,
                geometry.margin_left_pt + (element.bbox.x2 - bounds.x1) * scale,
                geometry.margin_top_pt + (element.bbox.y2 - bounds.y1) * scale,
            )

        planned = tuple(
            PlannedElement(
                element.element_id,
                element.kind,
                role_id=uniform_list_roles.get(element.element_id, element.role_id or default_body_role),
                text=element.text,
                text_structure=element.text_structure,
                text_rows=element.text_rows,
                row_alignments=row_alignments[element.element_id],
                payload={
                    **dict(element.payload),
                    "source_bbox": (element.bbox.x1, element.bbox.y1, element.bbox.x2, element.bbox.y2),
                    "source_frame_width_px": max(
                        container_frames.get(element.element_id, (bounds.x1, bounds.x2))[1]
                        - container_frames.get(element.element_id, (bounds.x1, bounds.x2))[0],
                        1.0,
                    ),
                    "target_bbox_pt": target_bbox(element),
                    "target_width_pt": max(target_bbox(element)[2] - target_bbox(element)[0], 1.0),
                    "target_height_pt": max(target_bbox(element)[3] - target_bbox(element)[1], 1.0),
                    "width_fraction": span_constraints.get(element.element_id, {}).get(
                        "width_fraction",
                        min(
                            (
                                self._layout_bbox(element)
                                if element.kind == "figure_group"
                                else self._background_bbox(element)
                            ).width
                            / container_widths.get(element.element_id, max(bounds.width, 1.0)),
                            1.0,
                        ),
                    ),
                    "column": span_constraints.get(element.element_id, {}).get(
                        "column",
                        placement.get(element.element_id, 0),
                    ),
                    "alignment": (
                        "left"
                        if element.element_id in span_constraints and element.kind != "heading"
                        else alignments[element.element_id]
                    ),
                    "first_line_indent_pt": (
                        self._first_line_indent(element, scale)
                        - hanging_indents[element.element_id]
                        - option_indents[element.element_id]
                        + background_indents[element.element_id]
                    ),
                    "left_indent_pt": max(
                        horizontal_indents.get(
                            element.element_id,
                            (max(element.bbox.x1 * scale - geometry.margin_left_pt, 0.0), 0.0),
                        )[0]
                        + hanging_indents[element.element_id]
                        + option_indents[element.element_id],
                        span_constraints.get(element.element_id, {}).get("left_indent_pt", 0.0),
                    ),
                    "right_indent_pt": max(
                        horizontal_indents.get(
                            element.element_id,
                            (0.0, max((page.width_px - element.bbox.x2) * scale - geometry.margin_right_pt, 0.0)),
                        )[1],
                        span_constraints.get(element.element_id, {}).get("right_indent_pt", 0.0),
                    ),
                    "space_before_pt": vertical_spacing.get(element.element_id, 0.0),
                    "column_offset_pt": column_offsets.get(element.element_id, 0.0),
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
                content_bbox=element.content_bbox,
            )
            for element in page.elements
        )
        usable_height = geometry.height_pt - geometry.margin_top_pt - geometry.margin_bottom_pt
        fit_scale, gap_scale = self._fit_page(
            sections,
            planned,
            role_by_id,
            usable_width,
            usable_height,
            page_index=page.page_index,
        )
        planned = self._with_compressed_gaps(planned, gap_scale)
        sections = tuple(
            replace(
                section,
                row_heights_pt=tuple(
                    height / fit_scale
                    for height in self._grid_row_heights(section, planned, role_by_id, fit_scale)
                ),
            )
            if section.kind == FlowKind.GRID
            and not (
                fit_scale <= 0.05
                and section.row_heights_pt
                and sum(self._grid_row_heights(section, planned, role_by_id, fit_scale))
                > sum(section.row_heights_pt) * fit_scale * 1.5
            )
            else section
            for section in sections
        )
        planned = self._resolve_text_layouts(sections, planned, role_by_id, fit_scale, usable_width)
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
        span_constraints = {}
        anchor_lanes = self._anchor_lanes(elements, bounds.width)
        if len(anchor_lanes) < 2:
            anchor_lanes = self._local_visual_lanes(elements, bounds.width) or anchor_lanes
        lane_bounds = [
            (median(self._layout_bbox(item).x1 for item in lane), median(self._layout_bbox(item).x2 for item in lane))
            for lane in anchor_lanes
        ]
        separators = [(left[1] + right[0]) / 2.0 for left, right in zip(lane_bounds, lane_bounds[1:])]
        spanning = [
            element
            for element in elements
            if any(
                self._layout_bbox(element).x1 < separator - 1.0
                and self._layout_bbox(element).x2 > separator + 1.0
                for separator in separators
            )
        ]
        lane_elements = [element for element in elements if element not in spanning]
        refined_lanes = self._anchor_lanes(lane_elements, bounds.width)
        if len(refined_lanes) >= 2:
            lane_bounds = [
                (median(self._layout_bbox(item).x1 for item in lane), median(self._layout_bbox(item).x2 for item in lane))
                for lane in refined_lanes
            ]
            separators = [(left[1] + right[0]) / 2.0 for left, right in zip(lane_bounds, lane_bounds[1:])]
            spanning = [
                element
                for element in elements
                if any(
                    self._layout_bbox(element).x1 < separator - 1.0
                    and self._layout_bbox(element).x2 > separator + 1.0
                    for separator in separators
                )
            ]
        non_spanning = [element for element in elements if element not in spanning]
        spanning = [
            element
            for element in spanning
            if not any(
                max(0.0, min(element.bbox.y2, other.bbox.y2) - max(element.bbox.y1, other.bbox.y1))
                >= min(element.bbox.height, other.bbox.height) * 0.30
                for other in non_spanning
            )
        ]
        spanning.sort(key=lambda item: ((item.bbox.y1 + item.bbox.y2) / 2.0, item.bbox.x1))
        groups = []
        for element in spanning:
            if groups:
                group = groups[-1]
                ordered = sorted((*group, element), key=lambda item: item.bbox.x1)
                no_overlap = all(left.bbox.x2 <= right.bbox.x1 + 1.0 for left, right in zip(ordered, ordered[1:]))
                overlap = min(member.bbox.y2 for member in group) - max(member.bbox.y1 for member in group)
                reference = min(min(member.bbox.height for member in group), element.bbox.height)
                if no_overlap and overlap >= reference * 0.50:
                    group.append(element)
                    continue
            groups.append([element])
        cuts = [(min(item.bbox.y1 for item in group) + max(item.bbox.y2 for item in group)) / 2.0 for group in groups]
        bands = [[] for _ in range(len(groups) + 1)]
        spanning_ids = {element.element_id for element in spanning}
        for element in elements:
            if element.element_id not in spanning_ids:
                bands[bisect_right(cuts, (element.bbox.y1 + element.bbox.y2) / 2.0)].append(element)
        for index, band in enumerate(bands):
            if band:
                section, columns = self._narrow_section(band, bounds, usable_width, len(sections), anchor_lanes)
                sections.append(section)
                placement.update(columns)
            if index >= len(groups):
                continue
            group = groups[index]
            section_id = f"section_{len(sections)}"
            if len(group) == 1:
                element = group[0]
                span_constraints[element.element_id] = {
                    "left_indent_pt": max(element.bbox.x1 - bounds.x1, 0.0) / max(bounds.width, 1.0) * usable_width,
                    "right_indent_pt": max(bounds.x2 - element.bbox.x2, 0.0) / max(bounds.width, 1.0) * usable_width,
                    "width_fraction": max(element.bbox.width, 1.0) / max(bounds.width, 1.0),
                    "column": 0,
                }
                sections.append(FlowSection(section_id, FlowKind.SINGLE, (element.element_id,)))
                placement[element.element_id] = 0
                continue
            ordered = sorted(group, key=lambda item: item.bbox.x1)
            widths = [max(item.bbox.width / max(bounds.width, 1.0) * usable_width, 1.0) for item in ordered]
            gaps = [
                max(right.bbox.x1 - left.bbox.x2, 0.0) / max(bounds.width, 1.0) * usable_width
                for left, right in zip(ordered, ordered[1:])
            ]
            sections.append(
                FlowSection(
                    section_id,
                    FlowKind.SEQUENTIAL_COLUMNS,
                    tuple(item.element_id for item in ordered),
                    column_widths_pt=tuple(widths),
                    gutter_pt=median(gaps) if gaps else 0.0,
                )
            )
            placement.update({item.element_id: column for column, item in enumerate(ordered)})
        return tuple(sections), placement, span_constraints

    @staticmethod
    def _pair_heading_icons(sections, elements, placement, source_scale: float):
        by_id = {element.element_id: element for element in elements}
        paired = []
        updated_placement = dict(placement)
        index = 0
        while index < len(sections):
            current = sections[index]
            next_section = sections[index + 1] if index + 1 < len(sections) else None
            pair = None
            if (
                next_section is not None
                and current.kind == FlowKind.SINGLE
                and next_section.kind == FlowKind.SINGLE
                and len(next_section.element_ids) == 1
            ):
                heading = by_id[next_section.element_ids[0]]
                figures = [
                    by_id[identifier]
                    for identifier in current.element_ids
                    if by_id[identifier].kind == "figure_group"
                ]
                if heading.kind == "heading" and len(figures) == 1:
                    figure = figures[0]
                    overlap = max(0.0, min(figure.bbox.y2, heading.bbox.y2) - max(figure.bbox.y1, heading.bbox.y1))
                    gap = max(heading.bbox.x1 - figure.bbox.x2, figure.bbox.x1 - heading.bbox.x2, 0.0)
                    if overlap / max(min(figure.bbox.height, heading.bbox.height), 1.0) >= 0.50 and gap <= heading.bbox.height * 2.0:
                        pair = (figure, heading)
            if pair is None:
                paired.append(current)
                index += 1
                continue
            figure, heading = pair
            remaining_ids = tuple(identifier for identifier in current.element_ids if identifier != figure.element_id)
            if remaining_ids:
                paired.append(replace(current, element_ids=remaining_ids))
            widths = (
                max(figure.bbox.width * source_scale, 1.0),
                max(heading.bbox.width * source_scale, 1.0),
            )
            updated_placement[figure.element_id] = 0
            updated_placement[heading.element_id] = 1
            paired.append(
                FlowSection(
                    f"section_{len(paired)}",
                    FlowKind.GRID,
                    (figure.element_id, heading.element_id),
                    column_widths_pt=widths,
                    gutter_pt=max(max(figure.bbox.x1, heading.bbox.x1) - min(figure.bbox.x2, heading.bbox.x2), 0.0)
                    * source_scale,
                    grid_cells=(
                        GridCell(0, 0, (figure.element_id,)),
                        GridCell(0, 1, (heading.element_id,)),
                    ),
                )
            )
            index += 2
        return tuple(paired), updated_placement

    @staticmethod
    def _column_offsets(sections, elements, placement, source_scale: float):
        by_id = {element.element_id: element for element in elements}
        offsets = {}
        for section in sections:
            if section.kind != FlowKind.SEQUENTIAL_COLUMNS or not section.element_ids:
                continue
            first_by_column = {}
            for identifier in section.element_ids:
                element = by_id[identifier]
                column = int(placement.get(identifier, 0))
                if column not in first_by_column or element.bbox.y1 < first_by_column[column].bbox.y1:
                    first_by_column[column] = element
            top = min(element.bbox.y1 for element in first_by_column.values())
            offsets.update(
                {
                    element.element_id: max(element.bbox.y1 - top, 0.0) * source_scale
                    for element in first_by_column.values()
                }
            )
        return offsets

    @staticmethod
    def _uniform_list_roles(elements, role_by_id):
        ordered = sorted(
            (
                element
                for element in elements
                if element.kind == "paragraph_group" and element.text_structure.is_list
            ),
            key=lambda element: (element.bbox.y1, element.bbox.x1),
        )
        groups = []
        for element in ordered:
            if (
                groups
                and element.bbox.y1 <= max(member.bbox.y2 for member in groups[-1]) + max(element.bbox.height, 1.0) * 1.5
            ):
                groups[-1].append(element)
            else:
                groups.append([element])
        overrides = {}
        for group in groups:
            roles = [role_by_id.get(element.role_id) for element in group]
            roles = [role for role in roles if role is not None]
            if len(group) < 2 or not roles:
                continue
            target_size = median(role.font_size_pt for role in roles)
            target = min(roles, key=lambda role: abs(role.font_size_pt - target_size))
            overrides.update({element.element_id: target.role_id for element in group})
        return overrides

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
    def _embedded_table_text_ids(elements) -> set[str]:
        tables = [
            element
            for element in elements
            if element.kind == "table_group" and element.payload.get("html")
        ]
        embedded = set()
        for element in elements:
            if element.kind not in {"paragraph_group", "footnote", "caption"} or not element.text:
                continue
            text = "".join(str(element.text).split())
            for table in tables:
                cell_texts = [
                    "".join(node.get_text(" ", strip=True).split())
                    for node in BeautifulSoup(str(table.payload["html"]), "html.parser").find_all(["td", "th"])
                ]
                contains = False
                for table_text in cell_texts:
                    common_prefix = 0
                    for left, right in zip(text, table_text):
                        if left != right:
                            break
                        common_prefix += 1
                    if (
                        text in table_text
                        or (len(table_text) >= 4 and table_text in text)
                        or common_prefix >= 4
                    ):
                        contains = True
                        break
                if not text or not contains:
                    continue
                horizontal = max(
                    0.0,
                    min(element.bbox.x2, table.bbox.x2) - max(element.bbox.x1, table.bbox.x1),
                ) / max(element.bbox.width, 1.0)
                vertical_overlap = max(
                    0.0,
                    min(element.bbox.y2, table.bbox.y2) - max(element.bbox.y1, table.bbox.y1),
                )
                if horizontal >= 0.5 and vertical_overlap >= element.bbox.height * 0.5:
                    embedded.add(element.element_id)
                    break
        return embedded

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
                text_members = [
                    item
                    for item in members
                    if item.kind in {"paragraph_group", "heading", "caption"}
                ]
                if len(text_members) >= 2:
                    left = median(ReflowPlanner._layout_bbox(item).x1 for item in text_members)
                    right = median(ReflowPlanner._layout_bbox(item).x2 for item in text_members)
                else:
                    left = min(ReflowPlanner._layout_bbox(item).x1 for item in members)
                    right = max(ReflowPlanner._layout_bbox(item).x2 for item in members)
                column_frames[column] = (left, right)
            edge_columns = sorted(column_frames)
            for column, members in by_column.items():
                text_members = [
                    item
                    for item in members
                    if item.kind in {"paragraph_group", "heading", "caption"}
                ]
                if len(text_members) < 2 or min(item.bbox.x2 for item in text_members) > max(
                    item.bbox.x1 for item in text_members
                ):
                    continue
                if column == edge_columns[0] and len(edge_columns) > 1:
                    column_frames[column] = (bounds.x1, column_frames[edge_columns[1]][0])
                elif column == edge_columns[-1] and len(edge_columns) > 1:
                    column_frames[column] = (column_frames[edge_columns[-2]][1], bounds.x2)
            for column, members in by_column.items():
                left, right = column_frames[column]
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
            content_bbox = ReflowPlanner._content_bbox(element)
            visual_bbox = ReflowPlanner._background_bbox(element)
            centered = abs((content_bbox.x1 + content_bbox.x2 - left - right) / 2.0) <= source_width * 0.04
            width_fraction = visual_bbox.width / source_width
            section_kind = section_kinds[element.element_id]
            single_flow_uses_full_width = element.element_id in wrapped_figure_ids or (
                section_kind == FlowKind.SINGLE and (element.kind == "heading" or width_fraction > 0.85)
            )
            if element.kind == "table_group":
                target_width = target_widths[element.element_id]
                result[element.element_id] = (
                    max(element.bbox.x1 - left, 0.0) / source_width * target_width,
                    max(right - element.bbox.x2, 0.0) / source_width * target_width,
                )
                continue
            if element.kind == "heading":
                result[element.element_id] = (
                    0.0
                    if centered or width_fraction > 0.85
                    else max(visual_bbox.x1 - left, 0.0) / source_width * target_widths[element.element_id],
                    0.0,
                )
                continue
            if (
                element.kind == "paragraph_group"
                and section_kind == FlowKind.SINGLE
                and ReflowPlanner._alignment(element, (left, right)) in {"left", "justify"}
            ):
                target_width = target_widths[element.element_id]
                result[element.element_id] = (
                    max(visual_bbox.x1 - left, 0.0) / source_width * target_width,
                    0.0,
                )
                continue
            if element.kind != "paragraph_group" or width_fraction < 0.3 or single_flow_uses_full_width:
                result[element.element_id] = (0.0, 0.0)
                continue
            target_width = target_widths[element.element_id]
            alignment = ReflowPlanner._alignment(element, (left, right))
            right_indent = max(right - visual_bbox.x2, 0.0) / source_width * target_width
            if element.text_structure.preserve_source_lines or alignment in {"left", "justify"}:
                right_indent = 0.0
            result[element.element_id] = (
                max(visual_bbox.x1 - left, 0.0) / source_width * target_width,
                right_indent,
            )
        return result

    @staticmethod
    def _visual_width(element) -> float:
        return ReflowPlanner._layout_bbox(element).width

    @staticmethod
    def _layout_bbox(element) -> Rect:
        bbox = (
            element.payload.get("background_bbox")
            or element.payload.get("primary_bbox")
            or element.payload.get("source_bbox")
        )
        return Rect.from_sequence(bbox) if bbox else getattr(element, "bbox", Rect(0, 0, 1, 1))

    @staticmethod
    def _content_bbox(element) -> Rect:
        return element.content_bbox or ReflowPlanner._layout_bbox(element)

    @staticmethod
    def _background_bbox(element) -> Rect:
        bbox = element.payload.get("background_bbox")
        return Rect.from_sequence(bbox) if bbox else ReflowPlanner._content_bbox(element)

    def _narrow_section(self, elements, bounds: Rect, usable_width: float, section_index: int, fallback_lanes=()):
        if len(elements) == 1:
            section_id = f"section_{section_index}"
            return FlowSection(section_id, FlowKind.SINGLE, (elements[0].element_id,)), {elements[0].element_id: 0}
        lanes = self._anchor_lanes(elements, bounds.width)
        if len(fallback_lanes) >= 2 and self._uses_all_lanes(elements, fallback_lanes):
            lanes = list(fallback_lanes)
        elif len(lanes) < 2:
            lanes = (
                self._local_visual_lanes(elements, bounds.width)
                or (list(fallback_lanes) if len(fallback_lanes) >= 2 else lanes)
            )
        for rail in self._side_rail_lanes(elements, bounds, lanes):
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
            coverage_threshold = 0.20 if item.kind == "figure_group" else 0.30
            overlaps = [
                index
                for index, (left, right) in enumerate(lane_bounds)
                if max(0.0, min(bbox.x2, right) - max(bbox.x1, left)) / max(right - left, 1.0)
                >= coverage_threshold
            ]
            spans[item.element_id] = max(overlaps) - min(overlaps) + 1 if overlaps else 1
            if len(overlaps) > 1:
                placement[item.element_id] = min(overlaps)
        lane_members = [
            [item for item in elements if placement[item.element_id] == index and spans[item.element_id] == 1]
            for index in range(len(lane_bounds))
        ]
        updated_lane_bounds = []
        for index, members in enumerate(lane_members):
            if members:
                updated_lane_bounds.append(
                    (
                        min(self._layout_bbox(item).x1 for item in members),
                        max(self._layout_bbox(item).x2 for item in members),
                    )
                )
            else:
                updated_lane_bounds.append(lane_bounds[index])
        lane_bounds = updated_lane_bounds

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
        text_intervals = {}
        for item in elements:
            if item.kind not in {"paragraph_group", "heading", "caption"}:
                continue
            start = placement[item.element_id]
            text_intervals[start] = max(text_intervals.get(start, 0), spans[item.element_id])
        logical_columns = sorted((start, span) for start, span in text_intervals.items())
        logical_widths = [
            sum(column_widths[start : start + span]) + gutter * (span - 1)
            for start, span in logical_columns
        ]
        logical_text_counts = [
            sum(
                item.kind in {"paragraph_group", "heading", "caption"}
                and start <= placement[item.element_id] < start + span
                for item in elements
            )
            for start, span in logical_columns
        ]
        independent_columns = (
            len(logical_columns) >= 2
            and (
                max(logical_widths) <= min(logical_widths) * 1.5
                or all(count >= 3 for count in logical_text_counts)
            )
            and all(
                left + span <= right
                for (left, span), (right, _next_span) in zip(logical_columns, logical_columns[1:])
            )
            and all(
                count >= 2
                for count in logical_text_counts
            )
            and all(
                sum(
                    start <= placement[item.element_id]
                    and placement[item.element_id] + spans[item.element_id] <= start + span
                    for start, span in logical_columns
                ) == 1
                for item in elements
            )
        )
        ordered_columns = (
            max(spans.values(), default=1) == 1
            and collapsed == sorted(set(collapsed))
            and len(collapsed) == len(lanes)
            and balanced_columns
        )
        if independent_columns or ordered_columns:
            if independent_columns:
                remapped = {}
                logical_frames = []
                for new_column, (start, span) in enumerate(logical_columns):
                    members = [
                        item
                        for item in elements
                        if start <= placement[item.element_id] < start + span
                    ]
                    logical_frames.append(
                        (
                            min(self._layout_bbox(item).x1 for item in members),
                            max(self._layout_bbox(item).x2 for item in members),
                        )
                    )
                    for item in elements:
                        if start <= placement[item.element_id] < start + span:
                            remapped[item.element_id] = new_column
                placement = remapped
                logical_gaps = [
                    max(right[0] - left[1], 0.0)
                    for left, right in zip(logical_frames, logical_frames[1:])
                ]
                gutter = median(logical_gaps) / max(bounds.width, 1.0) * usable_width
                available = max(usable_width - gutter * (len(logical_frames) - 1), 1.0)
                frame_widths = [right - left for left, right in logical_frames]
                column_widths = tuple(width / max(sum(frame_widths), 1.0) * available for width in frame_widths)
            element_ids = tuple(
                item.element_id
                for column in sorted(set(placement.values()))
                for item in sorted(
                    (item for item in elements if placement[item.element_id] == column),
                    key=lambda value: (self._layout_bbox(value).y1, self._layout_bbox(value).x1),
                )
            )
            return FlowSection(
                section_id,
                FlowKind.SEQUENTIAL_COLUMNS,
                element_ids,
                column_widths_pt=column_widths,
                gutter_pt=gutter,
            ), placement

        lane_sizes = [sum(placement[item.element_id] == column for item in elements) for column in range(len(lanes))]
        if (
            max(spans.values(), default=1) == 1
            and min(widths) <= max(widths) * 0.50
            and min(lane_sizes) == 1
        ):
            cells = tuple(
                GridCell(
                    0,
                    column,
                    tuple(
                        item.element_id
                        for item in sorted(elements, key=lambda value: self._layout_bbox(value).y1)
                        if placement[item.element_id] == column
                    ),
                )
                for column in range(len(lanes))
                if any(placement[item.element_id] == column for item in elements)
            )
            return FlowSection(
                section_id,
                FlowKind.GRID,
                element_ids,
                column_widths_pt=column_widths,
                gutter_pt=gutter,
                grid_cells=cells,
            ), placement

        has_spanning = max(spans.values(), default=1) > 1

        rows = self._spanning_grid_rows(elements) if has_spanning else self._grid_rows(elements)
        cells = defaultdict(list)
        for item in elements:
            cells[(rows[item.element_id], placement[item.element_id], spans[item.element_id])].append(item.element_id)
        by_id = {item.element_id: item for item in elements}
        grid_cells = [
            GridCell(
                row,
                column,
                tuple(sorted(ids, key=lambda identifier: self._layout_bbox(by_id[identifier]).y1)),
                column_span=column_span,
            )
            for (row, column, column_span), ids in sorted(cells.items())
        ]
        row_count = max(cell.row + cell.row_span for cell in grid_cells)
        grid_cells = [
            replace(
                cell,
                row_span=max(
                    cell.row_span,
                    min(
                        (
                            other.row
                            for other in grid_cells
                            if other.row >= cell.row + cell.row_span
                            and cell.column < other.column + other.column_span
                            and other.column < cell.column + cell.column_span
                        ),
                        default=row_count,
                    )
                    - cell.row,
                ),
            )
            for cell in grid_cells
        ]
        grid_cells = ReflowPlanner._merge_contiguous_grid_cells(grid_cells, by_id)
        return FlowSection(
            section_id,
            FlowKind.GRID,
            element_ids,
            column_widths_pt=column_widths,
            gutter_pt=gutter,
            grid_cells=tuple(sorted(grid_cells, key=lambda cell: (cell.row, cell.column))),
        ), placement

    @staticmethod
    def _merge_contiguous_grid_cells(cells, by_id):
        merged = []
        locked = [cell for cell in cells if cell.column_span > 1]
        for column in sorted({cell.column for cell in cells if cell.column_span == 1}):
            column_cells = sorted(
                (cell for cell in cells if cell.column_span == 1 and cell.column == column),
                key=lambda cell: cell.row,
            )
            current = column_cells[0]
            for candidate in column_cells[1:]:
                current_ids = tuple(current.element_ids)
                candidate_ids = tuple(candidate.element_ids)
                text_kinds = {"paragraph_group", "heading", "caption"}
                if not all(by_id[item].kind in text_kinds for item in current_ids + candidate_ids):
                    merged.append(current)
                    current = candidate
                    continue
                if current.row_span == 1 and candidate.row == current.row + 1:
                    merged.append(current)
                    current = candidate
                    continue
                current_top = min(ReflowPlanner._layout_bbox(by_id[item]).y1 for item in current_ids)
                current_bottom = max(ReflowPlanner._layout_bbox(by_id[item]).y2 for item in current_ids)
                candidate_top = min(ReflowPlanner._layout_bbox(by_id[item]).y1 for item in candidate_ids)
                candidate_bottom = max(ReflowPlanner._layout_bbox(by_id[item]).y2 for item in candidate_ids)
                gap = max(candidate_top - current_bottom, 0.0)
                contact_limit = min(current_bottom - current_top, candidate_bottom - candidate_top) * 0.35
                blocked = any(
                    cell is not current
                    and cell is not candidate
                    and cell.column <= column < cell.column + cell.column_span
                    and cell.row < candidate.row + candidate.row_span
                    and cell.row + cell.row_span > current.row + current.row_span
                    for cell in locked
                )
                if not blocked and gap <= max(contact_limit, 1.0):
                    element_ids = tuple(
                        sorted(
                            current_ids + candidate_ids,
                            key=lambda identifier: ReflowPlanner._layout_bbox(by_id[identifier]).y1,
                        )
                    )
                    current = replace(
                        current,
                        row_span=candidate.row + candidate.row_span - current.row,
                        element_ids=element_ids,
                    )
                else:
                    merged.append(current)
                    current = candidate
            merged.append(current)
        return tuple(sorted((*merged, *locked), key=lambda cell: (cell.row, cell.column)))

    @staticmethod
    def _side_rail_lanes(elements, bounds: Rect, main_lanes=()):
        candidates = [
            item
            for item in elements
            if (item.kind == "figure_group" or item.text_structure.orientation == "vertical")
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
        rails = [
            group
            for group in groups
            if len(group) >= 2 and sum(item.bbox.height for item in group) >= bounds.height * 0.30
        ]
        lane_ids = {item.element_id for lane in main_lanes for item in lane}
        for lane in main_lanes:
            if len(lane) < 2:
                continue
            left = median(ReflowPlanner._layout_bbox(item).x1 for item in lane)
            right = median(ReflowPlanner._layout_bbox(item).x2 for item in lane)
            lane_width = max(right - left, 1.0)
            for item in elements:
                bbox = ReflowPlanner._layout_bbox(item)
                if (
                    item.element_id in lane_ids
                    or bbox.width > lane_width * 0.30
                    or not (bbox.x2 <= left or bbox.x1 >= right)
                ):
                    continue
                if any(
                    max(0.0, min(bbox.y2, ReflowPlanner._layout_bbox(peer).y2) - max(bbox.y1, ReflowPlanner._layout_bbox(peer).y1))
                    / max(min(bbox.height, ReflowPlanner._layout_bbox(peer).height), 1.0)
                    >= 0.20
                    for peer in lane
                ):
                    rails.append([item])
        return rails

    @staticmethod
    def _uses_all_lanes(elements, lanes) -> bool:
        lane_bounds = [
            (
                median(ReflowPlanner._layout_bbox(item).x1 for item in lane),
                median(ReflowPlanner._layout_bbox(item).x2 for item in lane),
            )
            for lane in lanes
        ]
        covered = set()
        for item in elements:
            bbox = ReflowPlanner._layout_bbox(item)
            overlaps = [
                index
                for index, (left, right) in enumerate(lane_bounds)
                if max(0.0, min(bbox.x2, right) - max(bbox.x1, left))
                / max(min(bbox.width, right - left), 1.0)
                >= 0.30
            ]
            if len(overlaps) == 1:
                covered.add(overlaps[0])
        return len(covered) == len(lane_bounds)

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
            if len(candidates) < 2:
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
        dominant = max(
            (lane for lanes in candidate_lanes for lane in lanes),
            key=len,
            default=None,
        )
        return [dominant] if dominant else []

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
    def _spanning_grid_rows(elements):
        ordered = sorted(elements, key=lambda element: ReflowPlanner._layout_bbox(element).y1)
        tolerance = median(ReflowPlanner._layout_bbox(element).height for element in ordered) * 0.15
        row_tops = []
        rows = {}
        for item in ordered:
            top = ReflowPlanner._layout_bbox(item).y1
            if not row_tops or top - row_tops[-1] > tolerance:
                row_tops.append(top)
            rows[item.element_id] = len(row_tops) - 1
        return rows

    @staticmethod
    def _vertical_spacing(elements, sections, placement, source_scale: float):
        by_id = {element.element_id: element for element in elements}
        spacing = {}
        for section in sections:
            by_column = defaultdict(list)
            grid_elements = [by_id[identifier] for identifier in section.element_ids]
            grid_cell_by_id = {
                identifier: cell
                for cell in section.grid_cells
                for identifier in cell.element_ids
            }
            for identifier in section.element_ids:
                current = by_id[identifier]
                column = placement.get(identifier, 0)
                predecessors = [
                    previous
                    for previous in (grid_elements if section.kind == FlowKind.GRID else by_column[column])
                    if previous.element_id != current.element_id
                    and previous.bbox.y1 <= current.bbox.y1
                    and max(
                        0.0,
                        min(previous.bbox.y2, current.bbox.y2) - max(previous.bbox.y1, current.bbox.y1),
                    )
                    < min(previous.bbox.height, current.bbox.height) * 0.45
                    and max(0.0, min(previous.bbox.x2, current.bbox.x2) - max(previous.bbox.x1, current.bbox.x1))
                    / max(min(previous.bbox.width, current.bbox.width), 1.0)
                    >= 0.30
                ]
                if predecessors:
                    previous = max(predecessors, key=lambda element: element.bbox.y2)
                    spacing[identifier] = ReflowPlanner._capped_paragraph_gap(
                        current,
                        previous,
                        max(current.bbox.y1 - previous.bbox.y2, 0.0),
                    ) * source_scale
                elif section.kind == FlowKind.GRID and identifier in grid_cell_by_id:
                    current_cell = grid_cell_by_id[identifier]
                    previous_rows = [
                        by_id[previous_id]
                        for cell in section.grid_cells
                        if cell.row + cell.row_span <= current_cell.row
                        for previous_id in cell.element_ids
                        if by_id[previous_id].bbox.y2 <= current.bbox.y1
                    ]
                    if previous_rows:
                        previous = max(previous_rows, key=lambda element: element.bbox.y2)
                        spacing[identifier] = ReflowPlanner._capped_paragraph_gap(
                            current,
                            previous,
                            max(current.bbox.y1 - previous.bbox.y2, 0.0),
                        ) * source_scale
                if section.kind != FlowKind.GRID:
                    by_column[column].append(current)
        section_elements = [
            [by_id[identifier] for identifier in section.element_ids]
            for section in sections
        ]
        for section_index, section in enumerate(sections[1:], start=1):
            current_elements = section_elements[section_index]
            section_top = min(element.bbox.y1 for element in section_elements[section_index])
            predecessors = [
                previous
                for previous in section_elements[section_index - 1]
                if previous.bbox.y2 <= section_top
            ]
            if not predecessors:
                continue
            gap = (section_top - max(previous.bbox.y2 for previous in predecessors)) * source_scale
            if section.kind == FlowKind.SEQUENTIAL_COLUMNS:
                first_ids = {
                    min(
                        (element for element in current_elements if placement.get(element.element_id, 0) == column),
                        key=lambda element: element.bbox.y1,
                    ).element_id
                    for column in set(placement.get(element.element_id, 0) for element in current_elements)
                }
            elif section.kind == FlowKind.GRID:
                first_row = min(cell.row for cell in section.grid_cells)
                first_ids = {
                    min(cell.element_ids, key=lambda identifier: by_id[identifier].bbox.y1)
                    for cell in section.grid_cells
                    if cell.row == first_row
                }
            else:
                first_ids = {
                    element.element_id
                    for element in current_elements
                    if element.bbox.y1 <= section_top + max(element.bbox.height * 0.10, 2.0)
                }
            spacing.update((identifier, gap) for identifier in first_ids)
        return spacing

    @staticmethod
    def _capped_paragraph_gap(current, previous, gap_px: float) -> float:
        if current.kind != "paragraph_group" or previous.kind != "paragraph_group":
            return gap_px
        return min(gap_px, max(ReflowPlanner._line_pitch_px(current), ReflowPlanner._line_pitch_px(previous)))

    @staticmethod
    def _line_pitch_px(element) -> float:
        rows = element.text_rows
        tops = tuple(row.bbox.y1 for row in rows) or element.payload.get("line_tops_px") or ()
        if len(tops) >= 2:
            pitches = [right - left for left, right in zip(tops, tops[1:]) if right > left]
            if pitches:
                return median(pitches)
        pitches = element.payload.get("line_pitches_px") or ()
        if pitches:
            return median(float(value) for value in pitches)
        line_count = len(rows) or len(element.payload.get("lines") or ()) or 1
        return max(element.bbox.height / line_count, 1.0)

    def _fit_scale(self, sections, elements, roles, usable_width: float, usable_height: float) -> float:
        return self._fit_page(sections, elements, roles, usable_width, usable_height)[0]

    def _fit_page(
        self,
        sections,
        elements,
        roles,
        usable_width: float,
        usable_height: float,
        page_index: int | None = None,
    ) -> tuple[float, float]:
        by_id = {element.element_id: element for element in elements}
        has_formulas = any(
            (element.kind == "equation_group" and element.payload.get("latex"))
            or element.payload.get("inline_formulas")
            or (element.kind == "table_group" and "docflow-formula" in str(element.payload.get("html") or ""))
            for element in elements
        )
        has_fixed_grid_tracks = any(
            section.kind == FlowKind.GRID
            and any(self._is_floating_figure_cell(cell, by_id) for cell in section.grid_cells)
            for section in sections
        )
        safety_factor = (
            min(
                self.word_safety_factor,
                self.GRID_WORD_SAFETY_FACTOR,
                self.FORMULA_WORD_SAFETY_FACTOR if has_formulas else 1.0,
            )
            if has_fixed_grid_tracks
            else min(self.word_safety_factor, self.FORMULA_WORD_SAFETY_FACTOR if has_formulas else 1.0)
        )
        # Word/LibreOffice section and table boxes need a small allowance beyond paragraph math.
        render_overhead = min(self.PAGE_RENDER_OVERHEAD_PT, usable_height * 0.015)
        target = max(usable_height * safety_factor - render_overhead, 1.0)
        section_overhead = sum(section.kind != FlowKind.SINGLE for section in sections) * 0.05
        section_overhead += sum(
            self.WRAPPED_SECTION_OVERHEAD_PT
            for section in sections
            if section.kind == FlowKind.WRAPPED
        )

        def content_height(scale: float, gap_scale: float) -> float:
            adjusted = self._with_compressed_gaps(elements, gap_scale)
            return section_overhead + sum(
                self._section_height(section, adjusted, roles, usable_width, scale) for section in sections
            )

        if content_height(1.0, 1.0) <= target:
            return 1.0, 1.0

        if content_height(1.0, self.MIN_GAP_SCALE) <= target:
            lower, upper = self.MIN_GAP_SCALE, 1.0
            for _ in range(10):
                middle = (lower + upper) / 2.0
                if content_height(1.0, middle) <= target:
                    lower = middle
                else:
                    upper = middle
            return 1.0, lower

        minimum_scale = self._minimum_page_scale(elements, roles)
        physical_available = max(usable_height, 1.0)
        if content_height(minimum_scale, self.MIN_GAP_SCALE) > physical_available + self.PAGE_HEIGHT_TOLERANCE_PT:
            available = physical_available
            required = content_height(minimum_scale, self.MIN_GAP_SCALE)
            raise LayoutFitError(
                f"page {page_index if page_index is not None else '?'} cannot fit within readable limits: "
                f"required={required:.2f}pt available={available:.2f}pt min_scale={minimum_scale:.3f}"
            )
        lower, upper = minimum_scale, 1.0
        for _ in range(12):
            middle = (lower + upper) / 2.0
            if content_height(middle, self.MIN_GAP_SCALE) <= target:
                lower = middle
            else:
                upper = middle
        return max(lower, minimum_scale), self.MIN_GAP_SCALE

    def _minimum_page_scale(self, elements, roles) -> float:
        minimum = self.MIN_PAGE_SCALE
        for element in elements:
            role = roles.get(element.role_id)
            if role is None or not element.text:
                continue
            if element.kind == "heading" or str(role.role_id).startswith("heading"):
                minimum_font = self.MIN_HEADING_FONT_PT
            elif element.kind == "caption" or str(role.role_id).startswith("caption"):
                minimum_font = self.MIN_CAPTION_FONT_PT
            else:
                minimum_font = self.MIN_BODY_FONT_PT
            minimum = max(minimum, minimum_font / max(float(role.font_size_pt), 0.5))
        return min(minimum, 1.0)

    @classmethod
    def _with_compressed_gaps(cls, elements, gap_scale: float):
        if gap_scale >= 1.0:
            return elements
        adjusted = []
        for element in elements:
            original = float(element.payload.get("space_before_pt", 0.0))
            if original <= 0:
                adjusted.append(element)
                continue
            floor = min(original, cls.MIN_GAP_PT)
            if element.kind == "caption":
                floor = max(floor, min(original, 12.0))
            compressed = max(original * gap_scale, floor)
            adjusted.append(
                replace(
                    element,
                    payload={**element.payload, "space_before_pt": compressed},
                )
            )
        return tuple(adjusted)

    @staticmethod
    def _is_floating_figure_cell(cell, elements) -> bool:
        return (
            cell.row_span > 1
            and len(cell.element_ids) == 1
            and elements[cell.element_ids[0]].kind == "figure_group"
        )

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
                    if floating.kind == "figure_group"
                    and float((by_id[identifier].payload.get("source_bbox") or (0, 0, 1, 1))[1])
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
                (section.floating_offset_y_pt + section.floating_width_pt * aspect) * fit_scale
                + self._caption_height(floating, fit_scale)
            )
            source_height = section.row_heights_pt[0] * fit_scale if section.row_heights_pt else 0.0
            if source_height:
                text_height -= sum(
                    float(by_id[identifier].payload.get("space_before_pt", 0.0)) * fit_scale
                    for identifier in section.element_ids
                    if identifier != section.floating_element_id
                )
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
        tracks = self._grid_row_heights(section, elements, roles, fit_scale)
        source_height = sum(section.row_heights_pt) * fit_scale
        # ponytail: only use source tracks after the font floor exposes overlap double-counting.
        return (
            source_height
            if fit_scale <= 0.05 and source_height and sum(tracks) > source_height * 1.5
            else sum(tracks)
        )

    def _grid_row_heights(self, section, elements, roles, fit_scale: float) -> tuple[float, ...]:
        by_id = {element.element_id: element for element in elements}
        widths = section.column_widths_pt
        row_count = max(cell.row + cell.row_span for cell in section.grid_cells)
        constraints = defaultdict(list)
        for cell in section.grid_cells:
            width = sum(widths[cell.column : cell.column + cell.column_span]) + section.gutter_pt * (cell.column_span - 1)
            height = sum(
                self._element_height(by_id[identifier], roles, width, fit_scale)
                for identifier in cell.element_ids
            )
            constraints[cell.row].append((cell.row + cell.row_span, height))
        boundaries = [0.0] * (row_count + 1)
        for start in range(row_count):
            boundaries[start + 1] = max(boundaries[start + 1], boundaries[start] + 0.01)
            for end, height in constraints[start]:
                boundaries[end] = max(boundaries[end], boundaries[start] + height)
        return tuple(right - left for left, right in zip(boundaries, boundaries[1:]))

    @staticmethod
    def _resolve_text_layouts(sections, elements, roles, fit_scale: float, usable_width: float):
        by_id = {element.element_id: element for element in elements}
        widths = {element.element_id: usable_width for element in elements}
        for section in sections:
            if section.kind == FlowKind.SEQUENTIAL_COLUMNS:
                for identifier in section.element_ids:
                    column = int(by_id[identifier].payload.get("column", 0))
                    widths[identifier] = section.column_widths_pt[column]
            elif section.kind == FlowKind.GRID:
                for cell in section.grid_cells:
                    width = (
                        sum(section.column_widths_pt[cell.column : cell.column + cell.column_span])
                        + section.gutter_pt * (cell.column_span - 1)
                    )
                    widths.update((identifier, width) for identifier in cell.element_ids)
            elif section.kind == FlowKind.WRAPPED:
                floating = by_id[section.floating_element_id]
                floating_bbox = floating.payload.get("source_bbox") or (0, 0, 1, 1)
                wrap_width = max(usable_width - section.floating_width_pt - section.gutter_pt, 1.0)
                widths[section.floating_element_id] = section.floating_width_pt
                for identifier in section.element_ids:
                    if identifier == section.floating_element_id:
                        continue
                    bbox = by_id[identifier].payload.get("source_bbox") or (0, 0, 1, 1)
                    widths[identifier] = wrap_width if float(bbox[1]) < float(floating_bbox[3]) else usable_width
        return tuple(
            replace(
                element,
                text_layout=resolve_text_layout(
                    element,
                    roles.get(element.role_id),
                    widths[element.element_id],
                    fit_scale,
                ),
            )
            for element in elements
        )

    def _with_vertical_tracks(self, section, elements, source_scale: float):
        if section.kind == FlowKind.SINGLE:
            return section
        by_id = {element.element_id: element for element in elements}
        boxes = sorted((by_id[identifier].bbox for identifier in section.element_ids), key=lambda box: box.y1)
        if section.kind != FlowKind.GRID:
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
            return section
        boundaries = starts + [bottom]
        return replace(
            section,
            row_heights_pt=tuple((boundaries[index + 1] - boundaries[index]) * source_scale for index in range(row_count)),
        )

    @staticmethod
    def _element_height(element, roles, width: float, fit_scale: float) -> float:
        role = roles.get(element.role_id)
        spacing = float(element.payload.get("space_before_pt", 0.0)) * fit_scale
        spacing += float(element.payload.get("column_offset_pt", 0.0)) * fit_scale
        if element.kind == "table_group" and element.payload.get("html"):
            soup = BeautifulSoup(str(element.payload["html"]), "html.parser")
            table = soup.find("table")
            rows = table.find_all("tr") if table else []
            weights = get_table_column_weights(table, element.payload.get("table_column_width_ratios") or ()) if table else (1.0,)
            base_size = float(element.payload.get("table_font_size_pt") or (role.font_size_pt if role else 10.5))
            font_size = max(
                round(base_size * fit_scale * 2) / 2.0,
                float(element.payload.get("table_min_font_size_pt", 0.5)),
            )
            placements = get_table_cell_placements(table) if table else []
            table_width = width * float(element.payload.get("width_fraction", 1.0))
            cell_geometry = {
                (int(row), int(column)): (vertical, top, bottom, start, end)
                for row, column, vertical, top, bottom, start, end in element.payload.get("table_cell_geometry", ())
            }
            minimum_size = float(element.payload.get("table_min_font_size_pt", 0.5))
            font_size = fit_table_font_size(
                table,
                font_size,
                table_width,
                weights,
                cell_geometry,
                minimum_size,
            )
            source_height = float(element.payload.get("table_height_pt") or 0.0) * fit_scale
            row_ratios = element.payload.get("table_row_height_ratios") or ()
            if len(row_ratios) == len(rows):
                source_rows = [source_height * float(ratio) for ratio in row_ratios]
            else:
                source_row = 0.0 if element.payload.get("table_content_fit") else source_height / max(len(rows), 1)
                source_rows = [source_row] * len(rows)
            row_heights = [max(height, font_size * 1.2 + 3.0) for height in source_rows]
            required_rows = [font_size * 1.2 + 3.0 for _ in range(len(rows))]
            visual_rows = [False for _ in rows]
            for row_index, column, row_span, span, cell in placements:
                cell_width = table_width * sum(weights[column : column + span]) / max(sum(weights), 1.0)
                geometry = cell_geometry.get((row_index, column), ("center", 0, 0, 0, 0))
                available_width = table_cell_available_width(cell_width, geometry[3], geometry[4])
                lines = max(1, estimate_table_cell_lines(cell, font_size, available_width))
                required = (lines * font_size * 1.2 + 3.0) / max(row_span, 1)
                for target_row in range(row_index, min(row_index + row_span, len(rows))):
                    required_rows[target_row] = max(required_rows[target_row], required)
                    visual_rows[target_row] = visual_rows[target_row] or bool(cell.find("img"))
                    row_heights[target_row] = max(row_heights[target_row], required)
            page_height = float(element.payload.get("page_height_px") or 0.0) * float(
                element.payload.get("source_scale") or 1.0
            )
            compact_text_rows = bool(element.payload.get("table_auto_height")) or (
                page_height > 0.0 and source_height > page_height * 0.45
            )
            if compact_text_rows:
                for row in range(len(rows)):
                    row_heights[row] = (
                        max(row_heights[row], required_rows[row])
                        if visual_rows[row]
                        else required_rows[row]
                    )
            height = sum(row_heights)
            return spacing + height + ReflowPlanner._caption_height(element, fit_scale)
        if role is not None and element.text:
            layouts = resolve_text_layout(element, role, width, fit_scale)
            return sum(layout.measured_height_pt for layout in layouts)
        bbox = element.payload.get("primary_bbox") or element.payload.get("source_bbox") or (0, 0, 1, 1)
        aspect = max(float(bbox[3]) - float(bbox[1]), 1.0) / max(float(bbox[2]) - float(bbox[0]), 1.0)
        source_width = (float(bbox[2]) - float(bbox[0])) * float(element.payload.get("source_scale", 1.0))
        visual_width = min(width * float(element.payload.get("width_fraction", 1.0)), source_width) * (
            1.0 if element.kind == "figure_group" else fit_scale
        )
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
        content_bbox = ReflowPlanner._content_bbox(element)
        layout_bbox = ReflowPlanner._layout_bbox(element)
        element_center = (content_bbox.x1 + content_bbox.x2) / 2.0
        centered = abs(element_center - (left + right) / 2.0) <= width * 0.04
        source_lines = tuple(row.text for row in element.text_rows) or element.payload.get("lines") or ()
        if element.kind == "paragraph_group" and element.text_structure.is_list:
            return "left"
        if (
            element.kind == "paragraph_group"
            and content_bbox.x1 - layout_bbox.x1 >= layout_bbox.width * 0.03
        ):
            return "left"
        widths = tuple(row.bbox.width for row in element.text_rows) or element.payload.get("line_widths_px") or ()
        lefts = tuple(row.bbox.x1 for row in element.text_rows) or element.payload.get("line_lefts_px") or ()
        if (
            element.kind == "paragraph_group"
            and element.text_structure.preserve_source_lines
            and len(source_lines) >= 2
            and len(lefts) == len(widths) == len(source_lines)
            and all(
                abs(float(source_left) + float(line_width) / 2.0 - (left + right) / 2.0) <= width * 0.025
                for source_left, line_width in zip(lefts, widths)
            )
        ):
            return "center"
        if (
            content_bbox.width <= width * 0.65
            and len(source_lines) >= 2
            and len(lefts) == len(widths) == len(source_lines)
            and all(
                abs(float(source_left) + float(line_width) / 2.0 - (left + right) / 2.0) <= width * 0.05
                for source_left, line_width in zip(lefts, widths)
            )
        ):
            return "center"
        if element.kind == "heading" and centered and content_bbox.width <= width * 0.95:
            return "center"
        if element.kind == "heading" and len(source_lines) > 1 and content_bbox.x1 <= left + width * 0.05:
            return "left"
        if element.kind == "paragraph_group":
            line_rights = tuple(
                float(source_left) + float(line_width)
                for source_left, line_width in zip(lefts, widths)
            )
            if len(line_rights) >= 3:
                right_edge = max(line_rights)
                body_lines = line_rights[:-1]
                aligned_lines = sum(
                    right >= right_edge - width * 0.03 for right in body_lines
                )
                justified = aligned_lines >= max(2, math.ceil(len(body_lines) * 0.60))
            elif not line_rights:
                justified = len(source_lines) >= 3 and len(element.text) >= 40
            else:
                justified = False
            if len(source_lines) == 1 and centered and content_bbox.width <= width * 0.65:
                return "center"
            if (
                len(source_lines) >= 2
                and len(lefts) == len(widths) == len(source_lines)
                and max(float(line_width) for line_width in widths) <= width * 0.65
                and all(
                    abs(float(source_left) + float(line_width) / 2.0 - (left + right) / 2.0) <= width * 0.04
                    for source_left, line_width in zip(lefts, widths)
                )
            ):
                return "center"
            if content_bbox.x2 >= right - width * 0.03 and content_bbox.x1 > left + width * 0.25 and len(source_lines) <= 2:
                return "right"
            return "justify" if justified and len(element.text) >= 40 else "left"
        if content_bbox.x2 >= right - width * 0.03 and content_bbox.x1 > left + width * 0.25 and len(source_lines) <= 2:
            return "right"
        return "justify" if element.kind == "paragraph_group" and len(source_lines) >= 3 and len(element.text) >= 40 else "left"

    @staticmethod
    def _is_edge_text_label(element, page) -> bool:
        if element.kind not in {"paragraph_group", "heading", "caption"}:
            return False
        if element.text_structure.orientation != "vertical":
            return False
        width = element.bbox.width
        height = element.bbox.height
        near_left = element.bbox.x2 <= page.width_px * 0.12
        near_right = element.bbox.x1 >= page.width_px * 0.88
        return width <= page.width_px * 0.09 and height >= width * 2.5 and (near_left or near_right)

    @staticmethod
    def _row_alignments(element, frame) -> tuple[str, ...]:
        if len(element.text_rows) < 2:
            return ()
        if element.kind == "paragraph_group" and element.text_structure.is_list:
            return ("left",) * len(element.text_rows)
        left, right = frame
        frame_width = max(right - left, 1.0)
        inferred = []
        for row in element.text_rows:
            line_left = row.bbox.x1
            line_width = row.bbox.width
            line_right = line_left + line_width
            centered = abs(line_left + line_width / 2.0 - (left + right) / 2.0) <= frame_width * 0.04
            if line_width <= frame_width * 0.92 and centered:
                inferred.append("center")
            elif line_right >= right - frame_width * 0.04 and line_left > left + frame_width * 0.15:
                inferred.append("right")
            elif line_left <= left + frame_width * 0.04 and line_right < right - frame_width * 0.15:
                inferred.append("left")
            else:
                return ()
        return tuple(inferred) if len(set(inferred)) > 1 else ()

    @staticmethod
    def _first_line_indent(element, source_scale: float) -> float:
        lefts = tuple(row.bbox.x1 for row in element.text_rows) or element.payload.get("line_lefts_px") or ()
        if len(lefts) < 2:
            return 0.0
        indent = float(lefts[0]) - median(float(value) for value in lefts[1:])
        return max(indent * source_scale, 0.0)

    @staticmethod
    def _heading_hanging_indent(element, source_scale: float) -> float:
        lefts = tuple(row.bbox.x1 for row in element.text_rows) or element.payload.get("line_lefts_px") or ()
        if element.kind != "heading" or len(lefts) < 2:
            return 0.0
        return max(median(float(value) for value in lefts[1:]) - float(lefts[0]), 0.0) * source_scale

    @staticmethod
    def _source_line_height(element, role, source_scale: float):
        lines = element.text_rows or element.payload.get("lines") or ()
        if not lines or role is None:
            return None
        line_heights = (
            tuple(row.ink_height_px or row.bbox.height for row in element.text_rows)
            or element.payload.get("line_heights_px")
            or ()
        )
        line_pitches = (
            tuple(
                right.bbox.y1 - left.bbox.y1
                for left, right in zip(element.text_rows, element.text_rows[1:])
            )
            or element.payload.get("line_pitches_px")
            or ()
        )
        if line_pitches:
            observed = median(line_pitches) * source_scale
        else:
            observed = (
                median(line_heights) * source_scale
                if line_heights
                else ReflowPlanner._content_bbox(element).height * source_scale / len(lines)
            )
        return min(max(observed, role.font_size_pt * 1.05), role.font_size_pt * 1.5)

    @staticmethod
    def _table_font_size(element, source_scale: float, body_font_size: float):
        if element.kind != "table_group" or not element.payload.get("html"):
            return None
        line_heights = element.payload.get("table_line_heights_px") or ()
        if line_heights:
            return min(median(float(height) for height in line_heights) * source_scale, body_font_size)
        table = BeautifulSoup(str(element.payload["html"]), "html.parser").find("table")
        row_count = len(table.find_all("tr")) if table else 0
        if row_count == 0:
            return None
        return min(ReflowPlanner._layout_bbox(element).height * source_scale / row_count / 1.45, body_font_size)

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
