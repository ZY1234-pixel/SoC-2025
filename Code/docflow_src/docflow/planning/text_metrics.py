"""Shared text geometry estimates for planning and document emission."""

from __future__ import annotations

import math

from docflow.model.stages import Rect, TextParagraphLayout


def estimate_text_units(text: str) -> float:
    return sum(1.0 if ord(char) >= 0x2E80 else 0.42 for char in text)


def fit_font_size_to_lines(
    font_size_pt: float,
    lines: tuple[str, ...],
    widths_pt: tuple[float, ...],
    occupancy: float,
) -> float:
    limits = [
        width * occupancy / max(estimate_text_units(line), 1.0)
        for line, width in zip(lines, widths_pt)
        if line
    ]
    return max(math.floor(min([font_size_pt, *limits]) * 2.0) / 2.0, 0.5)


def estimate_wrapped_lines(
    text: str,
    font_size_pt: float,
    width_pt: float,
    source_line_count: int = 0,
    source_width_pt: float = 0.0,
    fit_scale: float = 1.0,
) -> int:
    content_lines = max(1, math.ceil(estimate_text_units(text) * font_size_pt / max(width_pt, 1.0)))
    if not source_line_count or not source_width_pt:
        return content_lines
    observed_lines = max(1, round(source_line_count * source_width_pt / max(width_pt, 1.0) * fit_scale))
    return max(content_lines, observed_lines)


def infer_occupancy_line_height(
    font_size_pt: float,
    measured_line_height_pt: float,
    target_height_pt: float,
    rendered_lines: int,
) -> float:
    return min(
        max(measured_line_height_pt, target_height_pt / max(rendered_lines, 1), font_size_pt * 1.05),
        font_size_pt * 1.5,
    )


def resolve_text_layout(element, role, container_width_pt: float, fit_scale: float) -> tuple[TextParagraphLayout, ...]:
    """Resolve all editable text geometry once, before DOCX emission."""
    if role is None or not element.text:
        return ()
    payload = element.payload
    source_lines = tuple(row.text for row in element.text_rows) or tuple(
        str(line) for line in payload.get("lines") or ()
    )
    split_rows = (
        tuple(tuple(span.text for span in row.spans) for row in element.text_rows)
        if element.text_structure.tabular_rows
        else ()
    )
    row_alignments = element.row_alignments
    if split_rows:
        specs = [
            ("\t".join(str(value) for value in row), "left", tuple(str(value) for value in row), True, True, False)
            for row in split_rows
        ]
    elif len(row_alignments) == len(source_lines) >= 2:
        specs = [
            (line, str(alignment), (line,), True, False, True)
            for line, alignment in zip(source_lines, row_alignments)
        ]
    else:
        specs = [(
            visual_text(element, source_lines),
            "left" if element.text_structure.is_list else str(payload.get("alignment") or "left"),
            source_lines,
            element.text_structure.preserve_source_lines,
            False,
            False,
        )]

    layouts = []
    for index, (text, alignment, lines, preserve_lines, tabular, reset_indents) in enumerate(specs):
        left_indent = 0.0 if reset_indents else float(payload.get("left_indent_pt", 0.0))
        right_indent = 0.0 if reset_indents else float(payload.get("right_indent_pt", 0.0))
        first_indent = 0.0 if reset_indents else float(payload.get("first_line_indent_pt", 0.0)) * fit_scale
        width_fraction = 1.0 if reset_indents else float(payload.get("width_fraction", 1.0))
        if payload.get("background_color"):
            visual_width = container_width_pt * width_fraction * fit_scale
            right_indent = max(container_width_pt - left_indent - visual_width, 0.0)
        available_width = max(container_width_pt - left_indent - right_indent, 1.0)
        font_size = max(round(role.font_size_pt * fit_scale * 2) / 2.0, 0.5)
        if len(lines) > 1 and not tabular:
            line_widths = (max(available_width - first_indent, 1.0),) + (available_width,) * (len(lines) - 1)
            font_size = fit_font_size_to_lines(
                font_size,
                lines,
                line_widths,
                0.90 if element.kind == "heading" or preserve_lines else 0.99,
            )
        elif len(lines) == 1 and element.text_structure.orientation != "vertical" and not tabular:
            font_size = min(
                font_size,
                container_width_pt * width_fraction * 0.90 / max(estimate_text_units(text), 1.0),
            )

        source_line_count = len(lines) if preserve_lines else int(payload.get("visual_line_count") or len(lines) or 1)
        content_bbox = element.content_bbox or Rect.from_sequence(payload.get("source_bbox") or (0, 0, 1, 1))
        source_width = content_bbox.width * float(payload.get("source_scale", 1.0))
        rendered_lines = (
            1
            if tabular
            else max(len(lines), 1)
            if preserve_lines
            else estimate_wrapped_lines(text, font_size, available_width, source_line_count, source_width, fit_scale)
        )
        measured_line_height = float(payload.get("line_height_pt") or font_size * role.line_spacing)
        line_height = max(measured_line_height * fit_scale, font_size * 1.05)
        occupancy_fitted = (
            element.kind == "paragraph_group"
            and not preserve_lines
            and not tabular
            and element.text_structure.orientation == "horizontal"
            and bool(payload.get("line_height_pt"))
        )
        if occupancy_fitted:
            line_height = infer_occupancy_line_height(
                font_size,
                line_height,
                content_bbox.height * float(payload.get("source_scale", 1.0)) * fit_scale,
                max(rendered_lines, source_line_count),
            )
            rendered_lines = max(rendered_lines, source_line_count)
        cjk_count = sum(1 for char in text if ord(char) >= 0x2E80)
        reserve = 0.0 if occupancy_fitted else min(font_size * 0.15, 2.0)
        if not payload.get("line_height_pt"):
            line_height = font_size * role.line_spacing * 1.05
            reserve = font_size if cjk_count else font_size / 4.0
        layouts.append(
            TextParagraphLayout(
                text=text,
                alignment=alignment,
                font_size_pt=font_size,
                line_height_pt=line_height,
                rendered_line_count=rendered_lines,
                space_before_pt=(float(payload.get("space_before_pt", 0.0)) * fit_scale if index == 0 else 0.0),
                first_line_indent_pt=first_indent,
                left_indent_pt=left_indent,
                right_indent_pt=right_indent,
                right_tab_stop_pt=(max(container_width_pt - right_indent, 1.0) if tabular else None),
                layout_reserve_pt=reserve,
            )
        )
    return tuple(layouts)


def visual_text(element, lines: tuple[str, ...] | None = None) -> str:
    lines = lines if lines is not None else (
        tuple(row.text for row in element.text_rows)
        or tuple(str(line) for line in element.payload.get("lines") or ())
    )
    if element.text_structure.orientation == "vertical":
        return "\n".join(character for character in element.text if not character.isspace())
    if element.text_structure.preserve_source_lines:
        return "\n".join(lines)
    if element.kind == "heading" and element.text_rows:
        return "\n".join(lines)
    tops = element.payload.get("line_tops_px") or ()
    heights = element.payload.get("line_heights_px") or ()
    if element.kind != "heading" or len(lines) < 2 or len(tops) != len(lines) or len(heights) != len(lines):
        return element.text
    output = lines[0]
    row_bottom = float(tops[0]) + float(heights[0])
    for line, top, height in zip(lines[1:], tops[1:], heights[1:]):
        output += ("\n" if float(top) >= row_bottom - min(float(height), row_bottom - float(tops[0])) * 0.10 else " ") + line
        row_bottom = max(row_bottom, float(top) + float(height))
    return output
