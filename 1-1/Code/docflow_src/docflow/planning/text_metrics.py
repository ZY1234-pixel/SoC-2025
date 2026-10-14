"""Shared text geometry estimates for planning and document emission."""

from __future__ import annotations

import math
import json
import re
from functools import lru_cache
from pathlib import Path
from statistics import median

from docflow.model.stages import Rect, TextParagraphLayout


_TOKEN_RE = re.compile(r"\s+|[A-Za-z0-9]+(?:[._:/%+#-][A-Za-z0-9]+)*|[\u2e80-\u9fff]|.")
_FONT_METRICS_PATH = Path(__file__).with_name("font_metrics.json")


def estimate_text_units(text: str) -> float:
    return sum(1.0 if ord(char) >= 0x2E80 else 0.42 for char in text)


def fit_font_size_to_lines(
    font_size_pt: float,
    lines: tuple[str, ...],
    widths_pt: tuple[float, ...],
    occupancy: float,
    font_family: str | None = None,
    western_font_family: str | None = None,
    bold: bool = False,
) -> float:
    limits = [
        width * occupancy / max(_font_width_per_point(line, font_family, western_font_family, bold), 0.1)
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
    font_family: str | None = None,
    western_font_family: str | None = None,
    bold: bool = False,
) -> int:
    measured_lines = _wrapped_line_count(
        text,
        font_size_pt,
        width_pt,
        font_family,
        western_font_family,
        bold,
    )
    content_lines = max(1, measured_lines)
    if not source_line_count or not source_width_pt:
        return content_lines
    observed_lines = max(
        1,
        round(source_line_count * min(source_width_pt * fit_scale / max(width_pt, 1.0), 1.0)),
    )
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


def fit_text_width(
    text: str,
    font_size_pt: float,
    target_width_pt: float,
    font_family: str | None = None,
    western_font_family: str | None = None,
    bold: bool = False,
) -> tuple[int, float]:
    """Fit a single line without changing its font size."""
    natural_width = measure_text_width_pt(
        text,
        font_size_pt,
        font_family,
        western_font_family,
        bold,
    )
    if natural_width <= 0 or target_width_pt <= 0:
        return 100, 0.0
    ratio = target_width_pt / natural_width
    if not 0.90 <= ratio <= 1.10:
        return 100, 0.0
    scale_pct = max(90, min(110, round(ratio * 100)))
    scaled_width = natural_width * scale_pct / 100.0
    spacing_pt = (target_width_pt - scaled_width) / max(len(text), 1)
    return scale_pct, max(-2.0, min(2.0, spacing_pt))


def measure_text_width_pt(
    text: str,
    font_size_pt: float,
    font_family: str | None = None,
    western_font_family: str | None = None,
    bold: bool = False,
) -> float:
    """Measure text width from the committed fixed font metrics."""
    value = str(text or "")
    if not value:
        return 0.0
    return sum(_character_width_em(character, font_family, western_font_family) for character in value) * float(
        font_size_pt
    )


def _font_width_per_point(
    text: str,
    font_family: str | None,
    western_font_family: str | None,
    bold: bool,
) -> float:
    return measure_text_width_pt(text, 1.0, font_family, western_font_family, bold)


def _wrapped_line_count(
    text: str,
    font_size_pt: float,
    width_pt: float,
    font_family: str | None,
    western_font_family: str | None,
    bold: bool,
) -> int:
    width = max(float(width_pt), 1.0)
    count = 0
    for paragraph in str(text or "").split("\n"):
        if not paragraph:
            count += 1
            continue
        line_width = 0.0
        lines = 1
        for token in _TOKEN_RE.findall(paragraph):
            token_width = measure_text_width_pt(
                token,
                font_size_pt,
                font_family,
                western_font_family,
                bold,
            )
            if token.isspace():
                if line_width > 0:
                    line_width += token_width
                continue
            if line_width > 0 and line_width + token_width > width:
                lines += 1
                line_width = 0.0
            if token_width <= width:
                line_width += token_width
                continue
            for character in token:
                character_width = measure_text_width_pt(
                    character,
                    font_size_pt,
                    font_family,
                    western_font_family,
                    bold,
                )
                if line_width > 0 and line_width + character_width > width:
                    lines += 1
                    line_width = 0.0
                line_width += character_width
        count += lines
    return max(count, 1)


@lru_cache(maxsize=1)
def _font_metrics_data() -> dict:
    return json.loads(_FONT_METRICS_PATH.read_text(encoding="utf-8"))


def _character_width_em(
    character: str,
    font_family: str | None,
    western_font_family: str | None,
) -> float:
    metrics = _font_metrics_data()
    aliases = metrics.get("aliases") or {}
    cjk = _is_cjk(character)
    requested = font_family if cjk else (western_font_family or "Times New Roman")
    name = aliases.get(str(requested or ""), str(requested or ""))
    if name not in metrics.get("fonts", {}):
        name = "宋体" if cjk else "Times New Roman"
    font = metrics["fonts"][name]
    return float(
        (font.get("codepoint_widths") or {}).get(
            str(ord(character)),
            font.get("default_em", 1.0),
        )
    )


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x2E80 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0xFE30 <= codepoint <= 0xFE4F
        or 0xFF00 <= codepoint <= 0xFFEF
    )


def resolve_text_layout(element, role, container_width_pt: float, fit_scale: float) -> tuple[TextParagraphLayout, ...]:
    """Resolve all editable text geometry once, before DOCX emission."""
    if role is None or not element.text:
        return ()
    payload = element.payload
    source_lines = tuple(row.text for row in element.text_rows) or tuple(
        str(line) for line in payload.get("lines") or ()
    )
    source_bbox = element.content_bbox or Rect.from_sequence(payload.get("source_bbox") or (0, 0, 1, 1))
    source_scale = float(payload.get("source_scale", 1.0))
    source_line_widths = tuple(
        max(float(row.bbox.width) * source_scale, 1.0)
        for row in element.text_rows
    )
    measure_lines = source_lines or (element.text,)
    measure_widths = (
        source_line_widths
        if len(source_line_widths) == len(measure_lines)
        else (container_width_pt,) * len(measure_lines)
    )
    base_font_size = max(round(role.font_size_pt * fit_scale * 2) / 2.0, 0.5)
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

    line_specific_layout = bool(split_rows or row_alignments)
    heading_fit_lines = element.kind == "heading"
    base_font_size_before_fit = base_font_size
    allow_line_fit = (
        element.kind in {"paragraph_group", "heading"}
        and element.text_structure.orientation != "vertical"
        and (element.text_structure.preserve_source_lines or line_specific_layout or heading_fit_lines)
    )
    if (
        not split_rows
        and element.text_structure.orientation != "vertical"
        and (element.text_structure.preserve_source_lines or line_specific_layout or heading_fit_lines)
        and not (
            element.kind == "paragraph_group"
            and line_specific_layout
            and not element.text_structure.preserve_source_lines
        )
    ):
        base_font_size = fit_font_size_to_lines(
            base_font_size,
            tuple(measure_lines),
            tuple(measure_widths),
            0.98,
            font_family=role.font_family,
            western_font_family=role.western_font_family,
            bold=role.bold,
        )

    layouts = []
    for index, (text, alignment, lines, preserve_lines, tabular, reset_indents) in enumerate(specs):
        left_indent = 0.0 if reset_indents else float(payload.get("left_indent_pt", 0.0))
        right_indent = 0.0 if reset_indents else float(payload.get("right_indent_pt", 0.0))
        first_indent = 0.0 if reset_indents else float(payload.get("first_line_indent_pt", 0.0)) * fit_scale
        width_fraction = 1.0 if reset_indents else float(payload.get("width_fraction", 1.0))
        if payload.get("background_color"):
            visual_width = container_width_pt * width_fraction
            right_indent = max(container_width_pt - left_indent - visual_width, 0.0)
        available_width = max(container_width_pt - left_indent - right_indent, 1.0)
        font_size = base_font_size
        if len(source_line_widths) == len(source_lines):
            if len(lines) == 1 and index < len(source_line_widths):
                target_widths = (source_line_widths[index],)
            elif len(lines) == len(source_line_widths):
                target_widths = source_line_widths
            else:
                target_widths = (median(source_line_widths),) * len(lines)
        else:
            target_widths = (available_width,) * len(lines)
        layout_target_width = (
            max(median(target_widths), 1.0)
            if target_widths
            else available_width
        )
        font_width_scale, character_spacing = (
            fit_text_width(
                text,
                font_size,
                layout_target_width,
                font_family=role.font_family,
                western_font_family=role.western_font_family,
                bold=role.bold,
            )
            if allow_line_fit and font_size >= role.font_size_pt * 0.90
            else (100, 0.0)
        )

        source_line_count = len(lines) if preserve_lines else int(payload.get("visual_line_count") or len(lines) or 1)
        content_bbox = element.content_bbox or Rect.from_sequence(payload.get("source_bbox") or (0, 0, 1, 1))
        source_width = content_bbox.width * float(payload.get("source_scale", 1.0))
        rendered_lines = (
            1
            if tabular
            else max(len(lines), 1)
            if preserve_lines
            else estimate_wrapped_lines(
                text,
                font_size * font_width_scale / 100.0,
                available_width,
                source_line_count,
                source_width,
                fit_scale,
                font_family=role.font_family,
                western_font_family=role.western_font_family,
                bold=role.bold,
            )
        )
        font_fit_ratio = base_font_size / max(base_font_size_before_fit, 0.5)
        measured_line_height = float(payload.get("line_height_pt") or font_size * role.line_spacing) * font_fit_ratio
        minimum_line_height = font_size * 1.05
        line_height = max(measured_line_height * fit_scale, minimum_line_height)
        cjk_count = sum(1 for char in text if ord(char) >= 0x2E80)
        reserve = min(font_size * 0.15, 2.0)
        if not payload.get("line_height_pt"):
            line_height = font_size * role.line_spacing * 1.05
            reserve = font_size if cjk_count else font_size / 4.0
        space_before = float(payload.get("space_before_pt", 0.0)) * fit_scale if index == 0 else 0.0
        layouts.append(
            TextParagraphLayout(
                text=text,
                alignment=alignment,
                font_size_pt=font_size,
                line_height_pt=line_height,
                rendered_line_count=rendered_lines,
                space_before_pt=space_before,
                first_line_indent_pt=first_indent,
                left_indent_pt=left_indent,
                right_indent_pt=right_indent,
                right_tab_stop_pt=(max(container_width_pt - right_indent, 1.0) if tabular else None),
                layout_reserve_pt=reserve,
                font_width_scale_pct=font_width_scale,
                character_spacing_pt=character_spacing,
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
