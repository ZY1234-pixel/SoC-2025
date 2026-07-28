"""Shared text geometry estimates for planning and document emission."""

from __future__ import annotations

import math


def estimate_text_units(text: str) -> float:
    return sum(1.0 if ord(char) >= 0x2E80 else 0.42 for char in text)


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
