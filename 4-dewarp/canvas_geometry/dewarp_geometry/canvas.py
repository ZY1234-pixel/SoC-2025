"""Dynamic output-canvas utilities."""

from __future__ import annotations

import math


def canvas_from_aspect(
    aspect_ratio: float,
    *,
    long_edge: int = 1400,
    max_pixels: int | None = 8_000_000,
) -> tuple[int, int]:
    """Return ``(width, height)`` without assuming a paper standard."""
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be finite and positive")
    if long_edge < 2:
        raise ValueError("long_edge must be at least 2 pixels")
    if aspect_ratio >= 1:
        width, height = long_edge, max(1, round(long_edge / aspect_ratio))
    else:
        width, height = max(1, round(long_edge * aspect_ratio)), long_edge
    if max_pixels is not None and width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
    return width, height
