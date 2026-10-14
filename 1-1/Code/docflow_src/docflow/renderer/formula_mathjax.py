"""Offline LaTeX-to-SVG rendering through MathJax and QuickJS."""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import fitz


class MathJaxFormulaRenderer:
    """Small runtime wrapper around the prebuilt headless MathJax bundle."""

    _asset = Path(__file__).resolve().parents[3] / "assets" / "mathjax-tex-svg-lite.js"
    _lock = threading.Lock()
    _context = None

    @classmethod
    def render_pdf(cls, latex: str) -> bytes | None:
        value = str(latex or "").strip()
        if not value:
            return None
        return cls._render_pdf(value)

    @classmethod
    @lru_cache(maxsize=512)
    def _render_pdf(cls, latex: str) -> bytes | None:
        svg = cls._render_svg(latex)
        if not svg:
            return None
        try:
            with fitz.open(stream=svg.encode("utf-8"), filetype="svg") as document:
                return bytes(document.convert_to_pdf())
        except Exception:
            return None

    @classmethod
    def _render_svg(cls, latex: str) -> str | None:
        context = cls._get_context()
        if context is None:
            return None
        try:
            with cls._lock:
                result = context.eval(f"mathjaxRender({json.dumps(latex)})")
            if not isinstance(result, str) or 'data-mml-node="merror"' in result:
                return None
            # PyMuPDF turns stroke-width="0" into a hairline PDF stroke, which
            # makes the glyphs look bolder than the surrounding text.
            return result.replace(
                'stroke="currentColor" fill="currentColor" stroke-width="0"',
                'stroke="none" fill="currentColor"',
            )
        except Exception:
            return None

    @classmethod
    def _get_context(cls):
        if cls._context is not None:
            return cls._context
        with cls._lock:
            if cls._context is not None:
                return cls._context
            try:
                import quickjs

                if not cls._asset.is_file():
                    return None
                context = quickjs.Context()
                context.eval(cls._asset.read_text(encoding="utf-8"))
                cls._context = context
            except Exception:
                return None
        return cls._context
