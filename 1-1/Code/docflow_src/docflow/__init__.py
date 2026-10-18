"""Page-constrained document reflow from immutable recognition evidence."""

from docflow.analysis import DocumentAnalyzer
from docflow.planning import ReflowPlanner
from docflow.renderer import (
    FidelityPdfRenderer,
    HtmlRenderer,
    PlainTextRenderer,
    ReflowDocxRenderer,
    ReflowMarkdownRenderer,
    RtfRenderer,
    XlsxRenderer,
)

__version__ = "0.6.0"
__all__ = [
    "DocumentAnalyzer",
    "FidelityPdfRenderer",
    "HtmlRenderer",
    "PlainTextRenderer",
    "ReflowPlanner",
    "ReflowDocxRenderer",
    "ReflowMarkdownRenderer",
    "RtfRenderer",
    "XlsxRenderer",
    "__version__",
]
