"""Page-constrained document reflow from immutable recognition evidence."""

from docflow.analysis import DocumentAnalyzer
from docflow.planning import ReflowPlanner
from docflow.renderer import (
    FidelityPdfRenderer,
    ReflowDocxRenderer,
    ReflowMarkdownRenderer,
    ReflowPdfRenderer,
)

__version__ = "0.6.0"
__all__ = [
    "DocumentAnalyzer",
    "FidelityPdfRenderer",
    "ReflowPlanner",
    "ReflowDocxRenderer",
    "ReflowMarkdownRenderer",
    "ReflowPdfRenderer",
    "__version__",
]
