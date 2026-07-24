"""Page-constrained document reflow from immutable recognition evidence."""

from docflow.analysis import DocumentAnalyzer
from docflow.planning import ReflowPlanner
from docflow.renderer import ReflowDocxRenderer, ReflowMarkdownRenderer

__version__ = "0.6.0"
__all__ = [
    "DocumentAnalyzer",
    "ReflowPlanner",
    "ReflowDocxRenderer",
    "ReflowMarkdownRenderer",
    "__version__",
]
