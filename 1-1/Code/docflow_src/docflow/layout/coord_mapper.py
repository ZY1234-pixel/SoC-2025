"""向后兼容的重导出。

The canonical :class:`CoordMapper` now lives in
:mod:`docflow.model.page`.  This module re-exports it so that
existing imports continue to work.
"""

from docflow.model.page import CoordMapper  # noqa: F401
