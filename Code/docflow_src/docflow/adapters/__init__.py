"""Recognition-engine adapters."""

from docflow.adapters.base_adapter import BaseAdapter

try:
    from docflow.adapters.paddle_adapter import PaddleAdapter
except ImportError:
    PaddleAdapter = None  # type: ignore[assignment,misc]

__all__ = ["BaseAdapter", "PaddleAdapter"]
