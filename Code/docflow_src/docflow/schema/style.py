"""样式解析工具。

实现三级样式继承链：

    page.style_defaults  →  block.style  →  text_line.style

每一级覆盖前一级；``None`` 值会被跳过（继承）。
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Optional

from docflow.schema.models import (
    BlockStyle,
    PageStyleDefaults,
    TextLineStyle,
)


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """浅合并：当值不为 None 时 *override* 优先。"""
    merged = dict(base)
    for k, v in override.items():
        if v is not None:
            merged[k] = v
    return merged


def _dataclass_to_dict(obj) -> Dict[str, Any]:
    """将 dataclass（或普通 dict）转为 dict，仅保留非 None 的值。"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if v is not None}
    return {f.name: getattr(obj, f.name) for f in fields(obj)
            if getattr(obj, f.name) is not None}


def resolve_block_style(
    page_defaults: Optional[PageStyleDefaults] = None,
    block_style: Optional[BlockStyle] = None,
    config_defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """解析区块的有效样式。

    优先级（高 → 低）：block_style > page_defaults > config_defaults
    """
    base = dict(config_defaults or {})
    base = _merge(base, _dataclass_to_dict(page_defaults))
    base = _merge(base, _dataclass_to_dict(block_style))
    return base


def resolve_textline_style(
    page_defaults: Optional[PageStyleDefaults] = None,
    block_style: Optional[BlockStyle] = None,
    line_style: Optional[TextLineStyle] = None,
    config_defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """解析文本行的有效样式。

    优先级（高 → 低）：line_style > block_style > page_defaults > config_defaults
    """
    base = dict(config_defaults or {})
    base = _merge(base, _dataclass_to_dict(page_defaults))
    base = _merge(base, _dataclass_to_dict(block_style))
    base = _merge(base, _dataclass_to_dict(line_style))
    return base
