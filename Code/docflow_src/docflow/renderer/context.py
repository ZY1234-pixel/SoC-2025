"""贯穿渲染管线的渲染上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class RenderContext:
    """贯穿渲染调用的可变上下文。

    承载坐标映射、页面状态和列边界等每个区块渲染方法所需的信息。
    """

    coord_mapper: Any
    page: Any
    col_width_pt: float
    col_left_px: float = 0.0
    col_right_px: float = 0.0
    in_table_cell: bool = False
    image_loader: Optional[Callable] = None
