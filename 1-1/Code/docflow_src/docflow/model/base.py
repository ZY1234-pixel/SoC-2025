"""版面恢复数据模型的核心基类。"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# BBox —— 轴对齐边界框
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """像素坐标下的轴对齐边界框。"""

    x1: float
    y1: float
    x2: float
    y2: float

    # -- 派生属性 --------------------------------------------------

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    # -- 空间辅助方法 -----------------------------------------------------

    def has_y_overlap(self, other: "BBox", min_px: float = 4) -> bool:
        """若本框与 *other* 在垂直方向上重叠至少 *min_px* 像素，则返回 *True*。"""
        overlap = min(self.y2, other.y2) - max(self.y1, other.y1)
        return overlap >= min_px

    def intersection_width(self, other: "BBox") -> float:
        """返回水平方向的重叠宽度（≥0）。"""
        return max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))

    def contains(self, other: "BBox") -> bool:
        """若 *other* 完全在本框内部，则返回 *True*。"""
        return (
            self.x1 <= other.x1
            and self.y1 <= other.y1
            and self.x2 >= other.x2
            and self.y2 >= other.y2
        )

    def union(self, other: "BBox") -> "BBox":
        """返回包含两个框的最小外接框。"""
        return BBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )


# ---------------------------------------------------------------------------
# BlockType —— 版面分析识别的区块类别
# ---------------------------------------------------------------------------

class BlockType(enum.Enum):
    TEXT = "text"
    TITLE = "title"
    TABLE = "table"
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    TABLE_CAPTION = "table_caption"
    TABLE_FOOTNOTE = "table_footnote"
    FORMULA = "formula"
    FORMULA_CAPTION = "formula_caption"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    REFERENCE = "reference"
    ABSTRACT = "abstract"
    CODE = "code"
    LIST = "list"
    FOOTNOTE = "footnote"
    WATERMARK = "watermark"
    ABANDON = "abandon"

    EQUATION = "formula"  # "equation" 的别名


# ---------------------------------------------------------------------------
# Alignment —— 对齐方式
# ---------------------------------------------------------------------------

class Alignment(enum.Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"


# ---------------------------------------------------------------------------
# Block —— 所有版面元素的抽象基类
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """页面上每个版面元素的抽象基类。"""

    bbox: BBox
    block_type: BlockType
    block_id: str = ""
    confidence: float = 1.0
    col_count: int = 1
    col_index: int = 0
    spanned_cols: list = field(default_factory=lambda: [0])
    attributes: Optional[Dict[str, Any]] = None
    spans: List[Dict[str, Any]] = field(default_factory=list)
