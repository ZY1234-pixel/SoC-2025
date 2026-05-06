"""Zone —— 共享相同列布局的水平区块条带。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from docflow.model.base import Block, BlockType

_STRIP_ROW_TYPES = frozenset({
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
})


@dataclass
class Zone:
    """页面中所有区块共享相同列结构的水平区域。

    属性:
        col_count:    该区域的列数（1 = 单栏）。
        blocks:       属于该区域的版面区块。
        has_spanned:  若至少有一个区块跨多列则为 ``True``。
    """

    col_count: int
    blocks: List[Block] = field(default_factory=list)
    has_spanned: bool = False
    flow_id: str = ""
    flow_kind: str = ""

    @property
    def is_strip_row(self) -> bool:
        return bool(self.blocks) and all(block.block_type in _STRIP_ROW_TYPES for block in self.blocks)

    @property
    def rendering_strategy(self) -> str:
        """选择该区域的 DOCX 渲染策略。

        返回以下之一:

        * ``'single_col'``        -- 单栏流式排版（无表格包装）。
        * ``'multi_col_table'``   -- 使用无边框表格的多栏布局
          （适用于有跨列和无跨列的多栏区域）。
        """
        if self.is_strip_row:
            return "strip_row"
        if self.col_count == 1:
            return "single_col"
        return "multi_col_table"

    def blocks_by_column(self) -> Dict[int, List[Block]]:
        """按 :attr:`~Block.col_index` 分组区块并按从上到下排序
        （``bbox.y1``）。"""
        groups: Dict[int, List[Block]] = defaultdict(list)
        for blk in self.blocks:
            groups[blk.col_index].append(blk)
        for col_blocks in groups.values():
            col_blocks.sort(key=lambda b: b.bbox.y1)
        return dict(groups)
