"""公式版面区块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from docflow.model.base import Block, BlockType


@dataclass
class EquationBlock(Block):
    """包含数学公式的版面区块。"""

    block_type: BlockType = BlockType.EQUATION
    latex: Optional[str] = None
    image_data: Optional[bytes] = None
