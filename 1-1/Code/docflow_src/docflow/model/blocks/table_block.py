"""表格版面区块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from docflow.model.base import Block, BlockType


@dataclass
class TableBlock(Block):
    """包含表格的版面区块。"""

    block_type: BlockType = BlockType.TABLE
    html: Optional[str] = None
    image_data: Optional[bytes] = None
