"""图片（图形）版面区块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from docflow.model.base import Block, BlockType


@dataclass
class ImageBlock(Block):
    """包含图片/图形的版面区块。"""

    block_type: BlockType = BlockType.FIGURE
    image_data: Optional[bytes] = None
