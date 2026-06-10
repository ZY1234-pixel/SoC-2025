from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code" / "docflow_src"))

from docflow.layout.color_inferrer import infer_text_colors
from docflow.model.base import BBox, BlockType
from docflow.model.blocks.text_block import TextBlock, TextLine
from docflow.model.page import Page


def test_infer_text_colors_promotes_consistent_red_block(tmp_path):
    image = np.full((80, 220, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "TEST",
        (20, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 180),
        4,
        cv2.LINE_AA,
    )
    image_path = tmp_path / "red_text.png"
    cv2.imwrite(str(image_path), image)

    page = Page(index=0, image_width=220, image_height=80, image_path=str(image_path))
    block = TextBlock(
        bbox=BBox(15, 15, 145, 60),
        block_type=BlockType.TITLE,
        lines=[
            TextLine(
                text="TEST",
                text_region=[[15, 15], [150, 15], [150, 65], [15, 65]],
            )
        ],
    )

    stats = infer_text_colors(page, [block])

    assert stats["colored_blocks"] == 1
    assert block.style is not None
    assert block.style.color is not None
    assert block.style.color != "#000000"
    red = int(block.style.color[1:3], 16)
    green = int(block.style.color[3:5], 16)
    blue = int(block.style.color[5:7], 16)
    assert red > green
    assert red > blue
    assert green < 80
    assert blue < 100
