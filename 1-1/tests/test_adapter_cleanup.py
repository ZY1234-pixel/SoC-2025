from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code" / "docflow_src"))

import numpy as np

from docflow.adapters.paddle_adapter import PaddleAdapter


def test_caption_family_dedup_keeps_longer_caption():
    adapter = PaddleAdapter()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    results = [
        {"type": "figure_caption", "bbox": [10, 10, 30, 20], "score": 0.4, "res": [{"text": "表7"}]},
        {
            "type": "table_caption",
            "bbox": [8, 9, 80, 22],
            "score": 0.8,
            "res": [{"text": "表7 美人蕉植株高度及开花数"}],
        },
    ]
    converted = adapter.convert(results, image)
    blocks = converted["pages"][0]["blocks"]
    assert len(blocks) == 1
    assert "美人蕉植株高度及开花数" in blocks[0]["text"]
