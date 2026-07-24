from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code" / "docflow_src"))
sys.path.insert(0, str(ROOT / "Code"))

import numpy as np

from docflow.adapters.paddle_adapter import PaddleAdapter
from test import summarize_raw_result


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


def test_text_carry_over_trim_requires_two_exact_boundary_lines():
    adapter = PaddleAdapter()
    results = [
        {
            "type": "text",
            "bbox": [10, 10, 120, 60],
            "res": [
                {"text": "第一行", "text_region": [[10, 10], [120, 10], [120, 28], [10, 28]]},
                {"text": "重复行A", "text_region": [[10, 32], [120, 32], [120, 50], [10, 50]]},
                {"text": "重复行B", "text_region": [[10, 54], [120, 54], [120, 72], [10, 72]]},
            ],
        },
        {
            "type": "text",
            "bbox": [12, 74, 122, 150],
            "res": [
                {"text": "重复行A", "text_region": [[12, 74], [122, 74], [122, 92], [12, 92]]},
                {"text": "重复行B", "text_region": [[12, 96], [122, 96], [122, 114], [12, 114]]},
                {"text": "第二段正文", "text_region": [[12, 122], [122, 122], [122, 142], [12, 142]]},
            ],
        },
    ]

    filtered, report = adapter._trim_carry_over_text_regions(results)
    assert len(filtered) == 2
    assert filtered[1]["res"][0]["text"] == "第二段正文"
    assert any(item["reason"] == "text_carry_over_trim" for item in report)


def test_text_carry_over_trim_keeps_single_exact_boundary_line():
    adapter = PaddleAdapter()
    results = [
        {
            "type": "text",
            "bbox": [10, 10, 120, 60],
            "res": [
                {"text": "上一段正文", "text_region": [[10, 10], [120, 10], [120, 28], [10, 28]]},
                {"text": "借鉴。", "text_region": [[10, 32], [120, 32], [120, 50], [10, 50]]},
            ],
        },
        {
            "type": "text",
            "bbox": [12, 54, 122, 110],
            "res": [
                {"text": "借鉴。", "text_region": [[12, 54], [122, 54], [122, 72], [12, 72]]},
                {"text": "这是本段真正首行", "text_region": [[12, 78], [122, 78], [122, 98], [12, 98]]},
            ],
        },
    ]

    filtered, report = adapter._trim_carry_over_text_regions(results)
    assert len(filtered) == 2
    assert filtered[1]["res"][0]["text"] == "借鉴。"
    assert not report


def test_raw_result_summary_does_not_serialize_image_arrays():
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    raw = [
        {
            "type": "figure",
            "bbox": [1, 2, 7, 8],
            "score": 0.9,
            "img": image,
            "res": [{"text": "inside"}],
        }
    ]

    payload = summarize_raw_result(raw)
    encoded = json.dumps(payload, ensure_ascii=False)

    assert payload["regions"][0]["img_shape"] == [8, 9, 3]
    assert "img" not in payload["regions"][0]
    assert len(encoded) < 1000
