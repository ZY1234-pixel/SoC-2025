from __future__ import annotations

import numpy as np

from docflow.adapters.paddle_adapter import PaddleAdapter
from docflow.model.stages import Rect


def test_adapter_preserves_regions_bbox_and_model_order_without_cleanup() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    results = [
        {"type": "figure", "bbox": [20, 20, 280, 180], "model_order": 2, "score": 0.9},
        {
            "type": "text",
            "bbox": [40, 40, 260, 80],
            "model_order": 1,
            "score": 0.8,
            "res": [{"text": "independent", "text_region": [[40, 40], [260, 40], [260, 80], [40, 80]]}],
        },
    ]

    evidence = PaddleAdapter().collect_evidence(results, image, img_idx=3, source_file="sample.jpg")
    items = evidence.pages[0].items

    assert [item.category for item in items] == ["text", "figure"]
    assert items[0].bbox == Rect(40, 40, 260, 80)
    assert items[0].evidence_id == "p0003_r0001"
    assert items[0].image_base64
    assert evidence.to_dict()["pages"][0]["image_path"] == "sample.jpg"


def test_adapter_uses_source_order_only_when_model_order_is_missing() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    evidence = PaddleAdapter().collect_evidence(
        [
            {"type": "text", "bbox": [0, 0, 10, 10], "model_order": None},
            {"type": "text", "bbox": [0, 10, 10, 20], "model_order": 4},
        ],
        image,
    )

    assert [item.model_order for item in evidence.pages[0].items] == [0.0, 4.0]
