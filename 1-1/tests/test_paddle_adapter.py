from __future__ import annotations

from docflow.adapters.paddle_adapter import PaddleAdapter


def test_nested_same_category_figure_is_suppressed() -> None:
    adapter = PaddleAdapter()
    results = [
        {
            "type": "figure",
            "bbox": [0, 0, 100, 100],
            "score": 0.9,
        },
        {
            "type": "figure",
            "bbox": [10, 10, 90, 90],
            "score": 0.8,
        },
    ]
    filtered, _report = adapter._suppress_nested_duplicates(results)
    assert len(filtered) == 1
    assert filtered[0]["bbox"] == [0, 0, 100, 100]


def test_caption_family_short_caption_is_suppressed_by_longer_caption() -> None:
    adapter = PaddleAdapter()
    results = [
        {
            "type": "figure_caption",
            "bbox": [10, 10, 50, 20],
            "score": 0.4,
            "res": [{"text": "表7"}],
        },
        {
            "type": "table_caption",
            "bbox": [8, 9, 140, 24],
            "score": 0.8,
            "res": [{"text": "表7"}, {"text": "美人蕉植株高度及开花数"}],
        },
    ]
    filtered, _report = adapter._suppress_nested_duplicates(results)
    assert len(filtered) == 1
    assert filtered[0]["type"] == "table_caption"


def test_caption_family_keeps_distinct_child_caption() -> None:
    adapter = PaddleAdapter()
    results = [
        {
            "type": "figure_caption",
            "bbox": [0, 0, 200, 30],
            "score": 0.8,
            "res": [{"text": "图3 水松生长情况对比"}],
        },
        {
            "type": "figure_caption",
            "bbox": [20, 31, 70, 45],
            "score": 0.7,
            "res": [{"text": "（a）初始状态"}],
        },
    ]
    filtered, _report = adapter._suppress_nested_duplicates(results)
    assert len(filtered) == 2
