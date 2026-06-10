from __future__ import annotations

import numpy as np

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


def test_text_block_bbox_expands_to_cover_text_line_polys() -> None:
    adapter = PaddleAdapter()
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    results = [
        {
            "type": "text",
            "bbox": [20, 80, 160, 150],
            "score": 0.9,
            "res": [
                {
                    "text": "瓦的北红海省博物馆。",
                    "text_region": [[24, 58], [150, 58], [150, 78], [24, 78]],
                },
                {
                    "text": "后续正文",
                    "text_region": [[24, 90], [150, 90], [150, 110], [24, 110]],
                },
            ],
        }
    ]
    converted = adapter.convert(results, image)
    block = converted["pages"][0]["blocks"][0]
    assert block["bbox"] == [20.0, 58.0, 160.0, 150.0]


def test_sentence_like_title_duplicate_does_not_suppress_adjacent_text() -> None:
    adapter = PaddleAdapter()
    results = [
        {
            "type": "title",
            "bbox": [1153, 776, 1329, 797],
            "score": 0.49,
            "res": [{"text": "瓦的北红海省博物馆。"}],
        },
        {
            "type": "text",
            "bbox": [1154, 801, 1505, 990],
            "score": 0.30,
            "res": [
                {"text": "瓦的北红海省博物馆。"},
                {"text": "博物馆二层陈列着一个发掘自阿杜利斯古城的中国古代陶制酒器。"},
            ],
        },
    ]

    filtered, report = adapter._suppress_nested_duplicates(results)

    assert len(filtered) == 1
    assert filtered[0]["type"] == "text"
    assert any(entry["reason"] == "cross_category_text_duplicate" for entry in report)


def test_title_bbox_trims_leading_isolated_formula_number() -> None:
    adapter = PaddleAdapter()
    image = np.zeros((700, 1000, 3), dtype=np.uint8)
    results = [
        {
            "type": "title",
            "bbox": [450, 200, 900, 230],
            "score": 0.9,
            "res": [
                {
                    "text": "(19)",
                    "text_region": [[450, 200], [485, 200], [485, 224], [450, 224]],
                },
                {
                    "text": "4 Numerical Solution and its Validation",
                    "text_region": [[520, 200], [900, 200], [900, 230], [520, 230]],
                },
            ],
        }
    ]

    converted = adapter.convert(results, image)
    block = converted["pages"][0]["blocks"][0]

    assert block["text"] == "4 Numerical Solution and its Validation"
    assert block["bbox"] == [520.0, 200.0, 900.0, 230.0]


def test_low_score_merged_text_does_not_suppress_higher_score_split_text_blocks() -> None:
    adapter = PaddleAdapter()
    results = [
        {
            "type": "text",
            "bbox": [100, 100, 300, 180],
            "score": 0.81,
            "res": [{"text": "The testing we do serves two purposes."}],
        },
        {
            "type": "text",
            "bbox": [100, 182, 300, 260],
            "score": 0.74,
            "res": [{"text": "Second, it provides important information."}],
        },
        {
            "type": "text",
            "bbox": [98, 98, 304, 260],
            "score": 0.21,
            "res": [
                {"text": "The testing we do serves two purposes."},
                {"text": "Second, it provides important information."},
            ],
        },
    ]

    filtered, report = adapter._suppress_nested_duplicates(results)

    assert len(filtered) == 2
    assert [round(float(item.get("score", 0.0)), 2) for item in filtered] == [0.81, 0.74]
    assert any(entry["removed_index"] == 2 for entry in report)


def test_low_score_formula_duplicate_does_not_suppress_section_title() -> None:
    adapter = PaddleAdapter()
    results = [
        {
            "type": "title",
            "bbox": [260, 858, 659, 888],
            "score": 0.73,
            "res": [{"text": "III. EXPERIMENTS AND RESULTS"}],
        },
        {
            "type": "equation",
            "bbox": [261, 859, 664, 888],
            "score": 0.22,
            "res": [],
        },
    ]

    filtered, report = adapter._suppress_nested_duplicates(results)

    assert len(filtered) == 1
    assert filtered[0]["type"] == "title"
    assert any(entry["reason"] == "cross_category_visual_text_duplicate" for entry in report)


def test_figure_with_internal_source_text_is_not_dropped_by_adjacent_text_duplicate() -> None:
    adapter = PaddleAdapter()
    results = [
        {
            "type": "figure",
            "bbox": [526, 199, 722, 385],
            "score": 0.95,
            "res": [
                {"text": "S&P500"},
                {"text": "2022"},
                {"text": "2023"},
                {"text": "Source: LSEG Workspace"},
            ],
        },
        {
            "type": "text",
            "bbox": [529, 385, 635, 399],
            "score": 0.98,
            "res": [{"text": "Source: LSEG Workspace"}],
        },
    ]

    filtered, report = adapter._suppress_nested_duplicates(results)

    assert len(filtered) == 2
    assert all(entry["removed_index"] != 0 for entry in report)


def test_caption_anchored_visual_region_is_recalled_as_missing_figure() -> None:
    adapter = PaddleAdapter()
    image = np.full((900, 700, 3), 255, dtype=np.uint8)
    image[360:620, 170:520] = 210
    results = [
        {
            "type": "figure_caption",
            "bbox": [260, 645, 430, 675],
            "score": 0.8,
            "res": [{"text": "图6-21 绘制两侧曲线"}],
        }
    ]

    converted = adapter.convert(results, image)
    blocks = converted["pages"][0]["blocks"]

    assert [block["category"] for block in blocks] == ["figure_caption", "figure"]
    figure = blocks[1]
    assert figure["bbox"][0] <= 175
    assert figure["bbox"][1] <= 365
    assert figure["bbox"][2] >= 515
    assert figure["bbox"][3] >= 615


def test_caption_anchored_visual_recall_does_not_duplicate_existing_figure() -> None:
    adapter = PaddleAdapter()
    image = np.full((900, 700, 3), 255, dtype=np.uint8)
    image[360:620, 170:520] = 210
    results = [
        {
            "type": "figure",
            "bbox": [170, 360, 520, 620],
            "score": 0.9,
        },
        {
            "type": "figure_caption",
            "bbox": [260, 645, 430, 675],
            "score": 0.8,
            "res": [{"text": "图6-21 绘制两侧曲线"}],
        },
    ]

    converted = adapter.convert(results, image)
    blocks = converted["pages"][0]["blocks"]

    assert [block["category"] for block in blocks].count("figure") == 1
