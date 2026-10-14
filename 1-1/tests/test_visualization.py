import numpy as np

from docflow.utils.visualization import draw_layout_ocr


def test_layout_overlay_marks_inline_formula_after_table() -> None:
    image = np.full((220, 260, 3), 255, dtype=np.uint8)
    regions = [
        {"type": "table", "raw_type": "table", "bbox": [10, 10, 250, 210], "score": 0.95},
        {
            "type": "equation",
            "raw_type": "inline_formula",
            "bbox": [60, 80, 140, 115],
            "score": 0.75,
            "res": {"latex": "x^2"},
        },
    ]

    result = draw_layout_ocr(image, regions)

    # PIL 使用 RGB，OpenCV 数组为 BGR。
    assert tuple(result[97, 60]) == (153, 72, 236)
    assert tuple(result[97, 139]) == (153, 72, 236)
