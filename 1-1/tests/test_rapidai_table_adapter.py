import numpy as np

from docflow.adapters.rapidai_table_adapter import RapidAITableAdapter
from docflow.vendor.table_rec.fusion import (
    LayoutObject,
    TableCell,
    assign_ocr_text_to_cells,
    fused_to_html,
    layout_object_to_dict,
    normalize_layout,
    render_cell_images,
)
from docflow.vendor.table_rec.mixed_rules import Rect


def test_crop_layout_regions_uses_native_label_and_local_coordinates() -> None:
    regions = [
        {
            "type": "figure",
            "raw_type": "image",
            "bbox": [90, 40, 180, 140],
            "score": 0.8,
            "layout_model": "pp-doclayout-v3",
        },
        {"type": "text", "bbox": [0, 0, 20, 20], "score": 0.9},
    ]

    result = RapidAITableAdapter.crop_layout_regions(regions, [100, 50, 200, 150])

    assert result == [{"label": "image", "bbox": [0, 0, 80.0, 90.0], "score": 0.8, "res": {}}]


def test_table_crop_keeps_context_around_layout_box() -> None:
    assert RapidAITableAdapter._expand_bbox([100, 50, 200, 150], 300, 250) == [96, 46, 204, 154]


def test_table_rule_geometry_filters_ocr_outside_physical_bounds() -> None:
    image = np.full((220, 300, 3), 255, dtype=np.uint8)
    for y in (20, 80, 200):
        image[y : y + 2, :] = 0
    table_result = {
        "ocr_result": [
            {"text": "above", "rect": [10, 0, 90, 15]},
            {"text": "inside", "rect": [10, 30, 90, 50]},
            {"text": "below", "rect": [10, 205, 90, 215]},
        ]
    }

    RapidAITableAdapter._apply_table_rule_geometry(table_result, image)

    assert table_result["_horizontal_rule_ys"] == [20.0, 80.0, 200.0]
    assert table_result["_content_bbox"] == [0, 21, 300, 200]
    assert [item["text"] for item in table_result["ocr_result"]] == ["inside"]


def test_table_engine_fallback_uses_actual_structure_quality() -> None:
    clean = {
        "row_count": 2,
        "col_count": 2,
        "cells": [
            {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "text": "A"},
            {"row": 0, "col": 1, "rowspan": 1, "colspan": 1, "text": "B"},
            {"row": 1, "col": 0, "rowspan": 1, "colspan": 1, "text": "C"},
            {"row": 1, "col": 1, "rowspan": 1, "colspan": 1, "text": "D"},
        ],
        "diagnostics": {"empty_cells": 0},
    }
    overlapping = {
        **clean,
        "cells": [
            {"row": 0, "col": 0, "rowspan": 2, "colspan": 2, "text": "A"},
            {"row": 1, "col": 1, "rowspan": 1, "colspan": 1, "text": "B"},
        ],
        "diagnostics": {"empty_cells": 1},
    }

    assert RapidAITableAdapter._needs_engine_fallback(clean) is False
    assert RapidAITableAdapter._needs_engine_fallback(overlapping) is True
    assert RapidAITableAdapter._fusion_quality(clean) < RapidAITableAdapter._fusion_quality(overlapping)


def test_pages_without_table_regions_are_not_sent_to_table_models() -> None:
    adapter = object.__new__(RapidAITableAdapter)
    regions = [{"type": "text", "bbox": [0, 0, 20, 20]}]

    assert adapter.enrich(np.zeros((30, 30, 3), dtype=np.uint8), regions, 0, ".") is regions


def test_table_recognition_failure_preserves_source_region(tmp_path) -> None:
    class FailingRecognizer:
        def predict(self, _crop):
            raise RuntimeError("model failure")

    adapter = object.__new__(RapidAITableAdapter)
    adapter.recognizer = FailingRecognizer()
    regions = [{"type": "table", "bbox": [5, 5, 25, 25]}]

    assert adapter.enrich(np.zeros((30, 30, 3), dtype=np.uint8), regions, 0, tmp_path) == regions


def test_formula_layout_object_is_cropped_as_semantic_visual(tmp_path) -> None:
    fused = {
        "cells": [
            {
                "layout_objects": [{"label": "inline_formula", "bbox": [2, 3, 18, 12]}],
            }
        ]
    }

    RapidAITableAdapter._crop_formula_images(
        fused,
        np.full((20, 30, 3), 255, dtype=np.uint8),
        tmp_path,
    )

    formula = fused["cells"][0]["layout_objects"][0]
    assert formula["image_path"] == "assets/formula_000.png"
    assert formula["visual_kind"] == "semantic_visual"
    assert (tmp_path / "formula_000.png").is_file()


def test_engine_fallback_detects_unresolved_empty_row_header_run() -> None:
    fused = {
        "row_count": 2,
        "col_count": 2,
        "ocr_objects": [{"text": "value"}],
        "diagnostics": {"empty_cells": 1},
        "cells": [
            {
                "row": 0,
                "col": 0,
                "rowspan": 1,
                "colspan": 1,
                "role": "row_header",
                "text": "高度/cm",
                "layout_objects": [],
            },
            {
                "row": 1,
                "col": 0,
                "rowspan": 1,
                "colspan": 1,
                "role": "row_header",
                "text": "",
                "layout_objects": [],
            },
            {
                "row": 0,
                "col": 1,
                "rowspan": 1,
                "colspan": 1,
                "role": "body",
                "text": "第一组",
                "layout_objects": [],
            },
            {
                "row": 1,
                "col": 1,
                "rowspan": 1,
                "colspan": 1,
                "role": "body",
                "text": "第二组",
                "layout_objects": [],
            },
        ],
    }

    assert RapidAITableAdapter._semantic_defect_count(fused["cells"], 2, 2) == 1
    assert RapidAITableAdapter._needs_engine_fallback(fused) is True


def test_formula_overlap_removes_ocr_line_that_starts_inside_formula() -> None:
    assert RapidAITableAdapter._formula_overlaps_ocr(
        [10, 10, 100, 40],
        [20, 12, 90, 38],
    )
    assert not RapidAITableAdapter._formula_overlaps_ocr(
        [10, 10, 100, 40],
        [120, 12, 160, 38],
    )


def test_caption_ocr_above_table_top_is_not_assigned_to_header_cell() -> None:
    cells = [
        TableCell(row=0, col=0, bbox=Rect(0, 20, 100, 60), role="column_header"),
        TableCell(row=0, col=1, bbox=Rect(100, 20, 200, 60), role="column_header"),
    ]
    caption = LayoutObject("ocr_text", Rect(20, 0, 180, 15), role="ocr_text", text="Table caption")
    header = LayoutObject("ocr_text", Rect(10, 25, 80, 55), role="ocr_text", text="Header")

    assign_ocr_text_to_cells(cells, [caption, header])

    assert all("Table caption" not in cell.text for cell in cells)
    assert cells[0].text == "Header"


def test_inline_cjk_ocr_fragments_join_without_spaces() -> None:
    cells = [TableCell(row=0, col=0, bbox=Rect(0, 0, 100, 50))]
    ocr = [
        LayoutObject("ocr_text", Rect(10, 10, 40, 30), role="ocr_text", text="地"),
        LayoutObject("ocr_text", Rect(50, 10, 80, 30), role="ocr_text", text="区"),
    ]

    assign_ocr_text_to_cells(cells, ocr)

    assert cells[0].text == "地区"


def test_formula_latex_is_rendered_as_table_marker_instead_of_image() -> None:
    objects = normalize_layout(
        {
            "boxes": [
                {
                    "label": "inline_formula",
                    "bbox": [1, 2, 10, 12],
                    "res": {"latex": "x^2"},
                }
            ]
        }
    )

    html = render_cell_images([{"latex": objects[0].latex, "page_bbox": [101, 202, 110, 212]}])

    assert layout_object_to_dict(objects[0])["latex"] == "x^2"
    assert "docflow-formula" in html
    assert "data-latex='x^2'" in html
    assert "data-page-bbox='101.0,202.0,110.0,212.0'" in html
    assert "<img" not in html


def test_table_cell_page_bbox_is_kept_in_html_geometry() -> None:
    html = fused_to_html(
        {
            "row_count": 1,
            "col_count": 1,
            "cells": [
                {
                    "row": 0,
                    "col": 0,
                    "rowspan": 1,
                    "colspan": 1,
                    "text": "A",
                    "page_bbox": [101, 202, 110, 212],
                }
            ],
        }
    )

    assert "data-page-bbox='101.0,202.0,110.0,212.0'" in html


def test_table_cells_are_translated_to_render_crop_coordinates() -> None:
    cells = [
        {
            "bbox": [20, 30, 80, 90],
            "ocr_objects": [{"bbox": [30, 40, 70, 60], "text": "A"}],
        }
    ]

    translated = RapidAITableAdapter._translate_cells(cells, -10, -20)

    assert translated[0]["bbox"] == [10.0, 10.0, 70.0, 70.0]
    assert translated[0]["ocr_objects"][0]["bbox"] == [20.0, 20.0, 60.0, 40.0]
    assert cells[0]["bbox"] == [20, 30, 80, 90]


def test_ocr_text_below_table_edge_is_not_forced_into_a_cell() -> None:
    cells = [TableCell(row=0, col=0, bbox=Rect(0, 0, 100, 80))]
    ocr = [LayoutObject("ocr_text", Rect(20, 84, 80, 96), role="ocr_text", text="source")]

    assign_ocr_text_to_cells(cells, ocr)

    assert cells[0].text == ""
