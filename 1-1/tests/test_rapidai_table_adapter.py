import inspect

import numpy as np

from docflow.adapters.rapidai_table_adapter import RapidAITableAdapter


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

    assert result == [{"label": "image", "bbox": [0, 0, 80.0, 90.0], "score": 0.8}]


def test_table_crop_keeps_context_around_layout_box() -> None:
    assert RapidAITableAdapter._expand_bbox([100, 50, 200, 150], 300, 250) == [96, 46, 204, 154]


def test_full_page_fallback_is_opt_in() -> None:
    assert inspect.signature(RapidAITableAdapter).parameters["full_page_fallback"].default is False

    adapter = object.__new__(RapidAITableAdapter)
    adapter.full_page_fallback = False
    regions = [{"type": "text", "bbox": [0, 0, 20, 20]}]

    assert adapter.enrich(np.zeros((30, 30, 3), dtype=np.uint8), regions, 0, ".") is regions


def test_table_recognition_failure_preserves_source_region(tmp_path) -> None:
    class FailingRecognizer:
        def predict(self, _crop):
            raise RuntimeError("model failure")

    adapter = object.__new__(RapidAITableAdapter)
    adapter.full_page_fallback = False
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


def test_alternate_engine_is_only_needed_for_overlapping_structure() -> None:
    clean = {
        "row_count": 2,
        "col_count": 2,
        "cells": [
            {"row": 0, "col": 0},
            {"row": 0, "col": 1},
            {"row": 1, "col": 0},
            {"row": 1, "col": 1},
        ],
        "diagnostics": {"empty_cells": 0, "span_repairs": 0},
    }
    overlapping = {
        **clean,
        "cells": [
            {"row": 0, "col": 0, "colspan": 2},
            {"row": 0, "col": 1},
            {"row": 1, "col": 0},
            {"row": 1, "col": 1},
        ],
    }

    assert RapidAITableAdapter._needs_alternate_engine(clean) is False
    assert RapidAITableAdapter._needs_alternate_engine(overlapping) is True
