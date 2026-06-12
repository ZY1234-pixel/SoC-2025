from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code" / "docflow_src"))
sys.path.insert(0, str(ROOT / "Code" / "third_party" / "paddle_runtime"))

from ppstructure.layout.predict_layout import LayoutPredictor
from ppstructure.predict_system import StructureSystem


def _make_predictor_stub() -> LayoutPredictor:
    predictor = LayoutPredictor.__new__(LayoutPredictor)
    predictor.enable_tiled_recall = True
    predictor.tile_overlap_ratio = 0.18
    predictor.tile_trigger_ratio = 1.05
    predictor.tile_margin_ratio = 0.02
    predictor.tile_max_passes = 16
    predictor.ncnn_input_size = [1024, 1024]
    predictor.use_ncnn = False
    predictor.postprocess_op = None
    return predictor


def test_merge_layout_results_deduplicates_same_label_boxes():
    predictor = _make_predictor_stub()
    image = np.zeros((1600, 1200, 3), dtype=np.uint8)
    primary = [
        {"label": "text", "score": 0.90, "bbox": np.array([100, 100, 400, 300], dtype=np.float32)},
    ]
    fallback = [
        {"label": "text", "score": 0.88, "bbox": np.array([108, 108, 398, 298], dtype=np.float32)},
        {"label": "figure", "score": 0.87, "bbox": np.array([650, 900, 1100, 1300], dtype=np.float32)},
    ]

    merged = predictor._merge_layout_results(primary, fallback, image.shape)

    assert len(merged) == 2
    assert sum(1 for item in merged if item["label"] == "text") == 1
    assert any(item["label"] == "figure" for item in merged)


def test_call_runs_tiled_recall_and_merges_extra_boxes():
    predictor = _make_predictor_stub()
    image = np.zeros((2200, 1700, 3), dtype=np.uint8)

    def fake_predict_single(crop):
        h, w = crop.shape[:2]
        if h == 2200 and w == 1700:
            return [
                {"label": "title", "score": 0.80, "bbox": np.array([300, 80, 1300, 220], dtype=np.float32)},
            ], 0.1
        return [
            {"label": "text", "score": 0.86, "bbox": np.array([180, 220, 720, 520], dtype=np.float32)},
        ], 0.05

    predictor._predict_single = fake_predict_single

    merged, elapsed = predictor(image)

    assert elapsed > 0.1
    labels = [item["label"] for item in merged]
    assert "title" in labels
    assert "text" in labels
    assert len(merged) >= 2


def test_uncovered_ocr_line_recall_adds_missing_text_layout():
    layout = [
        {"label": "text", "score": 0.91, "bbox": np.array([100, 100, 700, 180], dtype=np.float32)},
        {"label": "text", "score": 0.90, "bbox": np.array([100, 250, 700, 320], dtype=np.float32)},
    ]
    text_res = [
        {
            "text": "covered line",
            "confidence": 0.98,
            "text_region": [[120, 120], [650, 120], [650, 150], [120, 150]],
        },
        {
            "text": "4) missing list item line",
            "confidence": 0.97,
            "text_region": [[110, 205], [680, 205], [680, 235], [110, 235]],
        },
    ]

    recalled = StructureSystem._recall_uncovered_text_layouts(layout, text_res, (1000, 800, 3))

    assert len(recalled) == 1
    assert recalled[0]["label"] == "text"
    assert recalled[0]["bbox"].tolist() == [110.0, 205.0, 680.0, 235.0]
