from __future__ import annotations

from pathlib import Path

from model import RuntimePaths
from test import build_layout_dict_from_inference


def test_runtime_paths_default_layout_model_points_to_headfloat100() -> None:
    paths = RuntimePaths.discover()
    assert paths.layout_model == (
        Path(__file__).resolve().parents[1]
        / "Code"
        / "models"
        / "layout"
        / "doclayout_yolo_docstructbench_headfloat100_runtime"
    )


def test_runtime_paths_can_switch_to_picodet_onnx_layout_model() -> None:
    paths = RuntimePaths.discover(layout_model_name="picodet-l_layout_17cls")
    assert paths.layout_model == (
        Path(__file__).resolve().parents[1]
        / "Code"
        / "models"
        / "layout"
        / "picodet-l_layout_17cls"
        / "picodet-l_layout_17cls.onnx"
    )
    assert paths.layout_model_spec.use_onnx is True


def test_runtime_paths_can_switch_to_pp_doclayout_onnx_model() -> None:
    paths = RuntimePaths.discover(layout_model_name="pp-doclayout-m")
    assert paths.layout_model == (
        Path(__file__).resolve().parents[1]
        / "Code"
        / "models"
        / "layout"
        / "pp-doclayout-m"
        / "pp-doclayout-m.onnx"
    )
    assert paths.layout_model_spec.use_onnx is True


def test_build_layout_dict_for_picodet_17cls(tmp_path: Path) -> None:
    paths = RuntimePaths.discover(layout_model_name="picodet-l_layout_17cls")
    fallback = paths.paddle_root / "ppocr" / "utils" / "dict" / "layout_dict" / "layout_cdla_dict.txt"
    dict_path = build_layout_dict_from_inference(paths.layout_model_spec, fallback, tmp_path)

    labels = dict_path.read_text(encoding="utf-8").splitlines()
    assert labels[:5] == ["text", "title", "list", "table", "figure"]
    assert "formula_caption" in labels
    assert len(labels) == 17


def test_build_layout_dict_for_pp_doclayout_m(tmp_path: Path) -> None:
    paths = RuntimePaths.discover(layout_model_name="pp-doclayout-m")
    fallback = paths.paddle_root / "ppocr" / "utils" / "dict" / "layout_dict" / "layout_cdla_dict.txt"
    dict_path = build_layout_dict_from_inference(paths.layout_model_spec, fallback, tmp_path)

    labels = dict_path.read_text(encoding="utf-8").splitlines()
    assert labels[0:4] == ["paragraph_title", "doc_title", "text", "number"]
    assert labels[-3:] == ["aside_text", "formula_number", "figure_caption"]
    assert len(labels) == 23
