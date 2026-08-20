from __future__ import annotations

from pathlib import Path

from model import RuntimePaths
from model_integration.runtime import build_layout_dict_from_inference


def test_runtime_paths_default_models_point_to_pp_doclayout_v3_and_ppocr_v6() -> None:
    paths = RuntimePaths.discover()
    assert paths.layout_model == (
        Path(__file__).resolve().parents[1]
        / "Code"
        / "models_openvino"
        / "PP-DocLayoutV3_openvino"
        / "PP-DocLayoutV3.xml"
    )
    assert paths.layout_model_spec.name == "pp-doclayout-v3"
    assert paths.layout_model_spec.use_onnx is True
    assert paths.det_model.name == "PP-OCRv6_small_det_openvino_fp32.xml"
    assert paths.rec_model.name == "PP-OCRv6_small_rec_openvino_fp32.xml"
    assert paths.rec_char_dict == paths.rec_model.parent / "ppocrv6_dict.txt"
    assert paths.rapidocr_rec_char_dict == paths.rec_model.parent / "ppocrv6_rapidocr_dict.txt"
    assert paths.table_model == (
        Path(__file__).resolve().parents[1]
        / "Code"
        / "models"
        / "table"
        / "SLANet_plus"
        / "SLANet_plus.onnx"
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


def test_build_layout_dict_for_pp_doclayout_v3(tmp_path: Path) -> None:
    paths = RuntimePaths.discover(layout_model_name="pp-doclayout-v3")
    fallback = paths.paddle_root / "ppocr" / "utils" / "dict" / "layout_dict" / "layout_cdla_dict.txt"
    dict_path = build_layout_dict_from_inference(paths.layout_model_spec, fallback, tmp_path)

    labels = dict_path.read_text(encoding="utf-8").splitlines()
    assert labels[:5] == ["abstract", "algorithm", "aside_text", "chart", "content"]
    assert labels[-5:] == ["seal", "table", "text", "vertical_text", "vision_footnote"]
    assert len(labels) == 25


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
    assert labels[0:4] == ["paragraph_title", "image", "text", "number"]
    assert labels[-3:] == ["header_image", "footer_image", "aside_text"]
    assert len(labels) == 23
