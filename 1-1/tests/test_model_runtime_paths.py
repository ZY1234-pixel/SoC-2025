from pathlib import Path

import pytest

from Code.model import RuntimePaths
from Code.model_integration.runtime import FormulaRecognitionRunner, build_layout_dict


def test_runtime_paths_only_reference_current_models() -> None:
    paths = RuntimePaths.discover()

    assert paths.layout_model.name == "PP-DocLayoutV3.xml"
    assert paths.det_model.name == "PP-OCRv6_small_det_openvino_fp32.xml"
    assert paths.rec_model.name == "PP-OCRv6_small_rec_openvino_fp32.xml"
    assert paths.rec_char_dict == paths.rec_model.parent / "ppocrv6_dict.txt"
    assert paths.rapidocr_rec_char_dict == paths.rec_model.parent / "ppocrv6_rapidocr_dict.txt"
    assert paths.formula_model.name == "PP-FormulaNet_plus-M_infer"


def test_build_layout_dict_writes_pp_doclayout_v3_labels(tmp_path: Path) -> None:
    dict_path = build_layout_dict(tmp_path)

    labels = dict_path.read_text(encoding="utf-8").splitlines()
    assert labels[:5] == ["abstract", "algorithm", "aside_text", "chart", "content"]
    assert labels[-5:] == ["seal", "table", "text", "vertical_text", "vision_footnote"]
    assert len(labels) == 25


def test_formula_runner_requires_onnx_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="inference.onnx"):
        FormulaRecognitionRunner(tmp_path)._load()
