import numpy as np

from Code.model_integration import OpenVINOModelRuntime
from Code.model_integration.runtime import recognize_formula_regions, sanitize_formula_latex


def test_model_runtime_normalizes_layout_and_ocr_outputs() -> None:
    class Engine:
        @staticmethod
        def layout_predictor(_image):
            return [{"label": "text", "bbox": [1, 2, 3, 4]}], 0.12

        @staticmethod
        def text_system(_image):
            return np.array([[[1, 2], [3, 2], [3, 4], [1, 4]]]), [("示例", 0.95)], {"det": 0.02, "rec": 0.03}

    runtime = object.__new__(OpenVINOModelRuntime)
    runtime.engine = Engine()
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    assert runtime.run_layout(image)["regions"][0]["label"] == "text"
    assert runtime.run_ocr(image) == {
        "lines": [
            {
                "text_region": [[1, 2], [3, 2], [3, 4], [1, 4]],
                "text": "示例",
                "confidence": 0.95,
            }
        ],
        "timing": {"det": 0.02, "rec": 0.03},
    }


def test_model_runtime_runs_table_with_table_cls_engine() -> None:
    class Recognizer:
        table_engine_type = "auto"

        def predict(self, _image):
            return {"status": "ok", "table_type": self.table_engine_type}

    class Adapter:
        recognizer = Recognizer()

    runtime = object.__new__(OpenVINOModelRuntime)
    runtime._table_adapter = Adapter()
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    result = runtime.run_table(image)

    assert result == {"status": "ok", "table_type": "auto"}
    assert runtime._table_adapter.recognizer.table_engine_type == "auto"


def test_model_runtime_runs_document_and_enriches_table_regions(tmp_path) -> None:
    class Engine:
        @staticmethod
        def __call__(_image, img_idx=0):
            return [{"type": "table", "bbox": [1, 2, 8, 9], "img_idx": img_idx}], {"layout": 0.01}

    class Adapter:
        @staticmethod
        def enrich(_image, regions, page_index, output_dir):
            assert page_index == 2
            assert output_dir == tmp_path
            regions[0]["res"] = {"html": "<table></table>"}
            return regions

    runtime = object.__new__(OpenVINOModelRuntime)
    runtime.engine = Engine()
    runtime._table_adapter = Adapter()
    runtime.runtime_dir = tmp_path
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    result = runtime.run_document(image, page_index=2, table_output_dir=tmp_path)

    assert result["regions"][0]["res"]["html"] == "<table></table>"
    assert result["timing"]["layout"] == 0.01
    assert result["timing"]["formula"] == 0
    assert result["timing"]["rapidai_table"] >= 0
    assert result["timing"]["all_with_table"] >= 0


def test_formula_recognition_updates_matching_regions_only() -> None:
    class Recognizer:
        def predict(self, images):
            assert len(images) == 1
            return [r"x^2+y^2"]

    image = np.zeros((20, 40, 3), dtype=np.uint8)
    regions = [
        {"raw_type": "inline_formula", "bbox": [1, 2, 8, 9], "img": image[2:9, 1:8]},
        {"raw_type": "text", "bbox": [10, 2, 20, 9]},
    ]

    elapsed = recognize_formula_regions(image, regions, Recognizer())

    assert elapsed >= 0
    assert regions[0]["res"] == {"latex": "x^2+y^2"}
    assert "res" not in regions[1]


def test_formula_latex_sanitizer_repairs_common_model_artifacts() -> None:
    assert sanitize_formula_latex(
        r"\mathrm{b e d}+\fracfrac{1}{2}+\bar{\mathfrak{u}}+\boldsymbol{\Psi}"
        r"+\begin{array}{r}{x=1}\end{array}"
    ) == (
        r"\mathrm{bed}+\frac{1}{2}+\overline{\mathrm{u}}+\mathbf{\Psi}"
        r"+{x=1}"
    )
