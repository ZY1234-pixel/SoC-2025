"""全流程测试包的模型路径配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    """当前交付模型的固定路径。"""

    package_root: Path
    code_root: Path
    dataset_root: Path
    result_root: Path
    docflow_src: Path
    paddle_root: Path
    models_openvino_root: Path
    layout_model: Path
    det_model: Path
    rec_model: Path
    rec_char_dict: Path
    rapidocr_rec_char_dict: Path
    formula_model: Path

    @classmethod
    def discover(cls) -> "RuntimePaths":
        code_root = Path(__file__).resolve().parent
        package_root = code_root.parent
        models_openvino_root = code_root / "models_openvino"
        ocr_root = models_openvino_root / "PP-OCRv6_small_rec_openvino"
        return cls(
            package_root=package_root,
            code_root=code_root,
            dataset_root=package_root / "dataset",
            result_root=package_root / "test-result",
            docflow_src=code_root / "docflow_src",
            paddle_root=code_root / "third_party" / "paddle_runtime",
            models_openvino_root=models_openvino_root,
            layout_model=models_openvino_root / "PP-DocLayoutV3_openvino" / "PP-DocLayoutV3.xml",
            det_model=(
                models_openvino_root
                / "PP-OCRv6_small_det_openvino"
                / "PP-OCRv6_small_det_openvino_fp32.xml"
            ),
            rec_model=ocr_root / "PP-OCRv6_small_rec_openvino_fp32.xml",
            rec_char_dict=ocr_root / "ppocrv6_dict.txt",
            rapidocr_rec_char_dict=ocr_root / "ppocrv6_rapidocr_dict.txt",
            formula_model=(
                code_root
                / "models"
                / "formula"
                / "PP-FormulaNet_plus-M"
                / "PP-FormulaNet_plus-M_infer"
            ),
        )
