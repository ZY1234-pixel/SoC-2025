"""全流程测试包的路径与模型配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LayoutModelSpec:
    """版面模型的规范化描述。"""

    name: str
    model_path: Path
    use_onnx: bool = False
    dict_name: Optional[str] = None


@dataclass(frozen=True)
class RuntimePaths:
    """解析并保存测试包内的关键路径。"""

    package_root: Path
    code_root: Path
    dataset_root: Path
    result_root: Path
    docflow_src: Path
    paddle_root: Path
    models_root: Path
    models_openvino_root: Path
    layout_model: Path
    layout_model_spec: LayoutModelSpec
    det_model: Path
    rec_model: Path
    rec_char_dict: Path
    rapidocr_rec_char_dict: Path
    table_model: Path

    @staticmethod
    def resolve_layout_model_spec(models_root: Path, layout_model_name: str) -> LayoutModelSpec:
        layout_root = models_root / "layout"
        openvino_root = models_root.parent / "models_openvino"
        name = (layout_model_name or "").strip()
        if not name:
            raise ValueError("layout_model_name must not be empty")

        specs = {
            "pp-doclayout-v3": LayoutModelSpec(
                name="pp-doclayout-v3",
                model_path=openvino_root / "PP-DocLayoutV3_openvino" / "PP-DocLayoutV3.xml",
                use_onnx=True,
                dict_name="layout_pp_doclayout_v3_dict.txt",
            ),
            "PP-DocLayoutV3": LayoutModelSpec(
                name="pp-doclayout-v3",
                model_path=openvino_root / "PP-DocLayoutV3_openvino" / "PP-DocLayoutV3.xml",
                use_onnx=True,
                dict_name="layout_pp_doclayout_v3_dict.txt",
            ),
            "doclayout_yolo": LayoutModelSpec(
                name="doclayout_yolo",
                model_path=layout_root / "doclayout_yolo_docstructbench_headfloat100_runtime",
                use_onnx=False,
                dict_name=None,
            ),
            "doclayout_yolo_docstructbench_headfloat100_runtime": LayoutModelSpec(
                name="doclayout_yolo",
                model_path=layout_root / "doclayout_yolo_docstructbench_headfloat100_runtime",
                use_onnx=False,
                dict_name=None,
            ),
            "picodet-l_layout_17cls": LayoutModelSpec(
                name="picodet-l_layout_17cls",
                model_path=layout_root / "picodet-l_layout_17cls" / "picodet-l_layout_17cls.onnx",
                use_onnx=True,
                dict_name="layout_picodet_l_layout_17cls_dict.txt",
            ),
            "pp-doclayout-m": LayoutModelSpec(
                name="pp-doclayout-m",
                model_path=layout_root / "pp-doclayout-m" / "pp-doclayout-m.onnx",
                use_onnx=True,
                dict_name="layout_pp_doclayout_m_dict.txt",
            ),
        }

        spec = specs.get(name)
        if spec is None:
            candidate_dir = layout_root / name
            if candidate_dir.is_dir():
                return LayoutModelSpec(name=name, model_path=candidate_dir, use_onnx=False, dict_name=None)
            raise ValueError(f"Unsupported layout model: {layout_model_name}")
        return spec

    @classmethod
    def discover(cls, layout_model_name: str = "pp-doclayout-v3") -> "RuntimePaths":
        code_root = Path(__file__).resolve().parent
        package_root = code_root.parent
        dataset_root = package_root / "dataset"
        result_root = package_root / "test-result"
        docflow_src = code_root / "docflow_src"
        paddle_root = code_root / "third_party" / "paddle_runtime"
        models_root = code_root / "models"
        models_openvino_root = code_root / "models_openvino"
        layout_model_spec = cls.resolve_layout_model_spec(models_root, layout_model_name)
        return cls(
            package_root=package_root,
            code_root=code_root,
            dataset_root=dataset_root,
            result_root=result_root,
            docflow_src=docflow_src,
            paddle_root=paddle_root,
            models_root=models_root,
            models_openvino_root=models_openvino_root,
            layout_model=layout_model_spec.model_path,
            layout_model_spec=layout_model_spec,
            det_model=models_openvino_root / "PP-OCRv6_small_det_openvino" / "PP-OCRv6_small_det_openvino_fp32.xml",
            rec_model=models_openvino_root / "PP-OCRv6_small_rec_openvino" / "PP-OCRv6_small_rec_openvino_fp32.xml",
            rec_char_dict=models_openvino_root / "PP-OCRv6_small_rec_openvino" / "ppocrv6_dict.txt",
            rapidocr_rec_char_dict=models_openvino_root / "PP-OCRv6_small_rec_openvino" / "ppocrv6_rapidocr_dict.txt",
            table_model=models_root / "table" / "SLANet_plus" / "SLANet_plus.onnx",
        )
