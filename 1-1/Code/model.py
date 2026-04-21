"""全流程测试包的路径与模型配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    layout_model: Path
    det_model: Path
    rec_model: Path
    table_model: Path

    @classmethod
    def discover(cls) -> "RuntimePaths":
        code_root = Path(__file__).resolve().parent
        package_root = code_root.parent
        dataset_root = package_root / "dataset"
        result_root = package_root / "test-result"
        docflow_src = code_root / "docflow_src"
        paddle_root = code_root / "third_party" / "paddle_runtime"
        models_root = code_root / "models"
        return cls(
            package_root=package_root,
            code_root=code_root,
            dataset_root=dataset_root,
            result_root=result_root,
            docflow_src=docflow_src,
            paddle_root=paddle_root,
            models_root=models_root,
            layout_model=models_root / "layout" / "picodet_lcnet_x1_0_fgd_layout_cdla_infer",
            det_model=models_root / "det" / "ch" / "PP-OCRv5_mobile_det_infer",
            rec_model=models_root / "rec" / "ch" / "PP-OCRv5_mobile_rec_infer",
            table_model=models_root / "table" / "SLANet_plus_infer",
        )
