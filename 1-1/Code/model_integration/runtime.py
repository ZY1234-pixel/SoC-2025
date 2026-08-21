"""版面分析、OCR 和表格识别模型的统一调用入口。"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
DOCFLOW_SRC_ROOT = CODE_ROOT / "docflow_src"
if str(DOCFLOW_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(DOCFLOW_SRC_ROOT))

from ..model import LayoutModelSpec, RuntimePaths
from ..utils import ensure_runtime_paths
from docflow.adapters.rapidai_table_adapter import RapidAITableAdapter

DOCLAYOUT_YOLO_LABELS = [
    "title",
    "plain text",
    "abandon",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "table_footnote",
    "isolate_formula",
    "formula_caption",
]

DEFAULT_LAYOUT_SCORE_THRESHOLD = 0.50
DEFAULT_PP_DOCLAYOUT_V3_SCORE_THRESHOLD = 0.40
DEFAULT_DOCLAYOUT_YOLO_SCORE_THRESHOLD = 0.18
RAW_RESULT_PREVIEW_MAX_TEXT = 300
PICODET_LAYOUT_17CLS_LABELS = [
    "text",
    "title",
    "list",
    "table",
    "figure",
    "header",
    "footer",
    "reference",
    "equation",
    "abstract",
    "content",
    "figure_caption",
    "table_caption",
    "table_footnote",
    "formula_caption",
    "algorithm",
    "seal",
]
PP_DOCLAYOUT_M_LABELS = [
    "paragraph_title",
    "image",
    "text",
    "number",
    "abstract",
    "content",
    "figure_title",
    "formula",
    "table",
    "table_title",
    "reference",
    "doc_title",
    "footnote",
    "header",
    "algorithm",
    "footer",
    "seal",
    "chart_title",
    "chart",
    "formula_number",
    "header_image",
    "footer_image",
    "aside_text",
]
PP_DOCLAYOUT_V3_LABELS = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
]


def bootstrap_import_paths(paths: RuntimePaths) -> None:
    """将打包内的 Paddle 运行时与 DocFlow 源码加入导入路径。"""
    for path in (paths.paddle_root, paths.docflow_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def resolve_layout_score_threshold(layout_spec: LayoutModelSpec) -> str:
    """按模型类型选择版面置信度阈值，并允许通过环境变量覆盖。"""
    raw_override = os.environ.get("DOCFLOW_LAYOUT_SCORE_THRESHOLD", "").strip()
    if raw_override:
        try:
            value = float(raw_override)
        except ValueError:
            value = DEFAULT_DOCLAYOUT_YOLO_SCORE_THRESHOLD
        else:
            value = min(1.0, max(0.01, value))
        return f"{value:.2f}"

    spec_name = layout_spec.name.lower()
    if "pp-doclayout-v3" in spec_name:
        return f"{DEFAULT_PP_DOCLAYOUT_V3_SCORE_THRESHOLD:.2f}"
    if "doclayout_yolo" in spec_name:
        return f"{DEFAULT_DOCLAYOUT_YOLO_SCORE_THRESHOLD:.2f}"
    return f"{DEFAULT_LAYOUT_SCORE_THRESHOLD:.2f}"


def build_layout_dict_from_inference(layout_spec: LayoutModelSpec, fallback_dict_path: Path, out_dir: Path) -> Path:
    """按模型 inference.yml 自动生成 layout 字典文件。"""
    layout_model_dir = layout_spec.model_path if layout_spec.model_path.is_dir() else layout_spec.model_path.parent
    inference_yml = layout_model_dir / "inference.yml"
    if not inference_yml.is_file():
        explicit_labels = {
            "picodet-l_layout_17cls": PICODET_LAYOUT_17CLS_LABELS,
            "pp-doclayout-m": PP_DOCLAYOUT_M_LABELS,
            "pp-doclayout-v3": PP_DOCLAYOUT_V3_LABELS,
        }.get(layout_spec.name)
        if explicit_labels:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = layout_spec.dict_name or f"layout_{layout_model_dir.name.replace('-', '_')}_dict.txt"
            out_path = out_dir / out_name
            out_path.write_text("\n".join(explicit_labels) + "\n", encoding="utf-8")
            return out_path
        metadata_yml = layout_model_dir / "metadata.yaml"
        if metadata_yml.is_file():
            try:
                import yaml

                metadata = yaml.safe_load(metadata_yml.read_text(encoding="utf-8")) or {}
                names = metadata.get("names")
                if isinstance(names, dict):
                    ordered = [names[idx] for idx in sorted(names)]
                    if ordered:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_name = layout_spec.dict_name or f"layout_{layout_model_dir.name.replace('-', '_')}_dict.txt"
                        out_path = out_dir / out_name
                        out_path.write_text("\n".join(ordered) + "\n", encoding="utf-8")
                        return out_path
            except Exception:
                pass
        if "doclayout_yolo" in layout_spec.name.lower():
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = layout_spec.dict_name or f"layout_{layout_model_dir.name.replace('-', '_')}_dict.txt"
            out_path = out_dir / out_name
            out_path.write_text("\n".join(DOCLAYOUT_YOLO_LABELS) + "\n", encoding="utf-8")
            return out_path
        return fallback_dict_path

    labels = []
    in_label_list = False
    with open(inference_yml, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not in_label_list:
                if stripped == "label_list:":
                    in_label_list = True
                continue
            if stripped.startswith("- "):
                label = stripped[2:].strip()
                if label:
                    labels.append(label)
                continue
            if stripped and not stripped.startswith("-"):
                break

    if not labels:
        return fallback_dict_path

    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = layout_spec.dict_name or f"layout_{layout_model_dir.name.replace('-', '_')}_dict.txt"
    out_path = out_dir / out_name
    out_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    return out_path


def make_engine(paths: RuntimePaths, layout_dict_dir: Path, table_backend: str = "rapidai"):
    """初始化 PaddleOCR 的 StructureSystem 引擎。"""
    from ppstructure.utility import parse_args
    from ppstructure.predict_system import StructureSystem

    ppstructure_dir = paths.paddle_root / "ppstructure"
    fallback_layout_dict = paths.paddle_root / "ppocr" / "utils" / "dict" / "layout_dict" / "layout_cdla_dict.txt"
    rec_char_dict = paths.rec_char_dict
    table_char_dict = paths.paddle_root / "ppocr" / "utils" / "dict" / "table_structure_dict_ch.txt"
    layout_dict = build_layout_dict_from_inference(paths.layout_model_spec, fallback_layout_dict, layout_dict_dir)
    layout_score_threshold = resolve_layout_score_threshold(paths.layout_model_spec)

    argv = [
        "--recovery",
        "True",
        "--use_gpu",
        "False",
        "--formula",
        "False",
        "--show_log",
        "False",
        "--table",
        "False" if table_backend == "rapidai" else "True",
        "--layout_model_dir",
        str(paths.layout_model),
        "--layout_score_threshold",
        layout_score_threshold,
        "--layout_dict_path",
        str(layout_dict),
        "--det_model_dir",
        str(paths.det_model),
        "--rec_model_dir",
        str(paths.rec_model),
        "--rec_char_dict_path",
        str(rec_char_dict),
        "--table_model_dir",
        str(paths.table_model),
        "--table_char_dict_path",
        str(table_char_dict),
    ]

    old_argv = sys.argv[:]
    old_cwd = os.getcwd()
    os.chdir(ppstructure_dir)
    try:
        sys.argv = ["test.py"] + argv
        args = parse_args()
        engine = StructureSystem(args)
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
    return engine


class OpenVINOModelRuntime:
    """复用已加载模型，分别提供版面、OCR、表格和整页调用。"""

    def __init__(
        self,
        layout_model_name: str = "pp-doclayout-v3",
        table_engine: str = "auto",
        full_page_table_fallback: bool = False,
        runtime_dir: str | Path | None = None,
    ) -> None:
        self.paths = RuntimePaths.discover(layout_model_name)
        ensure_runtime_paths(self.paths)
        bootstrap_import_paths(self.paths)
        self.runtime_dir = Path(runtime_dir or Path(tempfile.gettempdir()) / "docflow_model_runtime")
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.engine = make_engine(self.paths, self.runtime_dir)
        self.table_engine = table_engine
        self.full_page_table_fallback = full_page_table_fallback
        self._table_adapter = None

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
            raise ValueError("image must be a non-empty numpy array")

    def _get_table_adapter(self) -> RapidAITableAdapter:
        if self._table_adapter is None:
            self._table_adapter = RapidAITableAdapter(
                self.paths.models_openvino_root,
                table_engine=self.table_engine,
                full_page_fallback=self.full_page_table_fallback,
            )
        return self._table_adapter

    def run_layout(self, image: np.ndarray) -> dict:
        """只执行版面分析，返回区域列表和模型耗时。"""
        self._validate_image(image)
        regions, elapsed = self.engine.layout_predictor(image)
        return {"regions": regions, "elapsed": float(elapsed)}

    def run_ocr(self, image: np.ndarray) -> dict:
        """执行 DET、文本裁剪、REC 和解码，返回行级结果。"""
        self._validate_image(image)
        boxes, recognition, timing = self.engine.text_system(image)
        lines = []
        box_items = boxes if boxes is not None else ()
        recognition_items = recognition if recognition is not None else ()
        for box, item in zip(box_items, recognition_items):
            lines.append(
                {
                    "text_region": np.asarray(box).tolist(),
                    "text": str(item[0]),
                    "confidence": float(item[1]),
                }
            )
        return {"lines": lines, "timing": dict(timing or {})}

    def run_table(self, image: np.ndarray, table_engine: str | None = None) -> dict:
        """识别一张已经裁剪好的表格图片。"""
        self._validate_image(image)
        adapter = self._get_table_adapter()
        previous = adapter.recognizer.table_engine_type
        try:
            if table_engine is not None:
                adapter.recognizer.table_engine_type = table_engine
            return adapter.recognizer.predict(image)
        finally:
            adapter.recognizer.table_engine_type = previous

    def run_document(
        self,
        image: np.ndarray,
        page_index: int = 0,
        table_output_dir: str | Path | None = None,
    ) -> dict:
        """执行版面分析、整页 OCR 和版面框内的表格识别。"""
        self._validate_image(image)
        started = time.perf_counter()
        regions, timing = self.engine(image, img_idx=page_index)
        table_started = time.perf_counter()
        has_table = any(str(region.get("type") or "").lower() == "table" for region in regions)
        if has_table or self.full_page_table_fallback:
            regions = self._get_table_adapter().enrich(
                image,
                regions,
                page_index,
                Path(table_output_dir or self.runtime_dir / "tables"),
            )
        result_timing = dict(timing or {})
        result_timing["rapidai_table"] = time.perf_counter() - table_started
        result_timing["all_with_table"] = time.perf_counter() - started
        return {"regions": regions, "timing": result_timing}
