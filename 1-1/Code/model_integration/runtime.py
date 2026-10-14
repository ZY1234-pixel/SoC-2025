"""版面分析、OCR 和表格识别模型的统一调用入口。"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
DOCFLOW_SRC_ROOT = CODE_ROOT / "docflow_src"
if str(DOCFLOW_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(DOCFLOW_SRC_ROOT))

from ..model import RuntimePaths
from ..utils import ensure_runtime_paths
from docflow.adapters.rapidai_table_adapter import RapidAITableAdapter

DEFAULT_PP_DOCLAYOUT_V3_SCORE_THRESHOLD = 0.25
RAW_RESULT_PREVIEW_MAX_TEXT = 300
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

FORMULA_RAW_TYPES = {"display_formula", "equation", "formula", "inline_formula"}


def sanitize_formula_latex(latex: str) -> str:
    """修复公式模型常见的空格和重复命令，不做语义猜测。"""
    value = str(latex or "").strip()
    value = re.sub(
        r"\\mathrm\{([A-Za-z ]+)\}",
        lambda match: "\\mathrm{" + match.group(1).replace(" ", "") + "}",
        value,
    )
    value = value.replace("\\bar{", "\\overline{")
    value = value.replace("\\boldsymbol{", "\\mathbf{")
    value = value.replace("\\mathfrak{", "\\mathrm{")
    value = value.replace("\\fracfrac", "\\frac")
    value = re.sub(
        r"\\begin\{array\}\{[^{}]*\}(.*?)\\end\{array\}",
        _unwrap_single_row_array,
        value,
        flags=re.S,
    )
    return value


def _unwrap_single_row_array(match: re.Match) -> str:
    body = match.group(1)
    if len(re.split(r"\\\\", body)) != 1:
        return match.group(0)
    body = body.replace("&{{}=}&", "=").replace("&=", "=").replace("&", "")
    return body.replace("\\\\", " ")


def bootstrap_import_paths(paths: RuntimePaths) -> None:
    """将打包内的 Paddle 运行时与 DocFlow 源码加入导入路径。"""
    for path in (paths.paddle_root, paths.docflow_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def resolve_layout_score_threshold() -> str:
    """返回 PP-DocLayoutV3 置信度阈值，允许通过环境变量覆盖。"""
    raw_override = os.environ.get("DOCFLOW_LAYOUT_SCORE_THRESHOLD", "").strip()
    if raw_override:
        try:
            value = float(raw_override)
        except ValueError:
            value = DEFAULT_PP_DOCLAYOUT_V3_SCORE_THRESHOLD
        else:
            value = min(1.0, max(0.01, value))
        return f"{value:.2f}"
    return f"{DEFAULT_PP_DOCLAYOUT_V3_SCORE_THRESHOLD:.2f}"


def build_layout_dict(out_dir: Path) -> Path:
    """生成 PP-DocLayoutV3 运行时需要的类别文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layout_pp_doclayout_v3_dict.txt"
    out_path.write_text("\n".join(PP_DOCLAYOUT_V3_LABELS) + "\n", encoding="utf-8")
    return out_path


def make_engine(paths: RuntimePaths, layout_dict_dir: Path):
    """初始化 PaddleOCR 的 StructureSystem 引擎。"""
    from ppstructure.utility import parse_args
    from ppstructure.predict_system import StructureSystem

    ppstructure_dir = paths.paddle_root / "ppstructure"
    rec_char_dict = paths.rec_char_dict
    layout_dict = build_layout_dict(layout_dict_dir)
    layout_score_threshold = resolve_layout_score_threshold()

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
        "False",
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


class FormulaRecognitionRunner:
    """懒加载 PP-FormulaNet ONNX；缺少 ONNX 权重时直接报错。"""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self._session = None
        self._preprocess = None
        self._postprocess = None
        self._input_name = None

    def _onnx_path(self) -> Path | None:
        for name in ("inference.onnx", "model.onnx"):
            path = self.model_dir / name
            if path.is_file():
                return path
        return None

    def _load(self):
        if self._session is not None:
            return
        onnx_path = self._onnx_path()
        if onnx_path is None:
            raise FileNotFoundError(
                f"缺少公式 ONNX 权重：{self.model_dir / 'inference.onnx'}"
            )
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        import onnxruntime as ort
        import yaml
        from paddlex.inference.models.formula_recognition.processors import (
            LatexImageFormat,
            UniMERNetDecode,
            UniMERNetImgDecode,
            UniMERNetTestTransform,
        )

        config_path = self.model_dir / "inference.yml"
        if not config_path.is_file():
            raise FileNotFoundError(f"缺少公式配置：{config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._preprocess = (
            UniMERNetImgDecode((384, 384)),
            UniMERNetTestTransform(),
            LatexImageFormat(),
        )
        self._postprocess = UniMERNetDecode(config["PostProcess"]["character_dict"])
        self._input_name = self._session.get_inputs()[0].name

    def _predict_onnx(self, images: list[np.ndarray]) -> list[str]:
        img_decode, test_transform, image_format = self._preprocess
        decoded = img_decode(images)
        transformed = test_transform(decoded)
        formatted = image_format(transformed)
        batch = np.concatenate(formatted)
        output = self._session.run(None, {self._input_name: batch})[0]
        preds = [np.asarray(output[index]).reshape(-1) for index in range(len(images))]
        return [sanitize_formula_latex(str(value)) for value in self._postprocess(preds)]

    def predict(self, images: list[np.ndarray]) -> list[str]:
        if not images:
            return []
        self._load()
        return self._predict_onnx(images)


def recognize_formula_regions(image: np.ndarray, regions: list, recognizer) -> float:
    """识别公式区域并将 LaTeX 写回原始 region。"""
    if recognizer is None:
        return 0.0
    targets = []
    crops = []
    height, width = image.shape[:2]
    for region in regions or ():
        if not isinstance(region, dict):
            continue
        raw_type = str(region.get("raw_type") or region.get("type") or "").lower()
        if raw_type not in FORMULA_RAW_TYPES:
            continue
        roi = region.get("img")
        if not isinstance(roi, np.ndarray) or roi.size == 0:
            try:
                x1, y1, x2, y2 = (round(float(value)) for value in region["bbox"][:4])
            except (KeyError, TypeError, ValueError):
                continue
            roi = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        if not isinstance(roi, np.ndarray) or roi.size == 0:
            continue
        targets.append(region)
        crops.append(roi)

    if not targets:
        return 0.0
    started = time.perf_counter()
    latexes = recognizer.predict(crops)
    for region, latex in zip(targets, latexes):
        if latex:
            region["res"] = {"latex": latex}
    return time.perf_counter() - started


class OpenVINOModelRuntime:
    """复用已加载模型，分别提供版面、OCR、表格和整页调用。"""

    def __init__(
        self,
        runtime_dir: str | Path | None = None,
        formula_enabled: bool = True,
    ) -> None:
        self.paths = RuntimePaths.discover()
        ensure_runtime_paths(self.paths, formula_enabled=formula_enabled)
        bootstrap_import_paths(self.paths)
        self.runtime_dir = Path(runtime_dir or Path(tempfile.gettempdir()) / "docflow_model_runtime")
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.engine = make_engine(self.paths, self.runtime_dir)
        self._table_adapter = None
        self._formula_recognizer = FormulaRecognitionRunner(self.paths.formula_model) if formula_enabled else None

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
            raise ValueError("image must be a non-empty numpy array")

    def _get_table_adapter(self) -> RapidAITableAdapter:
        if self._table_adapter is None:
            self._table_adapter = RapidAITableAdapter(
                self.paths.models_openvino_root,
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

    def run_table(self, image: np.ndarray) -> dict:
        """识别一张已经裁剪好的表格图片。"""
        self._validate_image(image)
        adapter = self._get_table_adapter()
        return adapter.recognizer.predict(image)

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
        formula_elapsed = recognize_formula_regions(image, regions, getattr(self, "_formula_recognizer", None))
        table_started = time.perf_counter()
        has_table = any(str(region.get("type") or "").lower() == "table" for region in regions)
        if has_table:
            regions = self._get_table_adapter().enrich(
                image,
                regions,
                page_index,
                Path(table_output_dir or self.runtime_dir / "tables"),
            )
        result_timing = dict(timing or {})
        result_timing["formula"] = formula_elapsed
        result_timing["rapidai_table"] = time.perf_counter() - table_started
        result_timing["all_with_table"] = time.perf_counter() - started
        return {"regions": regions, "timing": result_timing}
