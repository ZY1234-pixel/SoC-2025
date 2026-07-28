"""全流程测试入口：图片/PDF -> PaddleOCR -> 合并 JSON -> DocFlow 文档输出。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import fitz
from docx import Document as DocxDocument

CODE_ROOT = Path(__file__).resolve().parent
DOCFLOW_SRC_ROOT = CODE_ROOT / "docflow_src"
docflow_src_str = str(DOCFLOW_SRC_ROOT)
if docflow_src_str not in sys.path:
    sys.path.insert(0, docflow_src_str)

from dataset import collect_samples, is_pdf_file
from model import LayoutModelSpec, RuntimePaths
from preprocess import expand_to_pages
from utils import ensure_runtime_paths, find_libreoffice, parse_formats, print_list
from docflow.adapters.paddle_adapter import PaddleAdapter
from docflow.analysis import DocumentAnalyzer
from docflow.layout.font_classifier import FontClassifier
from docflow.model.stages import RecognitionEvidence
from docflow.planning import ReflowPlanner
from docflow.renderer.reflow_docx_renderer import ReflowDocxRenderer
from docflow.renderer.reflow_markdown_renderer import ReflowMarkdownRenderer
from docflow.utils.result_layout import (
    ResultRunLayout,
    build_main_run_manifest,
    write_json,
)


TEXT_TYPES = {
    "text",
    "title",
    "reference",
    "header",
    "footer",
    "figure_caption",
    "table_caption",
}

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
DEFAULT_PP_DOCLAYOUT_V3_SCORE_THRESHOLD = 0.50
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


def resolve_cli_path(raw_path: str | None, default_path: Path) -> Path:
    """解析命令行路径，并兼容常见的 Windows 风格相对路径误输。"""
    if not raw_path:
        return default_path.resolve()

    normalized = raw_path
    if os.name != "nt":
        normalized = normalized.replace("\\", "/")
        if normalized.startswith("..") and not normalized.startswith("../") and "/" not in normalized[2:]:
            normalized = "../" + normalized[2:]
    return Path(normalized).resolve()


def resolve_layout_score_threshold(layout_spec: LayoutModelSpec) -> str:
    """Resolve layout score threshold with a lower doclayout-yolo default.

    The doclayout-yolo detector tends to miss sparse centered section headings
    around tables when the threshold is too high, so we use a lower default for
    that model family and keep an env override for quick ablation.
    """
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


def make_engine(paths: RuntimePaths, layout_dict_dir: Path):
    """初始化 PaddleOCR 的 StructureSystem 引擎。"""
    from ppstructure.utility import parse_args
    from ppstructure.predict_system import StructureSystem

    ppstructure_dir = paths.paddle_root / "ppstructure"
    fallback_layout_dict = paths.paddle_root / "ppocr" / "utils" / "dict" / "layout_dict" / "layout_cdla_dict.txt"
    rec_char_dict = (
        paths.models_root
        / "rec"
        / "ch"
        / "PP-OCRv6_small_rec"
        / "ppocrv6_dict.txt"
    )
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


def print_regions(result: list) -> None:
    for region in result:
        region_type = region.get("type", "?")
        bbox = [int(value) for value in region.get("bbox", [])]
        score = float(region.get("score", 0.0))
        text_preview = ""
        if region_type in TEXT_TYPES and isinstance(region.get("res"), list):
            texts = []
            for item in region["res"]:
                if isinstance(item, dict):
                    texts.append(item.get("text", ""))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    rhs = item[1]
                    if isinstance(rhs, (list, tuple)) and rhs:
                        texts.append(str(rhs[0]))
                    else:
                        texts.append(str(rhs))
            text_preview = " | ".join(texts)[:60]
        print(f"  [{region_type:>15s}] score={score:.2f} bbox={bbox} {text_preview}")


def summarize_raw_result(result: list) -> dict:
    """Build a compact raw-result debug payload without serializing image arrays."""
    regions: list[dict] = []
    for index, region in enumerate(result or []):
        if not isinstance(region, dict):
            regions.append({"index": index, "repr": repr(region)[:RAW_RESULT_PREVIEW_MAX_TEXT]})
            continue

        item: dict = {
            "index": index,
            "type": region.get("type"),
            "bbox": region.get("bbox"),
            "score": float(region.get("score", 0.0) or 0.0),
        }
        if "img_idx" in region:
            item["img_idx"] = region.get("img_idx")
        for key in ("raw_type", "model_order", "layout_model"):
            value = region.get(key)
            if value is not None:
                item[key] = value
        img = region.get("img")
        if hasattr(img, "shape"):
            item["img_shape"] = list(img.shape)

        res_preview: list[dict] = []
        for res_item in region.get("res") or []:
            if isinstance(res_item, dict):
                preview = {
                    "text": str(res_item.get("text", ""))[:RAW_RESULT_PREVIEW_MAX_TEXT],
                }
                if "confidence" in res_item:
                    preview["confidence"] = float(res_item.get("confidence", 0.0) or 0.0)
                if "text_region" in res_item:
                    preview["text_region"] = res_item.get("text_region")
                res_preview.append(preview)
            elif isinstance(res_item, (list, tuple)) and len(res_item) == 2:
                rhs = res_item[1]
                text = rhs[0] if isinstance(rhs, (list, tuple)) and rhs else rhs
                res_preview.append({"text": str(text)[:RAW_RESULT_PREVIEW_MAX_TEXT]})
            else:
                res_preview.append({"repr": repr(res_item)[:RAW_RESULT_PREVIEW_MAX_TEXT]})
        if res_preview:
            item["res"] = res_preview
        regions.append(item)
    return {"regions": regions}


def analyze_page(engine, adapter: PaddleAdapter, image, page_index: int, source_path: str) -> tuple[RecognitionEvidence, list, float]:
    started_at = time.time()
    result, _ = engine(image, img_idx=page_index)
    elapsed = time.time() - started_at
    evidence = adapter.collect_evidence(result, image, img_idx=page_index, source_file=source_path)
    return evidence, result, elapsed


def save_debug_images(
    image,
    result: list,
    sample_layout,
    page_index: int,
) -> None:
    from docflow.utils.visualization import (
        draw_layout_ocr,
        draw_sorted_layout,
    )

    sample_layout.debug_dir.mkdir(parents=True, exist_ok=True)
    layout_path = sample_layout.debug_image_path(page_index, "layout_ocr")
    order_columns_path = sample_layout.debug_image_path(page_index, "reading_order_columns")
    vis_layout = draw_layout_ocr(
        image,
        result,
        font_path=None,
        show_text_preview=True,
    )
    cv2.imwrite(str(layout_path), vis_layout)
    ordered_blocks = [
        {
            "bbox": region.get("bbox", [0, 0, 0, 0]),
            "col_index": 0,
            "col_count": 1,
            "spanned_cols": [0],
        }
        for region in sorted(
            (item for item in result if isinstance(item, dict)),
            key=lambda item: float(item.get("model_order") or 0.0),
        )
    ]
    vis_order_columns = draw_sorted_layout(image, ordered_blocks)
    cv2.imwrite(str(order_columns_path), vis_order_columns)


def _native_docx_table_count(path: Path) -> int:
    if not path.exists():
        raise RuntimeError(f"DOCX was not produced: {path}")
    document = DocxDocument(path)
    return len(
        document.element.body.xpath(
            './/w:tbl[w:tblPr/w:tblCaption[@w:val="docflow-native-table"] or '
            'w:tblPr/w:tblStyle[@w:val="TableGrid"]]'
        )
    )


def _validate_content_integrity(evidence, analysis) -> None:
    source_ids = Counter(
        item.evidence_id
        for page in evidence.pages
        for item in page.items
    )
    resolved_ids = Counter(
        source_id
        for page in analysis.pages
        for element in page.elements
        for source_id in element.source_ids
    )
    if source_ids != resolved_ids:
        missing = sorted((source_ids - resolved_ids).elements())
        duplicated = sorted((resolved_ids - source_ids).elements())
        raise RuntimeError(
            f"Document Analysis provenance mismatch: missing={missing}, duplicated={duplicated}"
        )


def run_sample(
    engine,
    adapter: PaddleAdapter,
    analyzer: DocumentAnalyzer,
    sample_path: Path,
    sample_layout,
    formats: list[str],
    pdf_dpi: int,
    save_debug_vis: bool,
) -> int:
    pages = expand_to_pages(sample_path, dpi=pdf_dpi) if is_pdf_file(sample_path) else expand_to_pages(sample_path)
    evidence_pages = []
    for page_index, image in [(idx, img) for idx, (_page_name, img) in enumerate(pages)]:
        height, width = image.shape[:2]
        print(f"[页面] {sample_path.name} p{page_index + 1} ({width}x{height})")
        evidence, result, elapsed = analyze_page(engine, adapter, image, page_index, str(sample_path))
        print(f"[分析] {sample_path.name} p{page_index + 1}: 检测到 {len(result)} 个区域，耗时 {elapsed:.2f}s")
        print_regions(result)
        if save_debug_vis:
            write_json(sample_layout.sample_dir / "raw_result.json", summarize_raw_result(result))
            save_debug_images(image, result, sample_layout, page_index)
        evidence_pages.extend(evidence.pages)

    evidence = RecognitionEvidence(tuple(evidence_pages), source_file=str(sample_path))
    write_json(sample_layout.recognition_path, evidence.to_dict())
    analysis = analyzer.analyze(evidence)
    _validate_content_integrity(evidence, analysis)
    write_json(sample_layout.json_path, analysis.to_dict())
    reflow_plan = ReflowPlanner().plan(analysis)
    write_json(sample_layout.render_plan_path, reflow_plan.to_dict())

    native_docx_table_count = 0
    if "docx" in formats or "pdf" in formats:
        ReflowDocxRenderer().render(reflow_plan, str(sample_layout.docx_path))
        native_docx_table_count = _native_docx_table_count(sample_layout.docx_path)
        semantic_table_count = sum(
            element.kind == "table_group"
            for page in analysis.pages
            for element in page.elements
        )
        if native_docx_table_count < semantic_table_count:
            raise RuntimeError(
                f"native table loss: expected at least {semantic_table_count}, got {native_docx_table_count}"
            )
    if "markdown" in formats:
        ReflowMarkdownRenderer().render(analysis, str(sample_layout.markdown_path))
    if "pdf" in formats:
        subprocess.run(
            [
                find_libreoffice(),
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(sample_layout.sample_dir),
                str(sample_layout.docx_path),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        if not sample_layout.pdf_path.exists():
            raise RuntimeError("LibreOffice conversion completed without producing the PDF")
        with fitz.open(sample_layout.pdf_path) as pdf:
            if len(pdf) != len(evidence_pages):
                raise RuntimeError(
                    f"Page Budget exceeded: source={len(evidence_pages)}, rendered={len(pdf)}"
                )
    return len(evidence_pages), native_docx_table_count


def main() -> int:
    parser = argparse.ArgumentParser(description="DocFlow 全流程测试运行器")
    parser.add_argument("--input", "-i", default=None, help="输入样本文件或目录。默认: ../dataset")
    parser.add_argument("--output", "-o", default=None, help="结果根目录。默认: ../test-result")
    parser.add_argument("--formats", "-f", default="docx,markdown,pdf", help="逗号分隔输出格式：docx,markdown,pdf")
    parser.add_argument("--pdf-dpi", type=int, default=200, help="PDF 转图像的 DPI，默认 200")
    parser.add_argument("--no-debug-vis", action="store_true", help="关闭 debug 可视化图导出")
    parser.add_argument(
        "--layout-model",
        default="pp-doclayout-v3",
        choices=["pp-doclayout-v3", "PP-DocLayoutV3", "doclayout_yolo", "picodet-l_layout_17cls", "pp-doclayout-m"],
        help="选择版面分析模型。",
    )
    args = parser.parse_args()

    paths = RuntimePaths.discover(layout_model_name=args.layout_model)
    input_path = resolve_cli_path(args.input, paths.dataset_root)
    output_root = resolve_cli_path(args.output, paths.result_root)
    formats = parse_formats(args.formats)

    ensure_runtime_paths(paths)
    if "pdf" in formats and find_libreoffice() is None:
        raise RuntimeError("你请求了 PDF 输出，但系统 PATH 中未找到 LibreOffice/soffice。")

    bootstrap_import_paths(paths)
    samples = collect_samples(input_path)
    if not samples:
        print(f"未找到可处理样本：{input_path}")
        return 1

    run_layout = ResultRunLayout.create(output_root)
    print_list("样本列表：", [str(path) for path in samples])
    print(f"输出格式：{formats}")
    print(f"运行目录：{run_layout.run_dir}")
    print(f"版面模型：{paths.layout_model_spec.name} -> {paths.layout_model}")
    engine = make_engine(paths, run_layout.runtime_dir)
    adapter = PaddleAdapter()
    analyzer = DocumentAnalyzer(FontClassifier(str(paths.models_root / "font" / "mobilenetv3.ckpt")))
    failures: list[str] = []
    sample_records = []
    total_pages = 0
    quality_summary = {
        "native_docx_table_count": 0,
        "analysis_diagnostics_total": 0,
        "fit_scaled_pages": 0,
    }
    strategy_stats = {
        "flow_kinds": {},
    }
    for sample_path in samples:
        sample_layout = run_layout.create_sample(sample_path)
        try:
            page_count = run_sample(
                engine=engine,
                adapter=adapter,
                analyzer=analyzer,
                sample_path=sample_path,
                sample_layout=sample_layout,
                formats=formats,
                pdf_dpi=args.pdf_dpi,
                save_debug_vis=not args.no_debug_vis,
            )
            if isinstance(page_count, tuple):
                page_count, native_docx_table_count = page_count
            else:
                native_docx_table_count = 0
            total_pages += page_count
            quality_summary["native_docx_table_count"] += native_docx_table_count
            analysis_payload = json.loads(sample_layout.json_path.read_text(encoding="utf-8"))
            diagnostic_count = sum(len(page.get("diagnostics") or []) for page in analysis_payload.get("pages") or [])
            quality_summary["analysis_diagnostics_total"] += diagnostic_count
            plan_payload = json.loads(sample_layout.render_plan_path.read_text(encoding="utf-8"))
            for page in plan_payload.get("pages") or []:
                quality_summary["fit_scaled_pages"] += int(float(page.get("fit_scale", 1.0)) < 1.0)
                for section in page.get("sections") or []:
                    key = section.get("kind", "unknown")
                    strategy_stats["flow_kinds"][key] = strategy_stats["flow_kinds"].get(key, 0) + 1
            sample_records.append(
                {
                    "sample_key": sample_layout.sample_key,
                    "source_path": str(sample_path),
                    "sample_dir": str(sample_layout.sample_dir),
                    "page_count": page_count,
                    "analysis_diagnostic_count": diagnostic_count,
                    "native_docx_table_count": native_docx_table_count,
                    "render_plan_path": str(sample_layout.render_plan_path),
                    "status": "ok",
                }
            )
        except Exception as ex:
            failures.append(f"{sample_path}: {ex}")
            sample_records.append(
                {
                    "sample_key": sample_layout.sample_key,
                    "source_path": str(sample_path),
                    "sample_dir": str(sample_layout.sample_dir),
                    "page_count": 0,
                    "analysis_diagnostic_count": 0,
                    "native_docx_table_count": 0,
                    "render_plan_path": str(sample_layout.render_plan_path),
                    "status": "failed",
                    "error": str(ex),
                }
            )
            print(f"[错误] {sample_path}: {ex}")

    run_layout.write_run_manifest(
        build_main_run_manifest(
            run_layout=run_layout,
            input_path=input_path,
            formats=formats,
            layout_model_dir=paths.layout_model,
            samples=sample_records,
            total_pages=total_pages,
            quality_summary=quality_summary,
            strategy_stats=strategy_stats,
            failures=failures,
        )
    )

    print(f"\n完成页数：{total_pages}")
    if failures:
        print_list("失败列表：", failures)
        return 2
    print("全部样本处理成功。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
