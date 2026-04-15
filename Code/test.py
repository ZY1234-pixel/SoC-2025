"""全流程测试入口：图片/PDF -> PaddleOCR -> 合并 JSON -> DocFlow 文档输出。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
from docx import Document as DocxDocument

CODE_ROOT = Path(__file__).resolve().parent
DOCFLOW_SRC_ROOT = CODE_ROOT / "docflow_src"
docflow_src_str = str(DOCFLOW_SRC_ROOT)
if docflow_src_str not in sys.path:
    sys.path.insert(0, docflow_src_str)

from dataset import collect_samples, is_pdf_file
from model import RuntimePaths
from preprocess import expand_to_pages
from utils import ensure_runtime_paths, find_libreoffice, parse_formats, print_list
from docflow.adapters.paddle_adapter import PaddleAdapter
from docflow.pipeline import RecoveryPipeline
from docflow.utils.result_layout import (
    ResultRunLayout,
    build_main_run_manifest,
    merge_page_documents,
    write_json,
)
from docflow.utils.render_plan import build_render_plan


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


def build_layout_dict_from_inference(layout_model_dir: Path, fallback_dict_path: Path, out_dir: Path) -> Path:
    """按模型 inference.yml 自动生成 layout 字典文件。"""
    inference_yml = layout_model_dir / "inference.yml"
    if not inference_yml.is_file():
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
                        out_path = out_dir / f"layout_{layout_model_dir.name.replace('-', '_')}_dict.txt"
                        out_path.write_text("\n".join(ordered) + "\n", encoding="utf-8")
                        return out_path
            except Exception:
                pass
        if "doclayout_yolo" in layout_model_dir.name.lower():
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"layout_{layout_model_dir.name.replace('-', '_')}_dict.txt"
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
    out_path = out_dir / f"layout_{layout_model_dir.name.replace('-', '_')}_dict.txt"
    out_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    return out_path


def make_engine(paths: RuntimePaths, layout_dict_dir: Path):
    """初始化 PaddleOCR 的 StructureSystem 引擎。"""
    from ppstructure.utility import parse_args
    from ppstructure.predict_system import StructureSystem

    ppstructure_dir = paths.paddle_root / "ppstructure"
    fallback_layout_dict = paths.paddle_root / "ppocr" / "utils" / "dict" / "layout_dict" / "layout_cdla_dict.txt"
    rec_char_dict = paths.paddle_root / "ppocr" / "utils" / "dict" / "ppocrv5_dict.txt"
    table_char_dict = paths.paddle_root / "ppocr" / "utils" / "dict" / "table_structure_dict_ch.txt"
    layout_dict = build_layout_dict_from_inference(paths.layout_model, fallback_layout_dict, layout_dict_dir)

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
        "0.25" if "doclayout_yolo" in paths.layout_model.name.lower() else "0.5",
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


def analyze_page(engine, adapter: PaddleAdapter, image, page_index: int, source_path: str) -> tuple[dict, list, float]:
    started_at = time.time()
    result, _ = engine(image, img_idx=page_index)
    elapsed = time.time() - started_at
    page_document = adapter.convert(result, image, img_idx=page_index)
    page_document["pages"][0]["image_path"] = source_path
    return page_document, result, elapsed


def save_debug_images(
    pipeline: RecoveryPipeline,
    image,
    result: list,
    page_document: dict,
    sample_layout,
    page_index: int,
) -> None:
    from docflow.utils.visualization import draw_layout_ocr, draw_sorted_layout, extract_sorted_blocks

    sample_layout.debug_dir.mkdir(parents=True, exist_ok=True)
    layout_path = sample_layout.debug_image_path(page_index, "layout_ocr")
    sorted_path = sample_layout.debug_image_path(page_index, "sorted_layout")
    vis_layout = draw_layout_ocr(image, result, font_path=None, show_text_preview=True)
    cv2.imwrite(str(layout_path), vis_layout)
    document = pipeline.build_document(page_document)
    sorted_blocks = extract_sorted_blocks(document, page_index=0)
    vis_sorted = draw_sorted_layout(image, sorted_blocks)
    cv2.imwrite(str(sorted_path), vis_sorted)


def write_sample_manifest(sample_layout, sample_path: Path, page_count: int, formats: list[str]) -> None:
    write_json(
        sample_layout.sample_manifest_path,
        {
            "sample_key": sample_layout.sample_key,
            "source_path": str(sample_path),
            "source_name": sample_path.name,
            "page_count": page_count,
            "formats": formats,
            "artifacts": {
                "json": str(sample_layout.json_path),
                "render_plan": str(sample_layout.render_plan_path),
                "docx": str(sample_layout.docx_path) if "docx" in formats else None,
                "markdown": str(sample_layout.markdown_path) if "markdown" in formats else None,
                "pdf": str(sample_layout.pdf_path) if "pdf" in formats else None,
                "markdown_assets": str(sample_layout.markdown_assets_dir) if "markdown" in formats else None,
                "debug_dir": str(sample_layout.debug_dir),
            },
        },
    )


def _sample_cleanup_removed_count(merged_document: dict) -> int:
    total = 0
    for page in merged_document.get("pages") or []:
        attrs = page.get("attributes") or {}
        total += int(attrs.get("cleanup_removed_count") or 0)
    return total


def _native_docx_table_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        document = DocxDocument(path)
        return len(document.tables)
    except Exception:
        return 0


def run_sample(
    engine,
    adapter: PaddleAdapter,
    pipeline: RecoveryPipeline,
    sample_path: Path,
    sample_layout,
    formats: list[str],
    pdf_dpi: int,
    save_debug_vis: bool,
) -> int:
    pages = expand_to_pages(sample_path, dpi=pdf_dpi) if is_pdf_file(sample_path) else expand_to_pages(sample_path)
    page_documents = []
    for page_index, image in [(idx, img) for idx, (_page_name, img) in enumerate(pages)]:
        height, width = image.shape[:2]
        print(f"[页面] {sample_path.name} p{page_index + 1} ({width}x{height})")
        page_document, result, elapsed = analyze_page(engine, adapter, image, page_index, str(sample_path))
        print(f"[分析] {sample_path.name} p{page_index + 1}: 检测到 {len(result)} 个区域，耗时 {elapsed:.2f}s")
        print_regions(result)
        if save_debug_vis:
            save_debug_images(pipeline, image, result, page_document, sample_layout, page_index)
        page_documents.append(page_document)

    merged_document = merge_page_documents(page_documents, source_path=str(sample_path))
    document = pipeline.build_document(merged_document)
    write_json(sample_layout.json_path, merged_document)
    write_json(sample_layout.render_plan_path, build_render_plan(document, output_format="docx"))
    native_docx_table_count = 0
    if "docx" in formats:
        pipeline.recover(merged_document, str(sample_layout.docx_path), format="docx")
        native_docx_table_count = _native_docx_table_count(sample_layout.docx_path)
    if "markdown" in formats:
        pipeline.recover(merged_document, str(sample_layout.markdown_path), format="markdown")
    if "pdf" in formats:
        pipeline.recover(merged_document, str(sample_layout.pdf_path), format="pdf")
    write_sample_manifest(sample_layout, sample_path, page_count=len(page_documents), formats=formats)
    return len(page_documents), native_docx_table_count


def main() -> int:
    parser = argparse.ArgumentParser(description="DocFlow 全流程测试运行器")
    parser.add_argument("--input", "-i", default=None, help="输入样本文件或目录。默认: ../dataset")
    parser.add_argument("--output", "-o", default=None, help="结果根目录。默认: ../test-result")
    parser.add_argument("--formats", "-f", default="docx,markdown,pdf", help="逗号分隔输出格式：docx,markdown,pdf")
    parser.add_argument("--pdf-dpi", type=int, default=200, help="PDF 转图像的 DPI，默认 200")
    parser.add_argument("--no-debug-vis", action="store_true", help="关闭 debug 可视化图导出")
    args = parser.parse_args()

    paths = RuntimePaths.discover()
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
    engine = make_engine(paths, run_layout.layout_dict_dir)
    adapter = PaddleAdapter()
    pipeline = RecoveryPipeline()

    failures: list[str] = []
    sample_records = []
    total_pages = 0
    quality_summary = {
        "layout_profiles": {},
        "native_docx_table_count": 0,
        "cleanup_removed_total": 0,
    }
    strategy_stats = {
        "rendering_strategies": {},
    }
    for sample_path in samples:
        sample_layout = run_layout.create_sample(sample_path)
        try:
            page_count = run_sample(
                engine=engine,
                adapter=adapter,
                pipeline=pipeline,
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
            cleanup_removed = _sample_cleanup_removed_count(
                json.loads(sample_layout.json_path.read_text(encoding="utf-8"))
            )
            quality_summary["cleanup_removed_total"] += cleanup_removed
            quality_summary["native_docx_table_count"] += native_docx_table_count
            render_plan = json.loads(sample_layout.render_plan_path.read_text(encoding="utf-8"))
            for key, value in render_plan["summary"]["layout_profiles"].items():
                quality_summary["layout_profiles"][key] = quality_summary["layout_profiles"].get(key, 0) + value
            for key, value in render_plan["summary"]["rendering_strategies"].items():
                strategy_stats["rendering_strategies"][key] = strategy_stats["rendering_strategies"].get(key, 0) + value
            sample_records.append(
                {
                    "sample_key": sample_layout.sample_key,
                    "source_path": str(sample_path),
                    "sample_dir": str(sample_layout.sample_dir),
                    "page_count": page_count,
                    "cleanup_removed_count": cleanup_removed,
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
                    "cleanup_removed_count": 0,
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
