"""全流程测试入口：图片/PDF -> PaddleOCR -> 合并 JSON -> DocFlow 文档输出。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
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

DEFAULT_LAYOUT_SCORE_THRESHOLD = 0.50
DEFAULT_DOCLAYOUT_YOLO_SCORE_THRESHOLD = 0.18
RAW_RESULT_PREVIEW_MAX_TEXT = 300


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


def resolve_layout_score_threshold(layout_model_dir: Path) -> str:
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

    if "doclayout_yolo" in layout_model_dir.name.lower():
        return f"{DEFAULT_DOCLAYOUT_YOLO_SCORE_THRESHOLD:.2f}"
    return f"{DEFAULT_LAYOUT_SCORE_THRESHOLD:.2f}"


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
    layout_score_threshold = resolve_layout_score_threshold(paths.layout_model)

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
    from docflow.utils.visualization import (
        draw_layout_ocr,
        draw_reading_order_comparison,
        draw_reading_order_map,
        draw_sorted_layout,
        extract_sorted_blocks,
    )

    def _remap_to_source_bboxes(ordered_blocks: list[dict], source_page: dict) -> list[dict]:
        def _should_use_source_bbox(current_bbox: list[float], source_bbox: list[float]) -> bool:
            if len(current_bbox) != 4 or len(source_bbox) != 4:
                return False
            curr_w = max(0.0, float(current_bbox[2]) - float(current_bbox[0]))
            curr_h = max(0.0, float(current_bbox[3]) - float(current_bbox[1]))
            src_w = max(0.0, float(source_bbox[2]) - float(source_bbox[0]))
            src_h = max(0.0, float(source_bbox[3]) - float(source_bbox[1]))
            curr_area = curr_w * curr_h
            src_area = src_w * src_h
            if curr_area <= 1.0 or src_area <= 1.0:
                return True

            edge_delta = max(
                abs(float(current_bbox[0]) - float(source_bbox[0])),
                abs(float(current_bbox[1]) - float(source_bbox[1])),
                abs(float(current_bbox[2]) - float(source_bbox[2])),
                abs(float(current_bbox[3]) - float(source_bbox[3])),
            )
            area_ratio = curr_area / src_area
            # 仅在 bbox 基本一致时回写 source 坐标，避免把 flow 内合并/裁剪后的
            # 最终几何退回到原始 OCR 框，导致调试图“最后一行掉出框外”。
            return edge_delta <= 12.0 and 0.85 <= area_ratio <= 1.15

        source_by_id = {
            str(block.get("id", "")): block
            for block in source_page.get("blocks", [])
        }
        remapped: list[dict] = []
        for block in ordered_blocks:
            source = source_by_id.get(str(block.get("id", "")))
            remapped_block = dict(block)
            if (
                source
                and isinstance(source.get("bbox"), list)
                and len(source["bbox"]) == 4
                and _should_use_source_bbox(
                    list(remapped_block.get("bbox", []) or []),
                    source["bbox"],
                )
            ):
                remapped_block["bbox"] = [float(v) for v in source["bbox"]]
            remapped.append(remapped_block)
        return remapped

    sample_layout.debug_dir.mkdir(parents=True, exist_ok=True)
    layout_path = sample_layout.debug_image_path(page_index, "layout_ocr")
    sorted_path = sample_layout.debug_image_path(page_index, "sorted_layout")
    legacy_order_path = sample_layout.debug_image_path(page_index, "reading_order_legacy")
    xycutpp_order_path = sample_layout.debug_image_path(page_index, "reading_order_xycutpp")
    compare_path = sample_layout.debug_image_path(page_index, "reading_order_compare")
    vis_layout = draw_layout_ocr(image, result, font_path=None, show_text_preview=True)
    cv2.imwrite(str(layout_path), vis_layout)

    legacy_pipeline = RecoveryPipeline(
        config=replace(pipeline.config, reading_order_strategy="legacy")
    )
    xycutpp_pipeline = RecoveryPipeline(
        config=replace(pipeline.config, reading_order_strategy="xycutpp_hybrid")
    )
    actual_document = pipeline.build_document(deepcopy(page_document))
    legacy_document = legacy_pipeline.build_document(deepcopy(page_document))
    xycutpp_document = xycutpp_pipeline.build_document(deepcopy(page_document))
    source_page = (page_document.get("pages") or [{}])[0]

    sorted_blocks = extract_sorted_blocks(actual_document, page_index=0)
    legacy_blocks = _remap_to_source_bboxes(extract_sorted_blocks(legacy_document, page_index=0), source_page)
    xycutpp_blocks = _remap_to_source_bboxes(extract_sorted_blocks(xycutpp_document, page_index=0), source_page)

    vis_sorted = draw_sorted_layout(image, sorted_blocks)
    legacy_map = draw_reading_order_map(
        image,
        legacy_blocks,
        title="Legacy Reading Order",
    )
    xycutpp_map = draw_reading_order_map(
        image,
        xycutpp_blocks,
        title="XY-Cut++ Hybrid Reading Order",
    )
    compare_map = draw_reading_order_comparison(
        image,
        legacy_blocks,
        xycutpp_blocks,
    )

    cv2.imwrite(str(sorted_path), vis_sorted)
    cv2.imwrite(str(legacy_order_path), legacy_map)
    cv2.imwrite(str(xycutpp_order_path), xycutpp_map)
    cv2.imwrite(str(compare_path), compare_map)


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
            write_json(sample_layout.sample_dir / "raw_result.json", summarize_raw_result(result))
            save_debug_images(pipeline, image, result, page_document, sample_layout, page_index)
        page_documents.append(page_document)

    merged_document = merge_page_documents(page_documents, source_path=str(sample_path))
    document = pipeline.build_document(merged_document)
    render_plan = build_render_plan(document, output_format="docx")
    write_json(sample_layout.render_plan_path, render_plan)

    # 将 RenderPlan 的 per-page render_mode 注入 document 和 merged_document，
    # 使 docx renderer 和 pipeline.recover() 中的 renderer 都可以消费该提示。
    for page_info in render_plan.get("pages", []):
        idx = page_info["page_index"]
        render_mode = page_info.get("render_mode", "")
        # 注入到 merged_document（供 pipeline.recover 使用）
        if idx < len(merged_document["pages"]):
            page_entry = merged_document["pages"][idx]
            if page_entry.get("attributes") is None:
                page_entry["attributes"] = {}
            page_entry["attributes"]["render_mode"] = render_mode
        # 注入到 document（供 docx renderer 使用）
        if idx < len(document.pages):
            if document.pages[idx].attributes is None:
                document.pages[idx].attributes = {}
            document.pages[idx].attributes["render_mode"] = render_mode

    native_docx_table_count = 0
    if "docx" in formats:
        renderer = pipeline._get_renderer("docx")
        renderer.render(document, str(sample_layout.docx_path))
        # 将 renderer 记录的 render_fit 与 style_inferred 回写到 JSON（设计文档 §2.2）
        for page_index, page in enumerate(document.pages):
            if page_index >= len(merged_document["pages"]):
                continue
            page_entry = merged_document["pages"][page_index]
            if page_entry.get("attributes") is None:
                page_entry["attributes"] = {}
            attrs = page_entry["attributes"]
            render_fit = page.attributes.get("render_fit")
            if render_fit:
                attrs["render_fit"] = render_fit
            style_inferred = page.attributes.get("style_inferred")
            if style_inferred:
                attrs["style_inferred"] = style_inferred
        write_json(sample_layout.json_path, merged_document)
        native_docx_table_count = _native_docx_table_count(sample_layout.docx_path)
    else:
        write_json(sample_layout.json_path, merged_document)
    if "markdown" in formats:
        pipeline.recover(merged_document, str(sample_layout.markdown_path), format="markdown")
    if "pdf" in formats:
        pipeline.recover(merged_document, str(sample_layout.pdf_path), format="pdf")
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
    engine = make_engine(paths, run_layout.runtime_dir)
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
