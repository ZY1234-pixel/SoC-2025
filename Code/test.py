"""全流程测试入口：图片/PDF -> PaddleOCR -> 合并 JSON -> DocFlow 文档输出。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from difflib import SequenceMatcher
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
from model import LayoutModelSpec, RuntimePaths
from preprocess import expand_to_pages
from utils import ensure_runtime_paths, find_libreoffice, parse_formats, print_list
from docflow.adapters.paddle_adapter import PaddleAdapter
from docflow.analysis import DocumentAnalyzer
from docflow.model.stages import RecognitionEvidence
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
DEFAULT_PP_DOCLAYOUT_V3_SCORE_THRESHOLD = 0.50
DEFAULT_DOCLAYOUT_YOLO_SCORE_THRESHOLD = 0.18
RAW_RESULT_PREVIEW_MAX_TEXT = 300
OCR_RECHECK_TYPES = {"title", "text", "header"}
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


def _converted_blocks_to_debug_regions(page_document: dict) -> list[dict]:
    pages = page_document.get("pages") if isinstance(page_document, dict) else None
    if not pages:
        return []
    page = pages[0] or {}
    regions: list[dict] = []
    for block in page.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        region: dict = {
            "type": block.get("category", block.get("type", "?")),
            "bbox": block.get("bbox", [0, 0, 0, 0]),
            "score": float(block.get("confidence", block.get("score", 0.0)) or 0.0),
            "res": [],
        }
        if block.get("text_lines"):
            converted_lines = []
            for line in block.get("text_lines") or []:
                if not isinstance(line, dict):
                    continue
                converted_lines.append(
                    {
                        "text": line.get("text", ""),
                        "confidence": line.get("confidence", 1.0),
                        "text_region": line.get("poly"),
                    }
                )
            region["res"] = converted_lines
        elif block.get("text"):
            region["res"] = [{"text": block.get("text", "")}]
        regions.append(region)
    return regions


def _line_texts(region: dict) -> list[str]:
    texts: list[str] = []
    for item in region.get("res") or []:
        if isinstance(item, dict):
            text = str(item.get("text", "") or "").strip()
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            rhs = item[1]
            text = str(rhs[0] if isinstance(rhs, (list, tuple)) and rhs else rhs).strip()
        else:
            text = ""
        if text:
            texts.append(text)
    return texts


def _ocr_lines_from_crop(engine, crop) -> list[dict]:
    boxes, recs, _ = engine.text_system(crop)
    if boxes is None or recs is None:
        return []
    lines: list[dict] = []
    for box, rec in zip(boxes, recs):
        text = str(rec[0] or "").strip()
        if not text:
            continue
        lines.append(
            {
                "text": text,
                "confidence": float(rec[1] or 0.0),
                "text_region": box.tolist() if hasattr(box, "tolist") else box,
            }
        )
    return lines


def _offset_text_regions(lines: list[dict], x_offset: int, y_offset: int) -> list[dict]:
    shifted: list[dict] = []
    for line in lines:
        item = dict(line)
        region = item.get("text_region")
        if isinstance(region, list):
            item["text_region"] = [
                [float(point[0]) + x_offset, float(point[1]) + y_offset]
                for point in region
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
        shifted.append(item)
    return shifted


def _clean_text_len(lines: list[dict] | list[str]) -> int:
    total = 0
    for item in lines:
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        total += len("".join(str(text).split()))
    return total


def _should_accept_ocr_recheck(region: dict, candidate_lines: list[dict]) -> bool:
    if not candidate_lines:
        return False
    current_texts = _line_texts(region)
    current_len = _clean_text_len(current_texts)
    candidate_len = _clean_text_len(candidate_lines)
    if current_texts and abs(len(candidate_lines) - len(current_texts)) > 1:
        return False
    avg_conf = sum(float(line.get("confidence", 0.0) or 0.0) for line in candidate_lines) / len(candidate_lines)
    if avg_conf < 0.90:
        return False
    if candidate_len > current_len:
        return True
    current_joined = "".join(current_texts)
    candidate_joined = "".join(str(line.get("text", "") or "") for line in candidate_lines)
    similarity = SequenceMatcher(None, current_joined, candidate_joined).ratio()
    return candidate_joined != current_joined and similarity >= 0.72


def _merge_rechecked_lines(region: dict, candidate_lines: list[dict]) -> list[dict]:
    current = [
        dict(item) for item in (region.get("res") or [])
        if isinstance(item, dict) and str(item.get("text", "") or "").strip()
    ]
    if not current:
        return candidate_lines

    def _y_center(line: dict) -> float | None:
        region = line.get("text_region")
        if not isinstance(region, list) or not region:
            return None
        ys = [
            float(point[1])
            for point in region
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if not ys:
            return None
        return (min(ys) + max(ys)) * 0.5

    merged: list[dict] = []
    changed = False
    used_candidates: set[int] = set()
    for old in current:
        old_text = str(old.get("text", "") or "").strip()
        old_y = _y_center(old)
        best_index = -1
        best_score = -1.0
        for idx, candidate in enumerate(candidate_lines):
            if idx in used_candidates:
                continue
            new_text = str(candidate.get("text", "") or "").strip()
            if not new_text:
                continue
            similarity = SequenceMatcher(None, old_text, new_text).ratio()
            if similarity < 0.55:
                continue
            new_y = _y_center(candidate)
            y_score = 1.0
            if old_y is not None and new_y is not None:
                y_score = max(0.0, 1.0 - abs(old_y - new_y) / 64.0)
            score = similarity * 0.75 + y_score * 0.25
            if score > best_score:
                best_score = score
                best_index = idx

        if best_index < 0:
            merged.append(old)
            continue

        new = candidate_lines[best_index]
        new_text = str(new.get("text", "") or "").strip()
        old_conf = float(old.get("confidence", 0.0) or 0.0)
        new_conf = float(new.get("confidence", 0.0) or 0.0)
        similarity = SequenceMatcher(None, old_text, new_text).ratio()
        use_new = (
            new_text
            and new_text != old_text
            and similarity >= 0.60
            and (
                len("".join(new_text.split())) > len("".join(old_text.split()))
                or new_conf >= old_conf + 0.02
                or (new_conf >= 0.95 and old_conf < 0.95)
            )
        )
        if use_new:
            merged.append(new)
            changed = True
            used_candidates.add(best_index)
        else:
            merged.append(old)

    def _line_key(line: dict) -> tuple[float, float]:
        text_region = line.get("text_region")
        if not isinstance(text_region, list) or not text_region:
            return (float("inf"), float("inf"))
        xs = [
            float(point[0])
            for point in text_region
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        ys = [
            float(point[1])
            for point in text_region
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        return ((min(ys) + max(ys)) * 0.5, min(xs)) if xs and ys else (float("inf"), float("inf"))

    existing_texts = {str(line.get("text", "") or "").strip() for line in merged if str(line.get("text", "") or "").strip()}
    for idx, candidate in enumerate(candidate_lines):
        if idx in used_candidates:
            continue
        candidate_text = str(candidate.get("text", "") or "").strip()
        if not candidate_text or candidate_text in existing_texts:
            continue
        if any(SequenceMatcher(None, candidate_text, existing).ratio() >= 0.86 for existing in existing_texts):
            continue
        if float(candidate.get("confidence", 0.0) or 0.0) < 0.88:
            continue
        merged.append(candidate)
        existing_texts.add(candidate_text)
        changed = True

    if changed:
        merged.sort(key=_line_key)
        return merged
    return current


def recheck_text_ocr_with_preprocessing(engine, image, result: list) -> int:
    """用扩边二值化 OCR 复核容易漏行/漏字符的标题和页首短文本块。"""
    if not result:
        return 0
    h, w = image.shape[:2]
    changed = 0
    for region in result:
        region_type = str(region.get("type", "") or "").lower()
        if region_type not in OCR_RECHECK_TYPES:
            continue
        bbox = region.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        texts = _line_texts(region)
        if not texts or len(texts) > 5:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        is_top_text = region_type in {"text", "header"} and y1 <= max(96, int(round(h * 0.07)))
        if region_type not in {"title"} and not is_top_text:
            continue
        pad_x = max(16, int(round(bw * 0.08)))
        pad_y = max(8, int(round(bh * 0.12)))
        pad_top = max(pad_y, 36, int(round(bh * 0.45))) if is_top_text else pad_y
        ex1 = max(0, x1 - pad_x)
        ey1 = max(0, y1 - pad_top)
        ex2 = min(w, x2 + pad_x)
        ey2 = min(h, y2 + pad_y)
        crop = image[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        candidate_lines = _ocr_lines_from_crop(engine, binary_bgr)
        if not _should_accept_ocr_recheck(region, candidate_lines):
            continue
        shifted_candidates = _offset_text_regions(candidate_lines, ex1, ey1)
        region["res"] = _merge_rechecked_lines(region, shifted_candidates)
        xs = []
        ys = []
        for line in region["res"]:
            for point in line.get("text_region") or []:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    xs.append(float(point[0]))
                    ys.append(float(point[1]))
        if xs and ys:
            region["bbox"] = [max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys))]
        region.setdefault("attributes", {})
        region["ocr_rechecked"] = True
        changed += 1
    return changed


def analyze_page(engine, adapter: PaddleAdapter, image, page_index: int, source_path: str) -> tuple[dict, RecognitionEvidence, list, float]:
    started_at = time.time()
    result, _ = engine(image, img_idx=page_index)
    recheck_text_ocr_with_preprocessing(engine, image, result)
    elapsed = time.time() - started_at
    evidence = adapter.collect_evidence(result, image, img_idx=page_index, source_file=source_path)
    page_document = adapter.convert(result, image, img_idx=page_index)
    page_document["pages"][0]["image_path"] = source_path
    return page_document, evidence, result, elapsed


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
    order_columns_path = sample_layout.debug_image_path(page_index, "reading_order_columns")
    clean_regions = _converted_blocks_to_debug_regions(page_document)
    vis_layout = draw_layout_ocr(
        image,
        clean_regions or result,
        font_path=None,
        show_text_preview=True,
    )
    cv2.imwrite(str(layout_path), vis_layout)

    actual_document = pipeline.build_document(deepcopy(page_document))
    source_page = (page_document.get("pages") or [{}])[0]

    ordered_blocks = _remap_to_source_bboxes(extract_sorted_blocks(actual_document, page_index=0), source_page)
    vis_order_columns = draw_sorted_layout(image, ordered_blocks)
    cv2.imwrite(str(order_columns_path), vis_order_columns)


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
    evidence_pages = []
    for page_index, image in [(idx, img) for idx, (_page_name, img) in enumerate(pages)]:
        height, width = image.shape[:2]
        print(f"[页面] {sample_path.name} p{page_index + 1} ({width}x{height})")
        page_document, evidence, result, elapsed = analyze_page(engine, adapter, image, page_index, str(sample_path))
        print(f"[分析] {sample_path.name} p{page_index + 1}: 检测到 {len(result)} 个区域，耗时 {elapsed:.2f}s")
        print_regions(result)
        if save_debug_vis:
            write_json(sample_layout.sample_dir / "raw_result.json", summarize_raw_result(result))
            save_debug_images(pipeline, image, result, page_document, sample_layout, page_index)
        page_documents.append(page_document)
        evidence_pages.extend(evidence.pages)

    evidence = RecognitionEvidence(tuple(evidence_pages), source_file=str(sample_path))
    write_json(sample_layout.recognition_path, evidence.to_dict())
    analysis = DocumentAnalyzer().analyze(evidence)
    write_json(sample_layout.analysis_path, analysis.to_dict())
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
        # 将 build/render 阶段写入 page.attributes 的信息回写到 JSON
        # （如字体分类统计、render_fit 与 style_inferred）。
        for page_index, page in enumerate(document.pages):
            if page_index >= len(merged_document["pages"]):
                continue
            page_entry = merged_document["pages"][page_index]
            if page_entry.get("attributes") is None:
                page_entry["attributes"] = {}
            attrs = page_entry["attributes"]
            if page.attributes:
                attrs.update(page.attributes)
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
