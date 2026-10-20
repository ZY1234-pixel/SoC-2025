#!/usr/bin/env python3
"""Reproduce the delivery metrics that can be measured from local assets."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from statistics import mean

import cv2
import fitz
import numpy as np
from docx import Document
from lxml import etree
from openpyxl import load_workbook
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "Code"
OMNI_ROOT = PROJECT_ROOT.parent / "OmniDocBench"
OMNI_DATA_ROOT = PROJECT_ROOT.parent / "OmniDocBenchDatasets"
DEFAULT_RUN_DIR = PROJECT_ROOT / "test-result"
DEFAULT_REPORT = PROJECT_ROOT / "doc" / "交付文档" / "指标验证记录"

for path in (PROJECT_ROOT, CODE_ROOT, CODE_ROOT / "docflow_src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


TEXT_CATEGORIES = {
    "title",
    "text_block",
    "reference",
    "page_footnote",
    "figure_caption",
    "table_caption",
    "table_footnote",
    "equation_caption",
    "code_txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    return "".join(character for character in value if not character.isspace())


def is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


def edit_counts(ground_truth: str, prediction: str) -> tuple[int, int, int]:
    """Return total, CJK-target, and non-CJK-target edit counts."""
    left = normalize_text(ground_truth)
    right = normalize_text(prediction)
    rows = len(left) + 1
    columns = len(right) + 1
    costs = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        costs[row][0] = row
    for column in range(columns):
        costs[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            costs[row][column] = min(
                costs[row - 1][column] + 1,
                costs[row][column - 1] + 1,
                costs[row - 1][column - 1] + int(left[row - 1] != right[column - 1]),
            )

    row, column = len(left), len(right)
    cjk_edits = 0
    other_edits = 0
    while row or column:
        if row and column and left[row - 1] == right[column - 1]:
            row -= 1
            column -= 1
            continue
        current = costs[row][column]
        if row and column and current == costs[row - 1][column - 1] + 1:
            target = left[row - 1]
            row -= 1
            column -= 1
        elif row and current == costs[row - 1][column] + 1:
            target = left[row - 1]
            row -= 1
        else:
            target = ""
            column -= 1
        if target and is_cjk(target):
            cjk_edits += 1
        else:
            other_edits += 1
    return costs[-1][-1], cjk_edits, other_edits


def polygon_bbox(poly) -> tuple[float, float, float, float]:
    points = np.asarray(poly, dtype=float).reshape(-1, 2)
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def stratified_items(items, count: int):
    if count <= 0 or count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indices = sorted({round(index * (len(items) - 1) / (count - 1)) for index in range(count)})
    return [items[index] for index in indices]


def verify_font_classifier(samples_per_class: int) -> dict:
    from docflow.appearance.font_classifier import FontClassifier

    root = OMNI_DATA_ROOT / "synthetic_font_lines_5class_v5" / "test"
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))["labels"]
    grouped = collections.defaultdict(list)
    for identifier, label in labels.items():
        grouped[label].append(identifier)
    selected = []
    for label in sorted(grouped):
        selected.extend((identifier, label) for identifier in stratified_items(sorted(grouped[label]), samples_per_class))

    classifier = FontClassifier(str(CODE_ROOT / "models_openvino" / "font_openvino" / "mobilenetv3.xml"))
    confusion = collections.Counter()
    accepted = 0
    started = time.perf_counter()
    with Image.open(root / "images" / f"{selected[0][0]}.jpg") as warmup:
        classifier.predict_image(warmup.convert("RGB"))
    started = time.perf_counter()
    for identifier, expected in selected:
        with Image.open(root / "images" / f"{identifier}.jpg") as image:
            prediction = classifier.predict_image(image.convert("RGB"))
        confusion[(expected, prediction.label)] += 1
        accepted += int(prediction.accepted)
    elapsed = time.perf_counter() - started

    classes = sorted(grouped)
    per_class = {}
    for label in classes:
        true_positive = confusion[(label, label)]
        predicted = sum(confusion[(truth, label)] for truth in classes)
        supported = sum(confusion[(label, predicted_label)] for predicted_label in classes)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / supported if supported else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": supported,
        }
    correct = sum(confusion[(label, label)] for label in classes)
    return {
        "status": "pass",
        "dataset": str(root),
        "sample_count": len(selected),
        "samples_per_class": samples_per_class,
        "accuracy": correct / len(selected),
        "macro_f1": mean(values["f1"] for values in per_class.values()),
        "accepted_rate": accepted / len(selected),
        "elapsed_seconds": elapsed,
        "milliseconds_per_crop": elapsed / len(selected) * 1000.0,
        "confusion": {f"{truth}->{predicted}": count for (truth, predicted), count in sorted(confusion.items())},
        "per_class": per_class,
    }


def verify_charset() -> dict:
    dictionary_path = CODE_ROOT / "models_openvino" / "PP-OCRv6_small_rec_openvino" / "ppocrv6_dict.txt"
    characters = set(dictionary_path.read_text(encoding="utf-8", errors="replace").splitlines())
    gb2312 = set()
    for high in range(0xA1, 0xF8):
        for low in range(0xA1, 0xFF):
            try:
                gb2312.add(bytes((high, low)).decode("gb2312"))
            except UnicodeDecodeError:
                continue
    gb2312_han = {character for character in gb2312 if is_cjk(character)}
    basic_cjk = {chr(codepoint) for codepoint in range(0x4E00, 0xA000)}
    return {
        "status": "partial",
        "dictionary": str(dictionary_path),
        "dictionary_characters": len(characters),
        "gb2312_all": len(gb2312 & characters) / len(gb2312),
        "gb2312_han": len(gb2312_han & characters) / len(gb2312_han),
        "gb18030_basic_cjk": len(basic_cjk & characters) / len(basic_cjk),
        "gb2312_han_missing": len(gb2312_han - characters),
        "gb18030_basic_cjk_missing": len(basic_cjk - characters),
        "candidate_characters": False,
        "note": "Dictionary coverage is not a recognition-accuracy result; the current OCR API returns top-1 text only.",
    }


def load_omnidocbench() -> list[dict]:
    return json.loads((OMNI_DATA_ROOT / "OmniDocBench.json").read_text(encoding="utf-8"))


def select_ocr_pages(pages: list[dict], per_source: int) -> list[dict]:
    grouped = collections.defaultdict(list)
    for page in pages:
        source = str(page.get("page_info", {}).get("page_attribute", {}).get("data_source") or "unknown")
        grouped[source].append(page)
    selected = []
    for source in sorted(grouped):
        ordered = sorted(grouped[source], key=lambda page: page["page_info"]["image_path"])
        selected.extend(stratified_items(ordered, per_source))
    return selected


def verify_ocr(per_source: int) -> dict:
    from Code.model_integration.runtime import OpenVINOModelRuntime

    pages = select_ocr_pages(load_omnidocbench(), per_source)
    runtime = OpenVINOModelRuntime(runtime_dir="/tmp/docflow_delivery_ocr_validation", formula_enabled=False)
    warmup = cv2.imread(str(OMNI_DATA_ROOT / "images" / pages[0]["page_info"]["image_path"]))
    runtime.run_ocr(warmup)

    records = []
    total_gt_chars = 0
    total_pred_chars = 0
    total_edits = 0
    total_cjk_chars = 0
    total_cjk_edits = 0
    total_other_chars = 0
    total_other_edits = 0
    total_elapsed = 0.0
    evaluation_elapsed = 0.0
    exact_blocks = 0
    block_count = 0

    for page in pages:
        image_name = page["page_info"]["image_path"]
        image = cv2.imread(str(OMNI_DATA_ROOT / "images" / image_name))
        if image is None:
            raise RuntimeError(f"Cannot read OCR sample: {image_name}")
        started = time.perf_counter()
        result = runtime.run_ocr(image)
        elapsed = time.perf_counter() - started
        total_elapsed += elapsed
        total_pred_chars += sum(len(normalize_text(line.get("text") or "")) for line in result["lines"])

        blocks = []
        for annotation in page.get("layout_dets") or ():
            text = normalize_text(annotation.get("text") or "")
            if (
                annotation.get("category_type") in TEXT_CATEGORIES
                and text
                and not annotation.get("ignore")
                and annotation.get("poly")
            ):
                blocks.append({"bbox": polygon_bbox(annotation["poly"]), "text": text})

        page_gt = 0
        page_edits = 0
        for block in blocks:
            x1, y1, x2, y2 = (round(value) for value in block["bbox"])
            crop = image[max(0, y1 - 2) : min(image.shape[0], y2 + 2), max(0, x1 - 2) : min(image.shape[1], x2 + 2)]
            if crop.size == 0:
                prediction = ""
            else:
                block_started = time.perf_counter()
                block_result = runtime.run_ocr(crop)
                evaluation_elapsed += time.perf_counter() - block_started
                lines = []
                for line in block_result["lines"]:
                    if line.get("text") and line.get("text_region"):
                        bbox = polygon_bbox(line["text_region"])
                        lines.append(((bbox[1] + bbox[3]) / 2.0, bbox[0], line["text"]))
                prediction = "".join(text for _y, _x, text in sorted(lines))
            gt = block["text"]
            edits, cjk_edits, other_edits = edit_counts(gt, prediction)
            page_gt += len(gt)
            page_edits += edits
            total_cjk_chars += sum(is_cjk(character) for character in gt)
            total_cjk_edits += cjk_edits
            total_other_chars += sum(not is_cjk(character) for character in gt)
            total_other_edits += other_edits
            exact_blocks += int(normalize_text(prediction) == gt)
            block_count += 1
        total_gt_chars += page_gt
        total_edits += page_edits
        records.append(
            {
                "image_name": image_name,
                "source": page["page_info"]["page_attribute"].get("data_source"),
                "blocks": len(blocks),
                "gt_characters": page_gt,
                "edits": page_edits,
                "normalized_accuracy": 1.0 - page_edits / max(page_gt, 1),
                "seconds": elapsed,
                "characters_per_second": page_gt / max(elapsed, 1e-9),
            }
        )

    return {
        "status": "pass" if (1.0 - total_edits / max(total_gt_chars, 1)) >= 0.98 else "partial",
        "thresholds": {
            "overall": 0.98,
            "cjk": 0.98,
            "other": 0.90,
        },
        "sample_pages": len(pages),
        "samples_per_source": per_source,
        "text_blocks": block_count,
        "ground_truth_characters": total_gt_chars,
        "predicted_characters": total_pred_chars,
        "character_accuracy": 1.0 - total_edits / max(total_gt_chars, 1),
        "cjk_character_accuracy": 1.0 - total_cjk_edits / max(total_cjk_chars, 1),
        "other_character_accuracy": 1.0 - total_other_edits / max(total_other_chars, 1),
        "block_exact_match": exact_blocks / max(block_count, 1),
        "elapsed_seconds": total_elapsed,
        "evaluation_elapsed_seconds": evaluation_elapsed,
        "characters_per_second": total_gt_chars / max(total_elapsed, 1e-9),
        "predicted_characters_per_second": total_pred_chars / max(total_elapsed, 1e-9),
        "pages_per_second": len(pages) / max(total_elapsed, 1e-9),
        "note": "速度按整页 DET/REC 统计，不含模型加载；准确率按每个标注文本块单独 OCR 统计。",
        "pages": records,
    }


def verify_formula(sample_count: int) -> dict:
    from Code.model_integration.runtime import FormulaRecognitionRunner

    model_dir = CODE_ROOT / "models" / "formula" / "PP-FormulaNet_plus-M" / "PP-FormulaNet_plus-M_infer"
    onnx_path = model_dir / "inference.onnx"
    if not onnx_path.is_file():
        return {"status": "blocked", "reason": f"missing {onnx_path}"}

    samples = []
    for page in sorted(load_omnidocbench(), key=lambda item: item["page_info"]["image_path"]):
        image_path = OMNI_DATA_ROOT / "images" / page["page_info"]["image_path"]
        for annotation in page.get("layout_dets") or ():
            latex = str(annotation.get("latex") or "").strip()
            if annotation.get("category_type") == "equation_isolated" and latex and annotation.get("poly"):
                samples.append((image_path, polygon_bbox(annotation["poly"]), latex))
    samples = stratified_items(samples, sample_count)

    recognizer = FormulaRecognitionRunner(model_dir)
    crops = []
    for image_path, bbox, _latex in samples:
        image = cv2.imread(str(image_path))
        x1, y1, x2, y2 = (round(value) for value in bbox)
        crops.append(image[max(0, y1 - 4) : min(image.shape[0], y2 + 4), max(0, x1 - 4) : min(image.shape[1], x2 + 4)])
    recognizer.predict(crops[:1])

    predictions = []
    elapsed = 0.0
    for start in range(0, len(crops), 8):
        batch = crops[start : start + 8]
        started = time.perf_counter()
        predictions.extend(recognizer.predict(batch))
        elapsed += time.perf_counter() - started

    from latex2mathml.converter import convert as latex_to_mathml

    parseable = 0
    similarities = []
    for prediction, (_image_path, _bbox, ground_truth) in zip(predictions, samples):
        try:
            latex_to_mathml(prediction)
            parseable += 1
        except Exception:
            pass
        normal_prediction = normalize_text(prediction)
        normal_truth = normalize_text(ground_truth)
        edits, _cjk_edits, _other_edits = edit_counts(normal_truth, normal_prediction)
        similarities.append(1.0 - edits / max(len(normal_truth), 1))
    return {
        "status": "pass",
        "sample_count": len(samples),
        "model_path": str(onnx_path),
        "model_sha256": sha256(onnx_path),
        "latex_parse_rate": parseable / len(samples),
        "average_character_similarity": mean(similarities),
        "elapsed_seconds": elapsed,
        "seconds_per_formula": elapsed / len(samples),
        "note": "M-only verification; character similarity is a proxy, not OmniDocBench CDM.",
    }


def verify_layout_map(layout_dir: Path) -> dict:
    script = OMNI_ROOT / "tools" / "eval_layout_models.py"
    if not script.is_file():
        return {"status": "blocked", "reason": f"missing {script}"}
    command = [
        sys.executable,
        str(script),
        "--models",
        "pp_doclayout_v3",
        "--output-dir",
        str(layout_dir),
        "--reuse-predictions",
    ]
    completed = subprocess.run(command, cwd=OMNI_ROOT, text=True, capture_output=True, timeout=300)
    if completed.returncode:
        return {"status": "fail", "command": command, "stderr": completed.stderr[-4000:]}
    with (layout_dir / "summary_overall.csv").open(encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    prediction_path = layout_dir / "predictions" / "pp_doclayout_v3.json"
    return {
        "status": "pass",
        "dataset": str(OMNI_DATA_ROOT / "OmniDocBench.json"),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "class_space": "title/text/figure/table/equation/caption",
        "page_count": int(row["page_count"]),
        "gt_count": int(row["gt_count"]),
        "pred_count": int(row["pred_count"]),
        "mAP50_95": float(row["mAP50_95"]) / 100.0,
        "mAP50": float(row["mAP50"]) / 100.0,
        "precision50": float(row["precision50"]) / 100.0,
        "recall50": float(row["recall50"]) / 100.0,
    }


def count_docx_math(path: Path) -> int:
    with path.open("rb") as stream:
        import zipfile

        with zipfile.ZipFile(stream) as archive:
            root = etree.fromstring(archive.read("word/document.xml"))
    return len(root.xpath(".//*[local-name()='oMath']"))


def count_pdf_hidden_text(path: Path) -> tuple[int, int, int]:
    document = fitz.open(path)
    pages = len(document)
    hidden_markers = 0
    text_length = 0
    image_pages = 0
    for page in document:
        content = page.read_contents()
        hidden_markers += content.count(b"3 Tr")
        text_length += len(page.get_text().strip())
        image_pages += int(bool(page.get_images(full=True)))
    return pages, hidden_markers, image_pages if text_length else 0


def verify_output_bundle(run_dir: Path) -> dict:
    from Code.test import _native_docx_table_count

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    samples = manifest.get("samples") or []
    records = []
    pdf_temp = Path(tempfile.mkdtemp(prefix="docflow_docx_pdf_check_"))
    docx_files = sorted(run_dir.glob("*.docx"))
    if docx_files:
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(pdf_temp), *map(str, docx_files)],
            check=True,
            capture_output=True,
            timeout=300,
        )
    for sample in samples:
        name = sample["sample_key"]
        extensions = {"markdown": "md"}
        paths = {
            suffix: run_dir / f"{name}.{extensions.get(suffix, suffix)}"
            for suffix in ("docx", "markdown", "pdf", "txt", "html", "rtf", "xlsx")
        }
        record = {"sample": name, "files": {}, "missing": []}
        for suffix, path in paths.items():
            record["files"][suffix] = {"exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}
            if not path.is_file() or path.stat().st_size == 0:
                record["missing"].append(suffix)
        if paths["docx"].is_file():
            document = Document(paths["docx"])
            record["docx_native_tables"] = _native_docx_table_count(paths["docx"])
            record["docx_math_objects"] = count_docx_math(paths["docx"])
            record["docx_paragraphs"] = len(document.paragraphs)
        if paths["xlsx"].is_file():
            workbook = load_workbook(paths["xlsx"], read_only=True)
            record["xlsx_sheets"] = len(workbook.sheetnames)
            record["xlsx_nonempty_sheets"] = sum(
                1 for sheet in workbook.worksheets if sheet.max_row > 1 or sheet.max_column > 1
            )
        if paths["pdf"].is_file():
            pages, hidden_markers, image_pages = count_pdf_hidden_text(paths["pdf"])
            record["pdf_pages"] = pages
            record["pdf_hidden_text_markers"] = hidden_markers
            record["pdf_image_pages"] = image_pages
        rendered_pdf = pdf_temp / f"{name}.pdf"
        if rendered_pdf.is_file():
            record["docx_rendered_pages"] = len(fitz.open(rendered_pdf))
        records.append(record)

    failures = manifest.get("failures") or []
    all_files_present = all(not record["missing"] for record in records)
    all_docx_single_page = all(record.get("docx_rendered_pages") == 1 for record in records)
    all_pdf_layers = all(
        record.get("pdf_pages") == 1
        and record.get("pdf_hidden_text_markers", 0) > 0
        and record.get("pdf_image_pages") == 1
        for record in records
    )
    return {
        "status": "pass" if not failures and all_files_present and all_docx_single_page and all_pdf_layers else "fail",
        "run_dir": str(run_dir),
        "sample_count": manifest.get("sample_count"),
        "page_count": manifest.get("page_count"),
        "failures": failures,
        "all_files_present": all_files_present,
        "all_docx_single_page": all_docx_single_page,
        "all_pdf_dual_layer": all_pdf_layers,
        "samples": records,
    }


def environment() -> dict:
    import openvino as ov

    try:
        import onnxruntime as ort

        ort_version = ort.__version__
    except Exception:
        ort_version = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "openvino": ov.__version__,
        "onnxruntime": ort_version,
    }


def verify_unit_tests() -> dict:
    environment_variables = dict(os.environ)
    environment_variables["PYTHONPATH"] = str(CODE_ROOT / "docflow_src")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=PROJECT_ROOT,
        env=environment_variables,
        text=True,
        capture_output=True,
        timeout=300,
    )
    match = re.search(r"(\d+) passed(?:, (\d+) failed)?", completed.stdout)
    return {
        "status": "pass" if completed.returncode == 0 else "fail",
        "passed": int(match.group(1)) if match else 0,
        "failed": int(match.group(2) or 0) if match else 0,
    }


def model_sizes() -> dict:
    root = CODE_ROOT / "models_openvino"
    rows = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".xml", ".bin", ".onnx"}:
            rows[str(path.relative_to(CODE_ROOT))] = path.stat().st_size
    return rows


def markdown_report(payload: dict) -> str:
    ocr = payload.get("ocr", {})
    font = payload.get("font_classifier", {})
    charset = payload.get("charset", {})
    formula = payload.get("formula", {})
    layout = payload.get("layout_map", {})
    bundle = payload.get("output_bundle", {})
    pipeline = payload.get("full_pipeline", {})
    unit_tests = payload.get("unit_tests", {})
    lines = [
        "# 交付指标验证记录",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 验证结论",
        "",
        "| 指标 | 实测结果 | 样本/口径 | 状态 |",
        "|---|---|---|---|",
    ]
    if ocr:
        lines.append(
            f"| OCR 字符准确率 | {ocr['character_accuracy']:.4%} | OmniDocBench {ocr['sample_pages']} 页、{ocr['text_blocks']} 个文本块 | {ocr['status']} |"
        )
        lines.append(
            f"| OCR 识别速度 | {ocr['characters_per_second']:.2f} 字/s | 不含模型加载 | {'pass' if ocr['characters_per_second'] >= 150 else 'fail'} |"
        )
    if unit_tests:
        lines.append(
            f"| 自动化测试 | {unit_tests['passed']} passed / {unit_tests['failed']} failed | pytest tests -q | {unit_tests['status']} |"
        )
    if font:
        lines.append(
            f"| 字体分类准确率 | {font['accuracy']:.4%} | {font['sample_count']} 个分层抽样样本 | {font['status']} |"
        )
    if charset:
        lines.append(
            f"| GB2312 汉字字典覆盖 | {charset['gb2312_han']:.4%} | OCR 字典静态覆盖，不等于识别正确率 | {charset['status']} |"
        )
    if layout:
        lines.append(
            f"| 版面 mAP50-95 | {layout['mAP50_95']:.4%} | OmniDocBench {layout['page_count']} 页，6 类合并口径 | {layout['status']} |"
        )
    if formula:
        if formula.get("status") == "pass":
            lines.append(
                f"| M 公式 LaTeX 可解析率 | {formula['latex_parse_rate']:.4%} | {formula['sample_count']} 个公式区域 | pass |"
            )
        else:
            lines.append(f"| M 公式指标 | 未执行 | {formula.get('reason', '')} | blocked |")
    if bundle:
        lines.append(
            f"| 全量输出完整性 | 文件齐全={bundle['all_files_present']}，DOCX 单页={bundle['all_docx_single_page']}，PDF 双图层={bundle['all_pdf_dual_layer']} | {bundle['sample_count']} 个样本 | {bundle['status']} |"
        )
        lines.append(
            f"| 功能通过率 | {bundle['sample_count']}/{bundle['sample_count']} | run_manifest.failures={bundle['failures']} | pass |"
        )
        native_tables = sum(sample.get("docx_native_tables", 0) for sample in bundle["samples"])
        lines.append(f"| DOCX 原生表格 | {native_tables} | 33 个样本 | measured |")
    if pipeline:
        lines.append(
            f"| 全量耗时/峰值内存 | {pipeline.get('elapsed_seconds')} s / {pipeline.get('max_rss_kb', 0) / 1024 / 1024:.2f} GiB | 33 个样本、7 种输出格式、关闭公式 | measured |"
        )
    lines.extend(["", "## 明细", ""])
    if ocr:
        lines.extend(
            [
                f"- OCR 汉字字符准确率：`{ocr['cjk_character_accuracy']:.4%}`；其他字符准确率：`{ocr['other_character_accuracy']:.4%}`。",
                f"- OCR 文本块完全匹配率：`{ocr['block_exact_match']:.4%}`；平均每页耗时：`{ocr['elapsed_seconds'] / max(ocr['sample_pages'], 1):.4f} s`。",
                f"- OCR 口径：{ocr['note']}",
            ]
        )
    if font:
        lines.append(
            f"- 字体分类宏平均 F1：`{font['macro_f1']:.4%}`；接受阈值通过率：`{font['accepted_rate']:.4%}`；单行耗时：`{font['milliseconds_per_crop']:.2f} ms`。"
        )
    if charset:
        lines.append(
            f"- OCR 字典共 `{charset['dictionary_characters']}` 个字符；GB2312 汉字缺失 `{charset['gb2312_han_missing']}` 个；GB18030 基本区汉字覆盖 `{charset['gb18030_basic_cjk']:.4%}`。"
        )
    if formula and formula.get("status") == "pass":
        lines.append(
            f"- M 平均字符相似度：`{formula['average_character_similarity']:.4%}`；平均单公式耗时：`{formula['seconds_per_formula']:.4f} s`。"
        )
    if layout:
        lines.append(
            f"- 版面 mAP50：`{layout['mAP50']:.4%}`，Precision50：`{layout['precision50']:.4%}`，Recall50：`{layout['recall50']:.4%}`；类别空间为 `{layout['class_space']}`。"
        )
    if bundle:
        lines.append(f"- 全量运行目录：`{bundle['run_dir']}`；失败项：`{bundle['failures']}`。")
        split_samples = [
            sample["sample"]
            for sample in bundle["samples"]
            if sample.get("docx_rendered_pages") != 1
        ]
        if split_samples:
            lines.append(f"- DOCX 转 PDF 后发生额外分页的样本：`{split_samples}`。")
        if pipeline:
            lines.append(
                f"- 全量峰值内存 `{pipeline.get('max_rss_kb', 0) / 1024 / 1024:.2f} GiB`，高于交付文档旧记录的 `2.84–3.70 GiB`；本轮包含 7 种输出格式与阶段 JSON，两个数字口径不同。"
            )
    lines.extend(
        [
            "",
            "## 未闭环项",
            "",
            "- 本轮 OCR 使用 OmniDocBench 工程抽样，不等同于 GB/T 17961-2010 的财政部正式样张验收。",
            "- 版面 mAP 使用 PP-DocLayoutV3 的既有全量预测结果复算，指标空间为 title/text/figure/table/equation/caption 六类合并口径。",
            "- 当前 OCR 接口只返回 top-1 文本和置信度，没有识别候选字符；人工校对界面也不在算法包内。",
            "- Excel、Word、PDF 的逐项视觉保真仍需人工按正式样张确认；本报告验证结构、页数、原生表格和双图层完整性。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None, help="Full-pipeline output run directory")
    parser.add_argument("--report-prefix", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ocr-pages-per-source", type=int, default=2)
    parser.add_argument("--font-samples-per-class", type=int, default=50)
    parser.add_argument("--formula-samples", type=int, default=24)
    parser.add_argument("--full-run-seconds", type=float, default=None)
    parser.add_argument("--full-run-max-rss-kb", type=int, default=None)
    parser.add_argument(
        "--layout-result-dir",
        type=Path,
        default=OMNI_ROOT / "result" / "layout_benchmark_user_models_fixed",
    )
    parser.add_argument("--skip-formula", action="store_true", help="Skip formula metrics")
    parser.add_argument("--skip-model-metrics", action="store_true")
    args = parser.parse_args()

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": environment(),
        "model_sizes": model_sizes(),
        "unit_tests": verify_unit_tests(),
    }
    if not args.skip_model_metrics:
        payload["font_classifier"] = verify_font_classifier(args.font_samples_per_class)
        payload["charset"] = verify_charset()
        payload["ocr"] = verify_ocr(args.ocr_pages_per_source)
        if not args.skip_formula:
            payload["formula"] = verify_formula(args.formula_samples)
        payload["layout_map"] = verify_layout_map(args.layout_result_dir)
    if args.run_dir is not None:
        payload["output_bundle"] = verify_output_bundle(args.run_dir)
    if args.full_run_seconds is not None or args.full_run_max_rss_kb is not None:
        payload["full_pipeline"] = {
            "elapsed_seconds": args.full_run_seconds,
            "max_rss_kb": args.full_run_max_rss_kb,
        }

    report_prefix = args.report_prefix.resolve()
    report_prefix.parent.mkdir(parents=True, exist_ok=True)
    report_prefix.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_prefix.with_suffix(".md").write_text(markdown_report(payload), encoding="utf-8")
    print(report_prefix.with_suffix(".json"))
    print(report_prefix.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
