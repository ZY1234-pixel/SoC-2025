"""Inject RapidAI table recognition and layout fusion into Paddle regions."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from docflow.inference import OpenVINOInferSession
from docflow.vendor.table_rec import runner
from docflow.vendor.table_rec.fusion import (
    LayoutTableFusion,
    crop_layout_images,
    fused_to_html,
    image_to_data_uri,
)
from docflow.vendor.table_rec.slanet_plus import SLANetPlusTableEngine


LOGGER = logging.getLogger(__name__)
_FORMULA_LABELS = {"display_formula", "equation", "formula", "inline_formula"}


class RapidAITableAdapter:
    def __init__(
        self,
        models_openvino_root: Path,
    ):
        openvino_root = Path(models_openvino_root)
        table_model_dir = openvino_root / "RapidAI_TableRec_openvino"
        det_model = openvino_root / "PP-OCRv6_small_det_openvino" / "PP-OCRv6_small_det_openvino_fp32.xml"
        rec_model = openvino_root / "PP-OCRv6_small_rec_openvino" / "PP-OCRv6_small_rec_openvino_fp32.xml"
        rec_keys = openvino_root / "PP-OCRv6_small_rec_openvino" / "ppocrv6_rapidocr_dict.txt"
        cls_model = table_model_dir / "ocr_cls" / "ch_ppocr_mobile_v2.0_cls_infer.xml"
        required = (
            table_model_dir / "wired_table_v2" / "unet.xml",
            table_model_dir / "slanet_plus" / "slanet-plus.xml",
            table_model_dir / "slanet_plus" / "character.txt",
            table_model_dir / "table_cls" / "yolo_cls.xml",
            det_model,
            rec_model,
            rec_keys,
            cls_model,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"RapidAI model files not found: {missing}")
        self.recognizer = self._openvino_recognizer(table_model_dir, det_model, rec_model, rec_keys, cls_model)
        self.fusion = LayoutTableFusion()
        self.crop_layout_images = crop_layout_images
        self.fused_to_html = fused_to_html
        self.image_to_data_uri = image_to_data_uri

    @staticmethod
    def _openvino_recognizer(table_dir, det_model, rec_model, rec_keys, cls_model):
        import rapidocr_onnxruntime.ch_ppocr_cls.text_cls as ocr_cls_runtime
        import rapidocr_onnxruntime.ch_ppocr_det.text_detect as ocr_det_runtime
        import rapidocr_onnxruntime.ch_ppocr_rec.text_recognize as ocr_rec_runtime
        import table_cls.main as table_cls_runtime
        import wired_table_rec.table_structure_unet as unet_runtime
        from table_cls import TableCls
        from wired_table_rec import WiredTableRecognition
        from wired_table_rec.main import ModelType, WiredTableInput

        unet_runtime.OrtInferSession = OpenVINOInferSession
        table_cls_runtime.OrtInferSession = OpenVINOInferSession
        ocr_det_runtime.OrtInferSession = OpenVINOInferSession
        ocr_cls_runtime.OrtInferSession = OpenVINOInferSession
        ocr_rec_runtime.OrtInferSession = lambda config: OpenVINOInferSession(config, "bf16")

        recognizer = runner.RapidAITableRecognizer.__new__(runner.RapidAITableRecognizer)
        recognizer.table_engine_type = "auto"
        recognizer.wired_engine = WiredTableRecognition(
            WiredTableInput(model_type=ModelType.UNET.value, model_path=table_dir / "wired_table_v2" / "unet.xml")
        )
        recognizer.wired_mode = "openvino_unet"
        recognizer.lineless_engine = SLANetPlusTableEngine(
            table_dir / "slanet_plus" / "slanet-plus.xml"
        )
        recognizer.lineless_mode = "openvino_slanet_plus"
        recognizer.table_cls = TableCls(model_path=table_dir / "table_cls" / "yolo_cls.xml")
        recognizer.ocr_engine = runner.make_ocr_engine(
            None, det_model, rec_model, rec_keys, cls_model
        )
        recognizer.ocr_model_dir = rec_model.parent
        recognizer.model_paths = {
            "table": str(table_dir),
            "slanet_plus": str(table_dir / "slanet_plus" / "slanet-plus.xml"),
            "det": str(det_model),
            "rec": str(rec_model),
            "rec_keys": str(rec_keys),
            "ocr_cls": str(cls_model),
        }
        return recognizer

    def enrich(self, image: np.ndarray, regions: list[dict], page_index: int, output_dir: Path) -> list[dict]:
        height, width = image.shape[:2]
        candidates = [
            (index, self._expand_bbox(region.get("bbox"), width, height))
            for index, region in enumerate(regions)
            if str(region.get("raw_type") or region.get("type") or "").lower() == "table"
        ]
        candidates = [(index, bbox) for index, bbox in candidates if bbox]
        if not candidates:
            return regions

        page_dir = Path(output_dir) / "rapidai" / f"page_{page_index + 1:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        replacements: list[dict] = []
        replaced_indexes = set()
        for table_index, bbox in candidates:
            try:
                x0, y0, x1, y1 = bbox
                crop = image[y0:y1, x0:x1]
                table_dir = page_dir / f"table_{len(replacements) + 1:03d}"
                table_dir.mkdir(parents=True, exist_ok=True)
                crop_path = table_dir / "source.png"
                if not cv2.imwrite(str(crop_path), crop):
                    raise OSError(f"cannot write table crop: {crop_path}")
                table_result = self.recognizer.predict(crop)
                self._apply_table_rule_geometry(table_result, crop)
                crop_layout = {"boxes": self.crop_layout_regions(regions, bbox)}
                fused = self.fusion.fuse(crop_path, crop_layout, table_result)
                if self._needs_engine_fallback(fused):
                    selected_engine = str(table_result.get("table_type") or "")
                    alternate_engine = (
                        "slanet_plus" if selected_engine == "wired_table_v2" else "wired_table_v2"
                    )
                    previous_engine = self.recognizer.table_engine_type
                    try:
                        self.recognizer.table_engine_type = alternate_engine
                        alternate_result = self.recognizer.predict(crop)
                        alternate_fused = self.fusion.fuse(crop_path, crop_layout, alternate_result)
                    except Exception as exc:
                        LOGGER.warning("RapidAI alternate engine %s failed: %s", alternate_engine, exc)
                    else:
                        if self._fusion_quality(alternate_fused) < self._fusion_quality(fused):
                            table_result, fused = alternate_result, alternate_fused
                    finally:
                        self.recognizer.table_engine_type = previous_engine
                if fused.get("status") != "ok" or not fused.get("cells"):
                    continue

                assets_dir = table_dir / "assets"
                self.crop_layout_images(fused, crop_path, assets_dir)
                self._crop_formula_images(fused, crop, assets_dir)
                debug_path = table_dir / "fused.json"
                debug_path.write_text(json.dumps(fused, ensure_ascii=False, indent=2), encoding="utf-8")
                html_fused = copy.deepcopy(fused)
                source_cells = fused.get("cells") or ()
                for cell_index, cell in enumerate(html_fused.get("cells") or []):
                    bbox = cell.get("bbox")
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        cell["page_bbox"] = [
                            float(bbox[0]) + x0,
                            float(bbox[1]) + y0,
                            float(bbox[2]) + x0,
                            float(bbox[3]) + y0,
                        ]
                    for obj in cell.get("layout_objects") or []:
                        bbox = obj.get("bbox")
                        if obj.get("latex") and bbox:
                            obj["page_bbox"] = [
                                float(bbox[0]) + x0,
                                float(bbox[1]) + y0,
                                float(bbox[2]) + x0,
                                float(bbox[3]) + y0,
                            ]
                    formula_boxes = [
                        obj.get("bbox")
                        for obj in cell.get("layout_objects") or []
                        if obj.get("label") in _FORMULA_LABELS and obj.get("image_path")
                    ]
                    if formula_boxes:
                        cleaned_text = " ".join(
                            str(obj.get("text") or "").strip()
                            for obj in cell.get("ocr_objects") or []
                            if str(obj.get("text") or "").strip()
                            and not any(self._formula_overlaps_ocr(box, obj.get("bbox")) for box in formula_boxes)
                        )
                        cell["text"] = cleaned_text
                        if cell_index < len(source_cells):
                            source_cells[cell_index]["text"] = cleaned_text
                    for obj in cell.get("layout_objects") or []:
                        asset = obj.get("image_path")
                        if asset:
                            obj["image_path"] = self.image_to_data_uri(assets_dir.parent / asset)
                local = self._clip_bbox(fused.get("table_bbox"), crop.shape[1], crop.shape[0]) or [0, 0, crop.shape[1], crop.shape[0]]
                page_bbox = [local[0] + x0, local[1] + y0, local[2] + x0, local[3] + y0]
                table_cells = self._translate_cells(fused["cells"], -local[0], -local[1])
                source = regions[table_index] if table_index is not None else {}
                px0, py0, px1, py1 = page_bbox
                replacements.append(
                    {
                        "type": "table",
                        "raw_type": "table",
                        "bbox": page_bbox,
                        "img": image[py0:py1, px0:px1],
                        "res": {"html": self.fused_to_html(html_fused), "cells": table_cells},
                        "img_idx": page_index,
                        "score": float(source.get("score", 1.0)),
                        "model_order": source.get("model_order") if source.get("model_order") is not None else 0,
                        "layout_model": "rapidai-table-fusion",
                        "attributes": {
                            "rapidai": fused.get("diagnostics") or {},
                            "debug_path": str(debug_path),
                            "table_content_fit": True,
                            "table_auto_height": table_result.get("table_type") == "slanet_plus",
                        },
                    }
                )
                if table_index is not None:
                    replaced_indexes.add(table_index)
            except Exception as exc:
                LOGGER.warning("RapidAI table recognition failed for page %s bbox %s: %s", page_index + 1, bbox, exc)
        return [region for index, region in enumerate(regions) if index not in replaced_indexes] + replacements

    @classmethod
    def _apply_table_rule_geometry(cls, table_result: dict, image: np.ndarray) -> None:
        """Constrain table OCR to the physical horizontal-rule envelope.

        A table crop can include a continuation label above the top rule or a
        source note below the bottom rule. Those lines are page furniture, not
        table content, and must not be allowed to compete with real cells.
        """
        rules = cls._detect_horizontal_rules(image)
        if len(rules) < 2:
            return
        _, width = image.shape[:2]
        content_bbox = [0, int(rules[0]) + 1, width, int(rules[-1])]
        table_result["_content_bbox"] = content_bbox
        table_result["_horizontal_rule_ys"] = [float(value) for value in rules]
        table_result["ocr_result"] = [
            item
            for item in table_result.get("ocr_result") or []
            if cls._bbox_coverage(content_bbox, item.get("rect") or item.get("bbox")) >= 0.60
        ]

    @staticmethod
    def _detect_horizontal_rules(image: np.ndarray) -> list[int]:
        if image.ndim != 3 or image.size == 0:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel_width = max(40, int(width * 0.45))
        horizontal = cv2.morphologyEx(
            ink,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)),
        )
        counts = (horizontal > 0).sum(axis=1)
        active = np.flatnonzero(counts >= width * 0.55)
        groups: list[list[int]] = []
        for value in active:
            if groups and value - groups[-1][-1] <= 3:
                groups[-1].append(int(value))
            else:
                groups.append([int(value)])
        return [int(np.mean(group)) for group in groups]

    @staticmethod
    def _needs_engine_fallback(fused: dict) -> bool:
        row_count = int(fused.get("row_count") or 0)
        col_count = int(fused.get("col_count") or 0)
        cells = fused.get("cells") or ()
        diagnostics = fused.get("diagnostics") or {}
        if not cells or row_count <= 0 or col_count <= 0:
            return True
        collisions = RapidAITableAdapter._span_collisions(cells)
        empty_ratio = int(diagnostics.get("empty_cells") or 0) / max(len(cells), 1)
        collision_ratio = collisions / max(row_count * col_count, 1)
        semantic_defects = RapidAITableAdapter._semantic_defect_count(
            cells,
            row_count,
            col_count,
        )
        return collision_ratio > 0.01 or empty_ratio > 0.15 or semantic_defects > 0

    @staticmethod
    def _fusion_quality(fused: dict) -> float:
        row_count = max(int(fused.get("row_count") or 0), 1)
        col_count = max(int(fused.get("col_count") or 0), 1)
        cells = fused.get("cells") or ()
        diagnostics = fused.get("diagnostics") or {}
        collisions = RapidAITableAdapter._span_collisions(cells)
        empty_ratio = int(diagnostics.get("empty_cells") or 0) / max(len(cells), 1)
        text_units = sum(len(str(cell.get("text") or "")) for cell in cells)
        visual_count = sum(len(cell.get("layout_objects") or ()) for cell in cells)
        content_count = max(len(fused.get("ocr_objects") or ()), len(cells), 1)
        cell_coverage = min(len(cells) / content_count, 1.0)
        semantic_defects = RapidAITableAdapter._semantic_defect_count(
            cells,
            row_count,
            col_count,
        )
        return (
            collisions / max(row_count * col_count, 1) * 2.0
            + empty_ratio
            + semantic_defects * 0.20
            - min(text_units / 10000.0, 0.20)
            - min(visual_count / 100.0, 0.10)
            - cell_coverage * 0.50
        )

    @staticmethod
    def _semantic_defect_count(cells, row_count: int, col_count: int) -> int:
        def empty(cell) -> bool:
            return not str(cell.get("text") or "").strip() and not cell.get("layout_objects")

        occupied: dict[tuple[int, int], list] = {}
        for cell in cells:
            row = int(cell.get("row", 0))
            column = int(cell.get("col", 0))
            row_span = max(int(cell.get("rowspan", 1)), 1)
            column_span = max(int(cell.get("colspan", 1)), 1)
            for target_row in range(row, row + row_span):
                for target_column in range(column, column + column_span):
                    occupied.setdefault((target_row, target_column), []).append(cell)

        defects = 0
        for row in (0, row_count - 1):
            row_cells = [
                cell
                for column in range(col_count)
                for cell in occupied.get((row, column), ())
            ]
            if row_cells and all(empty(cell) for cell in row_cells):
                defects += 1

        row_headers: dict[int, list] = {}
        for cell in cells:
            if str(cell.get("role") or "") == "row_header":
                row_headers.setdefault(int(cell.get("col", 0)), []).append(cell)
        for column_cells in row_headers.values():
            column_cells.sort(key=lambda cell: int(cell.get("row", 0)))
            for left, right in zip(column_cells, column_cells[1:]):
                if (
                    str(left.get("text") or "").strip()
                    and empty(right)
                    and int(left.get("row", 0)) + max(int(left.get("rowspan", 1)), 1)
                    == int(right.get("row", 0))
                ):
                    defects += 1

        top_row = min((int(cell.get("row", 0)) for cell in cells), default=0)
        for row in (top_row, top_row + 1):
            row_cells = sorted(
                (cell for cell in cells if int(cell.get("row", 0)) == row),
                key=lambda cell: int(cell.get("col", 0)),
            )
            for left, right in zip(row_cells, row_cells[1:]):
                if (
                    int(left.get("colspan", 1)) == int(right.get("colspan", 1)) == 1
                    and int(left.get("rowspan", 1)) == int(right.get("rowspan", 1)) >= 2
                    and int(right.get("col", 0)) == int(left.get("col", 0)) + 1
                    and len(str(left.get("text") or "").strip()) == len(str(right.get("text") or "").strip()) == 1
                ):
                    defects += 1
        return defects

    @staticmethod
    def _span_collisions(cells) -> int:
        occupied: dict[tuple[int, int], int] = {}
        for cell in cells:
            row = int(cell.get("row", 0))
            column = int(cell.get("col", 0))
            row_span = max(int(cell.get("rowspan", 1)), 1)
            column_span = max(int(cell.get("colspan", 1)), 1)
            for target_row in range(row, row + row_span):
                for target_column in range(column, column + column_span):
                    key = (target_row, target_column)
                    occupied[key] = occupied.get(key, 0) + 1
        return sum(count > 1 for count in occupied.values())

    @staticmethod
    def _crop_formula_images(fused: dict, image: np.ndarray, assets_dir: Path) -> None:
        height, width = image.shape[:2]
        assets_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for cell in fused.get("cells") or []:
            for obj in cell.get("layout_objects") or []:
                if obj.get("label") not in _FORMULA_LABELS or obj.get("image_path"):
                    continue
                bbox = RapidAITableAdapter._clip_bbox(obj.get("bbox"), width, height)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                path = assets_dir / f"formula_{count:03d}.png"
                if cv2.imwrite(str(path), image[y1:y2, x1:x2]):
                    obj["image_path"] = f"assets/{path.name}"
                    obj["visual_kind"] = "semantic_visual"
                    obj["render_decision"] = "render"
                    count += 1

    @staticmethod
    def _bbox_coverage(outer, inner) -> float:
        if not outer or not inner or len(outer) < 4 or len(inner) < 4:
            return 0.0
        x1, y1 = max(float(outer[0]), float(inner[0])), max(float(outer[1]), float(inner[1]))
        x2, y2 = min(float(outer[2]), float(inner[2])), min(float(outer[3]), float(inner[3]))
        overlap = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        area = max(float(inner[2]) - float(inner[0]), 0.0) * max(float(inner[3]) - float(inner[1]), 0.0)
        return overlap / max(area, 1.0)

    @staticmethod
    def _formula_overlaps_ocr(formula_bbox, ocr_bbox) -> bool:
        if not formula_bbox or not ocr_bbox or len(formula_bbox) < 4 or len(ocr_bbox) < 4:
            return False
        if RapidAITableAdapter._bbox_coverage(formula_bbox, ocr_bbox) >= 0.35:
            return True
        center_x = (float(ocr_bbox[0]) + float(ocr_bbox[2])) / 2.0
        center_y = (float(ocr_bbox[1]) + float(ocr_bbox[3])) / 2.0
        return (
            float(formula_bbox[0]) <= center_x <= float(formula_bbox[2])
            and float(formula_bbox[1]) <= center_y <= float(formula_bbox[3])
        )

    @staticmethod
    def _translate_cells(cells, offset_x: float, offset_y: float):
        result = copy.deepcopy(cells)
        for cell in result:
            for obj in [cell, *(cell.get("layout_objects") or ()), *(cell.get("ocr_objects") or ())]:
                bbox = obj.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    obj["bbox"] = [
                        float(bbox[0]) + offset_x,
                        float(bbox[1]) + offset_y,
                        float(bbox[2]) + offset_x,
                        float(bbox[3]) + offset_y,
                    ]
        return result

    @staticmethod
    def crop_layout_regions(regions: list[dict], crop_bbox: list[int]) -> list[dict]:
        x0, y0, x1, y1 = crop_bbox
        result = []
        for region in regions:
            # OCR recall regions remain text evidence, but are not layout evidence.
            if not region.get("layout_model"):
                continue
            bbox = region.get("bbox") or ()
            if len(bbox) < 4:
                continue
            rx0, ry0, rx1, ry1 = map(float, bbox[:4])
            ix0, iy0, ix1, iy1 = max(x0, rx0), max(y0, ry0), min(x1, rx1), min(y1, ry1)
            overlap = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            if overlap <= 0 or overlap / max(1.0, (rx1 - rx0) * (ry1 - ry0)) < 0.25:
                continue
            result.append(
                {
                    "label": str(region.get("raw_type") or region.get("type") or ""),
                    "bbox": [ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0],
                    "score": float(region.get("score", 1.0)),
                    "res": region.get("res") if isinstance(region.get("res"), dict) else {},
                }
            )
        return result

    @staticmethod
    def _clip_bbox(value, width: int, height: int) -> list[int] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return None
        x0, y0, x1, y1 = map(float, value[:4])
        bbox = [max(0, int(x0)), max(0, int(y0)), min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))]
        return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None

    @classmethod
    def _expand_bbox(cls, value, width: int, height: int) -> list[int] | None:
        bbox = cls._clip_bbox(value, width, height)
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        pad_x = max(4, round((x1 - x0) * 0.04))
        pad_y = max(4, round((y1 - y0) * 0.04))
        return [max(0, x0 - pad_x), max(0, y0 - pad_y), min(width, x1 + pad_x), min(height, y1 + pad_y)]
