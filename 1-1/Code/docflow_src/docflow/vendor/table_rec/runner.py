"""项目使用的最小 RapidAI 表格识别流程。"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def _rect(poly_box: Any) -> list[float]:
    points = np.asarray(poly_box, dtype=np.float32).reshape(-1, 2)
    return [
        float(np.min(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 0])),
        float(np.max(points[:, 1])),
    ]


def _normalize_bboxes(boxes: Any) -> list[list[float]]:
    if boxes is None:
        return []
    values = np.asarray(boxes)
    if values.size == 0:
        return []
    result = []
    for box in values.reshape((values.shape[0], -1)):
        if len(box) == 4:
            x0, y0, x1, y1 = map(float, box)
            result.append([x0, y0, x1, y0, x1, y1, x0, y1])
        elif len(box) >= 8:
            result.append([float(value) for value in box[:8]])
    return result


def _normalize_ocr(ocr_result: list) -> list[dict]:
    result = []
    for item in ocr_result or []:
        if len(item) < 3 or not str(item[1]).strip():
            continue
        box, text, score = item[:3]
        result.append(
            {
                "text": str(text),
                "score": float(score),
                "poly": np.asarray(box, dtype=np.float32).reshape(-1, 2).tolist(),
                "rect": _rect(box),
            }
        )
    return result


def _table_output(output: Any) -> tuple[str, float | None, Any, Any, Any]:
    if isinstance(output, tuple):
        values = list(output) + [None] * 5
        return str(values[0] or ""), values[1], values[2], values[3], values[4]
    return (
        str(getattr(output, "pred_html", "") or ""),
        getattr(output, "elapse", None),
        getattr(output, "cell_bboxes", None),
        getattr(output, "logic_points", None),
        None,
    )


def make_ocr_engine(_model_dir, det_model, rec_model, rec_keys, cls_model):
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(
        det_model_path=str(det_model),
        rec_model_path=str(rec_model),
        rec_keys_path=str(rec_keys),
        cls_model_path=str(cls_model),
    )


class RapidAITableRecognizer:
    def predict(
        self,
        image: np.ndarray,
        small_box_cut_enhance: bool = True,
        char_ocr: bool = False,
        rotated_fix: bool = False,
        col_threshold: int = 15,
        row_threshold: int = 10,
    ) -> dict:
        started = time.perf_counter()
        class_started = time.perf_counter()
        table_class, class_elapsed = self.table_cls(image)
        class_elapsed = class_elapsed if class_elapsed is not None else time.perf_counter() - class_started
        table_type = "wired_table_v2" if table_class == "wired" else "lineless_table"
        if self.table_engine_type != "auto":
            table_type = self.table_engine_type
        if table_type not in {"wired_table_v2", "lineless_table"}:
            raise ValueError(f"unsupported table_engine_type: {self.table_engine_type}")

        ocr_started = time.perf_counter()
        ocr_result, ocr_elapsed = self.ocr_engine(image, return_word_box=char_ocr)
        ocr_elapsed_ms = (time.perf_counter() - ocr_started) * 1000
        if ocr_elapsed:
            ocr_elapsed_ms = float(sum(ocr_elapsed)) * 1000

        engine = self.wired_engine if table_type == "wired_table_v2" else self.lineless_engine
        output = engine(
            image,
            ocr_result=ocr_result or [],
            enhance_box_line=small_box_cut_enhance,
            rotated_fix=rotated_fix,
            col_threshold=col_threshold,
            row_threshold=row_threshold,
        )
        html, model_elapsed, boxes, logic_points, matched_ocr = _table_output(output)
        result = {
            "status": "ok" if html else "error",
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "table_elapsed_ms": float(model_elapsed) * 1000 if model_elapsed is not None else None,
            "table_type": table_type,
            "table_cls_result": table_class,
            "table_cls_elapsed_ms": float(class_elapsed) * 1000,
            "wired_table_v2_mode": self.wired_mode,
            "lineless_mode": self.lineless_mode,
            "ocr_model_dir": str(self.ocr_model_dir),
            "model_paths": self.model_paths,
            "ocr_elapsed_ms": ocr_elapsed_ms,
            "ocr_boxes": len(ocr_result or []),
            "bbox": _normalize_bboxes(boxes),
            "logic_points": np.asarray(logic_points).tolist() if logic_points is not None else [],
            "ocr_result": _normalize_ocr(ocr_result or []),
            "html": html,
        }
        if matched_ocr is not None:
            result["matched_ocr_count"] = len(matched_ocr)
        return result
