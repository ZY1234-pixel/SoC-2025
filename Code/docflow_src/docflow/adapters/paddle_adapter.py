"""Capture PaddleOCR output as immutable recognition evidence."""

from __future__ import annotations

import base64
from typing import Any, Optional

import cv2
import numpy as np

from docflow.model.stages import RecognitionEvidence, RecognitionItem, RecognitionPage, Rect, TextEvidence


class PaddleAdapter:
    """Translate engine-specific fields without deleting or repairing regions."""

    _CATEGORY_MAP = {
        "abstract": "abstract",
        "algorithm": "code",
        "aside_text": "text",
        "chart": "figure",
        "content": "text",
        "display_formula": "formula",
        "doc_title": "title",
        "equation": "formula",
        "figure": "figure",
        "figure_caption": "figure_caption",
        "figure_title": "figure_caption",
        "footer": "footer",
        "footer_image": "figure",
        "footnote": "footnote",
        "formula_number": "formula",
        "header": "header",
        "header_image": "figure",
        "image": "figure",
        "inline_formula": "formula",
        "number": "page_number",
        "paragraph_title": "title",
        "reference": "reference",
        "reference_content": "reference",
        "seal": "figure",
        "table": "table",
        "table_caption": "table_caption",
        "text": "text",
        "title": "title",
        "vertical_text": "text",
        "vision_footnote": "footnote",
    }

    def collect_evidence(
        self,
        results: list,
        image: np.ndarray,
        img_idx: int = 0,
        source_file: Optional[str] = None,
    ) -> RecognitionEvidence:
        height, width = image.shape[:2]
        items = []
        for source_index, region in enumerate(results or ()):
            if not isinstance(region, dict):
                continue
            type_name = str(region.get("type") or "text").lower()
            raw_type = str(region.get("raw_type") or type_name).lower()
            bbox = self._bbox(region.get("bbox"), width, height)
            result = region.get("res")
            lines = tuple(self._text_lines(result))
            roi = region.get("img")
            if not isinstance(roi, np.ndarray):
                roi = image[
                    max(0, round(bbox.y1)):min(height, round(bbox.y2)),
                    max(0, round(bbox.x1)):min(width, round(bbox.x2)),
                ]
            attributes = {
                "source_index": source_index,
                "source_type": type_name,
                "source_attributes": region.get("attributes") or {},
            }
            if isinstance(result, dict) and "cells" in result:
                attributes["table_cells"] = result["cells"]
            items.append(
                RecognitionItem(
                    evidence_id=f"p{img_idx:04d}_r{source_index:04d}",
                    category=self._CATEGORY_MAP.get(type_name, type_name),
                    bbox=bbox,
                    model_order=self._number(region.get("model_order"), source_index),
                    confidence=self._number(region.get("score"), 1.0),
                    text_lines=lines,
                    image_base64=self._encode_image(roi),
                    html=result.get("html") if isinstance(result, dict) else None,
                    latex=result.get("latex") if isinstance(result, dict) else None,
                    raw_type=raw_type,
                    layout_model=region.get("layout_model"),
                    attributes=attributes,
                )
            )
        items.sort(key=lambda item: (item.model_order, item.attributes["source_index"]))
        page = RecognitionPage(img_idx, width, height, tuple(items), image_path=source_file)
        return RecognitionEvidence((page,), source_file=source_file)

    @staticmethod
    def _bbox(value: Any, width: int, height: int) -> Rect:
        try:
            return Rect.from_sequence(value)
        except (TypeError, ValueError):
            return Rect(0.0, 0.0, float(width), float(height))

    @staticmethod
    def _number(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @classmethod
    def _text_lines(cls, result: Any):
        if not isinstance(result, list):
            return
        for item in result:
            if isinstance(item, dict):
                polygon = item.get("text_region") or item.get("poly") or ()
                yield TextEvidence(
                    text=str(item.get("text") or ""),
                    confidence=cls._number(item.get("confidence"), 1.0),
                    polygon=cls._polygon(polygon),
                )
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                polygon, text_confidence = item
                text = text_confidence[0] if isinstance(text_confidence, (list, tuple)) else text_confidence
                confidence = text_confidence[1] if isinstance(text_confidence, (list, tuple)) and len(text_confidence) > 1 else 1.0
                yield TextEvidence(str(text or ""), cls._number(confidence, 1.0), cls._polygon(polygon))

    @staticmethod
    def _polygon(value: Any) -> tuple[tuple[float, float], ...]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(
            (float(point[0]), float(point[1]))
            for point in value
            if isinstance(point, (list, tuple)) and len(point) >= 2
        )

    @staticmethod
    def _encode_image(image: np.ndarray) -> Optional[str]:
        if not isinstance(image, np.ndarray) or image.size == 0:
            return None
        success, buffer = cv2.imencode(".png", image)
        return base64.b64encode(buffer.tobytes()).decode("ascii") if success else None
