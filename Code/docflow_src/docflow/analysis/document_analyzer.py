"""Resolve immutable recognition evidence into semantic document elements."""

from __future__ import annotations

import base64
import io
import re
from dataclasses import replace
from statistics import median
from typing import Iterable, Optional

from bs4 import BeautifulSoup
from PIL import Image

from docflow.layout.color_inferrer import infer_crop_style, infer_table_row_fills
from docflow.layout.font_classifier import FONT_FAMILY_BY_LABEL
from docflow.model.blocks.text_block import join_text_segments
from docflow.model.stages import (
    AnalysisDiagnostic,
    AnalysisPage,
    DocumentAnalysis,
    RecognitionEvidence,
    RecognitionItem,
    Rect,
    SemanticElement,
    TypographicRole,
)


_FORMULA_NUMBER_RE = re.compile(r"^\s*\(?\s*\d{1,3}[a-zA-Z]?\s*\)?\s*$")
_CAPTION_TARGET = {
    "figure_caption": "figure",
    "table_caption": "table",
    "formula_caption": "formula",
}
_TEXT_CATEGORIES = {
    "abstract",
    "code",
    "figure_caption",
    "footnote",
    "footer",
    "header",
    "page_number",
    "reference",
    "table_caption",
    "table_footnote",
    "text",
    "title",
}
_FURNITURE = {"header", "footer", "page_number"}


class DocumentAnalyzer:
    """Build semantic groups without changing PP-DocLayoutV3 Model Order."""

    def __init__(self, font_classifier=None) -> None:
        self.font_classifier = font_classifier

    def analyze(self, evidence: RecognitionEvidence) -> DocumentAnalysis:
        pages = tuple(self._analyze_page(page) for page in evidence.pages)
        roles, assignments = self._infer_roles(pages)
        assigned_pages = tuple(
            replace(
                page,
                elements=tuple(
                    replace(element, role_id=assignments.get(element.element_id))
                    for element in page.elements
                ),
            )
            for page in pages
        )
        return DocumentAnalysis(assigned_pages, roles, source_file=evidence.source_file)

    def _analyze_page(self, page) -> AnalysisPage:
        items, duplicate_children, diagnostics = self._merge_duplicates(page.items)
        by_id = {item.evidence_id: item for item in page.items}
        consumed: set[str] = set()
        children: dict[str, list[str]] = {
            item.evidence_id: list(duplicate_children.get(item.evidence_id, ()))
            for item in items
        }

        for child in items:
            if child.raw_type != "inline_formula":
                continue
            parents = [
                parent
                for parent in items
                if parent.category in {"text", "abstract", "reference"}
                and self._coverage(parent.bbox, child.bbox) >= 0.80
            ]
            if parents:
                parent = min(parents, key=lambda item: self._area(item.bbox))
                children[parent.evidence_id].append(child.evidence_id)
                consumed.add(child.evidence_id)

        for child in items:
            parents = [
                parent
                for parent in items
                if parent.evidence_id != child.evidence_id
                and parent.category in {"figure", "table"}
                and child.category not in _CAPTION_TARGET
                and self._coverage(parent.bbox, child.bbox) >= 0.90
            ]
            if not parents:
                continue
            parent = min(parents, key=lambda item: self._area(item.bbox))
            if parent.category == "figure" and child.category not in _TEXT_CATEGORIES | {"formula", "figure"}:
                continue
            children[parent.evidence_id].append(child.evidence_id)
            consumed.add(child.evidence_id)

        captions: dict[str, list[str]] = {item.evidence_id: [] for item in items}
        for caption in items:
            target_category = self._caption_target(caption)
            if not target_category or caption.evidence_id in consumed:
                continue
            target = self._nearest_relation(caption, items, target_category, page.width_px, page.height_px)
            if target is None:
                diagnostics.append(
                    AnalysisDiagnostic(
                        "unresolved_caption",
                        f"No {target_category} is close enough to caption evidence.",
                        (caption.evidence_id,),
                        caption.confidence,
                    )
                )
                continue
            captions[target.evidence_id].append(caption.evidence_id)
            consumed.add(caption.evidence_id)

        formula_numbers: dict[str, str] = {}
        for number in items:
            if number.evidence_id in consumed or not self._is_formula_number(number):
                continue
            target = self._formula_for_number(number, items, page.width_px)
            if target is None:
                continue
            formula_numbers[target.evidence_id] = self._text(number)
            children[target.evidence_id].append(number.evidence_id)
            consumed.add(number.evidence_id)

        elements = []
        for item in items:
            if item.evidence_id in consumed:
                continue
            source_ids = [item.evidence_id]
            source_ids.extend(children[item.evidence_id])
            source_ids.extend(captions[item.evidence_id])
            related = [by_id[source_id] for source_id in source_ids]
            element = self._make_element(
                item,
                related,
                [by_id[source_id] for source_id in captions[item.evidence_id]],
                formula_numbers.get(item.evidence_id),
                page.page_index,
            )
            elements.append(element)
            if element.payload.get("structure_missing"):
                diagnostics.append(
                    AnalysisDiagnostic(
                        "table_structure_missing",
                        "Table evidence has no editable cell structure.",
                        element.source_ids,
                        item.confidence,
                    )
                )

        elements.sort(key=lambda item: item.model_order)
        return AnalysisPage(page.page_index, page.width_px, page.height_px, tuple(elements), tuple(diagnostics))

    def _make_element(
        self,
        primary: RecognitionItem,
        related: list[RecognitionItem],
        captions: list[RecognitionItem],
        formula_number: Optional[str],
        page_index: int,
    ) -> SemanticElement:
        kind = self._kind(primary)
        text = "" if primary.category in {"figure", "table", "formula"} else self._text(primary)
        if kind == "heading":
            text = self._normalize_heading(text)
        payload = {
            "confidence": primary.confidence,
            "image_base64": primary.image_base64,
            "html": primary.html,
            "latex": primary.latex,
            "lines": tuple(line.text for line in primary.text_lines),
            "line_heights_px": tuple(
                max(point[1] for point in line.polygon) - min(point[1] for point in line.polygon)
                for line in primary.text_lines
                if line.polygon
            ),
            "line_lefts_px": tuple(
                min(point[0] for point in line.polygon)
                for line in primary.text_lines
                if line.polygon
            ),
            "caption": " ".join(filter(None, (self._text(item) for item in captions))),
            "caption_position": (
                "before"
                if captions and min(item.bbox.y1 for item in captions) < primary.bbox.y1
                else "after"
            ),
            "embedded_source_ids": tuple(
                item.evidence_id
                for item in related[1:]
                if item not in captions and not self._is_formula_number(item)
            ),
        }
        payload.update(self._infer_visual_style(primary))
        if primary.category == "formula":
            number_item = next((item for item in related[1:] if self._is_formula_number(item)), None)
            detected_line = next(
                (line for line in primary.text_lines if _FORMULA_NUMBER_RE.match(line.text or "")),
                None,
            )
            detected_number = detected_line.text.strip() if detected_line else ""
            payload["number"] = formula_number or detected_number
            number_bbox = number_item.bbox if number_item else self._polygon_rect(detected_line.polygon if detected_line else ())
            payload["number_bbox"] = (
                (number_bbox.x1, number_bbox.y1, number_bbox.x2, number_bbox.y2)
                if number_bbox
                else None
            )
        payload["primary_bbox"] = (primary.bbox.x1, primary.bbox.y1, primary.bbox.x2, primary.bbox.y2)
        if primary.category == "table" and not primary.html and not primary.attributes.get("table_cells"):
            payload["structure_missing"] = True
        if primary.category == "table" and primary.html:
            row_count = len(BeautifulSoup(primary.html, "html.parser").find_all("tr"))
            payload["table_row_styles"] = infer_table_row_fills(primary.image_base64, row_count)
        return SemanticElement(
            element_id=f"p{page_index:04d}_{kind}_{primary.evidence_id}",
            kind=kind,
            bbox=self._union(item.bbox for item in related),
            model_order=min(item.model_order for item in related),
            source_ids=tuple(item.evidence_id for item in related),
            text=text,
            payload=payload,
        )

    def _infer_visual_style(self, item: RecognitionItem) -> dict:
        if item.category not in _TEXT_CATEGORIES or not item.image_base64:
            return {}
        style = infer_crop_style(item.image_base64)
        result = {}
        if style is not None:
            result["text_color"] = style.text_color
            if style.background_color:
                result["background_color"] = style.background_color
        if self.font_classifier is None or not any("\u4e00" <= char <= "\u9fff" for char in self._text(item)):
            return result
        try:
            image = Image.open(io.BytesIO(base64.b64decode(item.image_base64))).convert("RGB")
            prediction = self.font_classifier.predict_image(image)
        except Exception:
            self.font_classifier = None
            return result
        family = FONT_FAMILY_BY_LABEL.get(prediction.label)
        if prediction.accepted and family:
            result["font_family"] = family
        result["font_prediction"] = {
            "label": prediction.label,
            "confidence": prediction.confidence,
            "margin": prediction.margin,
            "accepted": prediction.accepted,
        }
        return result

    @staticmethod
    def _kind(item: RecognitionItem) -> str:
        if item.raw_type == "header_image":
            return "header"
        if item.raw_type == "footer_image":
            return "footer"
        category = item.category
        if category == "title":
            return "heading"
        if category == "figure":
            return "figure_group"
        if category == "table":
            return "table_group"
        if category == "formula":
            return "equation_group"
        if category in _FURNITURE:
            return category
        if category.endswith("caption") or category == "table_footnote":
            return "caption"
        return "paragraph_group"

    @staticmethod
    def _caption_target(item: RecognitionItem) -> Optional[str]:
        text = DocumentAnalyzer._text(item).strip().lower()
        if re.match(r"^(?:table|表)\s*\w", text):
            return "table"
        if re.match(r"^(?:fig(?:ure)?\.?|图)\s*\w", text):
            return "figure"
        return _CAPTION_TARGET.get(item.category)

    def _merge_duplicates(
        self, items: Iterable[RecognitionItem]
    ) -> tuple[list[RecognitionItem], dict[str, list[str]], list[AnalysisDiagnostic]]:
        canonical: list[RecognitionItem] = []
        duplicate_children: dict[str, list[str]] = {}
        diagnostics: list[AnalysisDiagnostic] = []
        for item in items:
            duplicate = next(
                (
                    kept
                    for kept in canonical
                    if kept.category == item.category
                    and self._coverage(kept.bbox, item.bbox) >= 0.85
                    and self._same_content(kept, item)
                ),
                None,
            )
            if duplicate is None:
                canonical.append(item)
                continue
            winner = max((duplicate, item), key=self._evidence_completeness)
            if winner is item:
                canonical[canonical.index(duplicate)] = item
                duplicate_children[item.evidence_id] = [
                    duplicate.evidence_id,
                    *duplicate_children.pop(duplicate.evidence_id, []),
                ]
            else:
                duplicate_children.setdefault(duplicate.evidence_id, []).append(item.evidence_id)
            diagnostics.append(
                AnalysisDiagnostic(
                    "duplicate_evidence_merged",
                    "Overlapping evidence describes the same semantic object.",
                    (duplicate.evidence_id, item.evidence_id),
                    min(duplicate.confidence, item.confidence),
                )
            )
        canonical.sort(key=lambda item: item.model_order)
        return canonical, duplicate_children, diagnostics

    @staticmethod
    def _same_content(left: RecognitionItem, right: RecognitionItem) -> bool:
        left_text = DocumentAnalyzer._text(left).replace(" ", "")
        right_text = DocumentAnalyzer._text(right).replace(" ", "")
        if not left_text and not right_text:
            return True
        return bool(left_text and right_text and (left_text in right_text or right_text in left_text))

    @staticmethod
    def _evidence_completeness(item: RecognitionItem) -> tuple[int, float, float]:
        content = len(DocumentAnalyzer._text(item)) + len(item.html or "") + len(item.latex or "")
        return content, item.confidence, DocumentAnalyzer._area(item.bbox)

    @staticmethod
    def _is_formula_number(item: RecognitionItem) -> bool:
        return item.raw_type == "formula_number" or (
            item.category == "formula" and bool(_FORMULA_NUMBER_RE.match(DocumentAnalyzer._text(item)))
        )

    @staticmethod
    def _text(item: RecognitionItem) -> str:
        parts = [line.text.strip() for line in item.text_lines if line.text.strip()]
        output = ""
        for part in parts:
            if output.endswith("-") and part[:1].islower():
                output = output[:-1] + part
            else:
                output = join_text_segments([output, part])
        return output

    @staticmethod
    def _normalize_heading(text: str) -> str:
        normalized = re.sub(r"^\s*[)）]\s*", "", text or "")
        return re.sub(r"^(\d+(?:\.\d+)*)(?=[A-Za-z\u4e00-\u9fff])", r"\1 ", normalized)

    @classmethod
    def _nearest_relation(
        cls,
        source: RecognitionItem,
        items: Iterable[RecognitionItem],
        category: str,
        page_width: int,
        page_height: int,
    ) -> Optional[RecognitionItem]:
        page_diagonal = max((page_width**2 + page_height**2) ** 0.5, 1.0)
        candidates = []
        for target in items:
            if target.category != category or target.evidence_id == source.evidence_id:
                continue
            distance = cls._rect_distance(source.bbox, target.bbox) / page_diagonal
            order_gap = abs(source.model_order - target.model_order)
            score = distance + min(order_gap, 10.0) * 0.01
            candidates.append((score, target))
        if not candidates:
            return None
        score, target = min(candidates, key=lambda item: item[0])
        return target if score <= 0.12 else None

    @classmethod
    def _formula_for_number(
        cls,
        number: RecognitionItem,
        items: Iterable[RecognitionItem],
        page_width: int,
    ) -> Optional[RecognitionItem]:
        candidates = []
        for formula in items:
            if formula.category != "formula" or cls._is_formula_number(formula):
                continue
            vertical_overlap = max(0.0, min(number.bbox.y2, formula.bbox.y2) - max(number.bbox.y1, formula.bbox.y1))
            if vertical_overlap / max(min(number.bbox.height, formula.bbox.height), 1.0) < 0.40:
                continue
            if formula.bbox.x1 >= number.bbox.x1:
                continue
            horizontal_gap = max(number.bbox.x1 - formula.bbox.x2, 0.0) / max(page_width, 1)
            order_gap = abs(number.model_order - formula.model_order) * 0.005
            candidates.append((horizontal_gap + order_gap, formula))
        if not candidates:
            return None
        score, target = min(candidates, key=lambda item: item[0])
        return target if score <= 0.50 else None

    def _infer_roles(
        self, pages: tuple[AnalysisPage, ...]
    ) -> tuple[tuple[TypographicRole, ...], dict[str, str]]:
        samples = []
        for page in pages:
            scale = 841.89 / max(page.width_px, page.height_px)
            for element in page.elements:
                if element.kind in {"figure_group", "table_group", "equation_group"}:
                    continue
                line_count = max(len(element.payload.get("lines") or ()), 1)
                line_heights = element.payload.get("line_heights_px") or ()
                source_height = median(line_heights) if line_heights else element.bbox.height / line_count
                raw_size = source_height * scale / 1.05
                raw_size = round(max(raw_size, 1.0) * 2.0) / 2.0
                base = "heading" if element.kind == "heading" else "caption" if element.kind == "caption" else "body"
                font = element.payload.get("font_family") or ("黑体" if base == "heading" else "宋体")
                color = element.payload.get("text_color") or "#000000"
                samples.append((element, raw_size, base, font, color))

        clusters = {}
        assignments: dict[str, str] = {}
        for element, size, base, font, color in sorted(samples, key=lambda item: (item[2], item[3], item[4], item[1])):
            groups = clusters.setdefault((base, font, color), [])
            if not groups or abs(size - median(groups[-1]["sizes"])) > max(1.25, median(groups[-1]["sizes"]) * 0.15):
                groups.append({"sizes": [size], "elements": [element]})
            else:
                groups[-1]["sizes"].append(size)
                groups[-1]["elements"].append(element)

        roles = []
        role_counts = {}
        for (base, font, color), groups in clusters.items():
            for group in groups:
                role_counts[base] = role_counts.get(base, 0) + 1
                role_id = f"{base}_{role_counts[base]}"
                roles.append(
                    TypographicRole(
                        role_id,
                        font,
                        "Times New Roman",
                        median(group["sizes"]),
                        1.0,
                        bold=base == "heading",
                        color=color,
                    )
                )
                for element in group["elements"]:
                    assignments[element.element_id] = role_id
        return tuple(roles), assignments

    @staticmethod
    def _area(rect: Rect) -> float:
        return max(rect.x2 - rect.x1, 0.0) * max(rect.y2 - rect.y1, 0.0)

    @classmethod
    def _coverage(cls, outer: Rect, inner: Rect) -> float:
        width = max(0.0, min(outer.x2, inner.x2) - max(outer.x1, inner.x1))
        height = max(0.0, min(outer.y2, inner.y2) - max(outer.y1, inner.y1))
        return width * height / max(cls._area(inner), 1.0)

    @staticmethod
    def _rect_distance(left: Rect, right: Rect) -> float:
        dx = max(left.x1 - right.x2, right.x1 - left.x2, 0.0)
        dy = max(left.y1 - right.y2, right.y1 - left.y2, 0.0)
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _polygon_rect(polygon) -> Optional[Rect]:
        if not polygon:
            return None
        return Rect(
            min(point[0] for point in polygon),
            min(point[1] for point in polygon),
            max(point[0] for point in polygon),
            max(point[1] for point in polygon),
        )

    @staticmethod
    def _union(rectangles: Iterable[Rect]) -> Rect:
        rects = tuple(rectangles)
        return Rect(
            min(rect.x1 for rect in rects),
            min(rect.y1 for rect in rects),
            max(rect.x2 for rect in rects),
            max(rect.y2 for rect in rects),
        )
