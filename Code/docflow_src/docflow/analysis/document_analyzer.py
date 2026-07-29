"""Resolve immutable recognition evidence into semantic document elements."""

from __future__ import annotations

import base64
import io
import re
from collections import Counter
from dataclasses import replace
from statistics import median
from typing import Iterable, Optional

from bs4 import BeautifulSoup
from PIL import Image

from docflow.appearance.color_inferrer import infer_crop_style, infer_table_row_fills, infer_table_rule_style
from docflow.appearance.font_classifier import FONT_FAMILY_BY_LABEL, FONT_INK_HEIGHT_RATIO
from docflow.model.stages import (
    AnalysisDiagnostic,
    AnalysisPage,
    DocumentAnalysis,
    RecognitionEvidence,
    RecognitionItem,
    Rect,
    SemanticElement,
    TextStructure,
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
_LIST_MARKER_RE = re.compile(
    r"^\s*(?:[\u2022\u25cf\u25aa\u25e6]|[\u2460-\u2473]|(?:\(?\d{1,3}\)?|[A-Za-z])[.)\u3001\uff0e])"
)


def _join_text_segments(left: str, right: str) -> str:
    if not left or not right:
        return left or right
    previous = left.rstrip()[-1]
    current = right.lstrip()[0]
    cjk = lambda char: "\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff"
    separator = " " if not cjk(previous) and not cjk(current) and (
        previous.isalnum() and current.isalnum()
        or previous in ",.;:!?)%]”’" and current.isalnum()
    ) else ""
    return left + separator + right


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
            if target_category == "figure" and self._is_shared_figure_caption(caption, items, page):
                continue
            text = self._text(caption).strip()
            peer = next(
                (
                    item
                    for item in items
                    if item.evidence_id != caption.evidence_id
                    and item.category in _CAPTION_TARGET
                    and -page.height_px * 0.005 <= item.bbox.y1 - caption.bbox.y2 <= page.height_px * 0.03
                    and abs((item.bbox.x1 + item.bbox.x2 - caption.bbox.x1 - caption.bbox.x2) / 2.0)
                    <= max(item.bbox.width, caption.bbox.width) * 0.15
                ),
                None,
            ) if re.match(r"^(?:table|fig(?:ure)?\.?)\s*\w+$", text, re.IGNORECASE) else None
            target = self._nearest_relation(peer or caption, items, target_category, page.width_px, page.height_px)
            if target is None and caption.category in _CAPTION_TARGET:
                nearby = filter(
                    None,
                    (
                        self._nearest_relation(caption, items, category, page.width_px, page.height_px)
                        for category in ("figure", "table", "formula")
                    ),
                )
                target = min(nearby, key=lambda item: self._rect_distance(caption.bbox, item.bbox), default=None)
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
        visible_lines = self._visible_lines(primary)
        line_boxes = tuple(bbox for _line, _value, bbox in visible_lines if bbox is not None)
        visual_rows = self._visual_row_boxes(line_boxes)
        text = "" if kind in {"figure_group", "table_group", "equation_group"} else self._text(primary)
        if text and len(visual_rows) == 1:
            text = self._single_row_text(visible_lines, text)
        if kind == "heading":
            text = self._normalize_heading(text)
        content_bbox = self._union(visual_rows) if visual_rows else primary.bbox
        split_text_rows = self._split_text_rows(primary, visible_lines, visual_rows)
        caption_lines = [
            line
            for caption in captions
            for line in self._visible_lines(caption)
        ]
        payload = {
            "confidence": primary.confidence,
            "image_base64": primary.image_base64,
            "html": primary.html,
            "latex": primary.latex,
            "lines": tuple(value for _line, value, _bbox in visible_lines),
            "line_heights_px": tuple(
                self._line_height(line, bbox) for line, _value, bbox in visible_lines if bbox is not None
            ),
            "line_tops_px": tuple(
                bbox.y1 for _line, _value, bbox in visible_lines if bbox is not None
            ),
            "line_lefts_px": tuple(
                bbox.x1 for _line, _value, bbox in visible_lines if bbox is not None
            ),
            "line_widths_px": tuple(
                bbox.width for _line, _value, bbox in visible_lines if bbox is not None
            ),
            "line_pitches_px": tuple(
                following.y1 - current.y1
                for current, following in zip(visual_rows, visual_rows[1:])
            ),
            "visual_line_count": len(visual_rows),
            "split_text_rows": split_text_rows,
            "caption": self._merge_caption_text(captions),
            "caption_line_heights_px": tuple(
                self._line_height(line, bbox) for line, _value, bbox in caption_lines if bbox is not None
            ),
            "caption_alignment": self._caption_alignment(primary, captions),
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
            payload["table_rule_style"] = infer_table_rule_style(primary.image_base64)
        return SemanticElement(
            element_id=f"p{page_index:04d}_{kind}_{primary.evidence_id}",
            kind=kind,
            bbox=self._union(item.bbox for item in related),
            model_order=primary.model_order,
            source_ids=tuple(item.evidence_id for item in related),
            text=text,
            payload=payload,
            text_structure=self._text_structure(primary, visible_lines, bool(split_text_rows)),
            content_bbox=content_bbox,
        )

    @staticmethod
    def _text_structure(primary: RecognitionItem, visible_lines, preserve_visual_rows: bool = False) -> TextStructure:
        lines = tuple(value for _line, value, _bbox in visible_lines)
        marked = tuple(bool(_LIST_MARKER_RE.match(str(line))) for line in lines)
        preserve_lines = preserve_visual_rows or len(lines) >= 2 and sum(marked) >= 2 and (
            all(marked) or not marked[0] and all(marked[1:])
        )
        boxes = tuple(bbox for _line, _value, bbox in visible_lines if bbox is not None)
        if len(boxes) == len(lines) and len(boxes) >= 2:
            center = (primary.bbox.x1 + primary.bbox.x2) / 2.0
            median_width = median(box.width for box in boxes)
            preserve_lines = preserve_lines or (
                max(box.x1 for box in boxes) - min(box.x1 for box in boxes) > primary.bbox.width * 0.05
                and all(abs((box.x1 + box.x2) / 2.0 - center) <= primary.bbox.width * 0.025 for box in boxes)
                and median_width <= primary.bbox.width * 0.92
            )
        lefts = tuple(bbox.x1 for _line, _value, bbox in visible_lines if bbox is not None)
        hanging_indent = 0.0
        if preserve_lines and not marked[0] and len(lefts) == len(lines):
            hanging_indent = max(median(lefts[1:]) - lefts[0], 0.0)
        text = DocumentAnalyzer._text(primary)
        vertical = primary.raw_type == "vertical_text" or (
            primary.category == "text"
            and primary.bbox.height >= primary.bbox.width * 2.5
            and sum("\u4e00" <= char <= "\u9fff" for char in text) >= 3
        )
        return TextStructure(
            preserve_source_lines=preserve_lines,
            is_list=any(marked),
            hanging_indent_px=hanging_indent,
            orientation="vertical" if vertical else "horizontal",
        )

    def _infer_visual_style(self, item: RecognitionItem) -> dict:
        if item.category not in _TEXT_CATEGORIES | {"table"} or not item.image_base64:
            return {}
        result = {}
        if item.category in _TEXT_CATEGORIES:
            style = infer_crop_style(item.image_base64)
            if style is not None:
                result["text_color"] = style.text_color
                if style.background_color:
                    result["background_color"] = style.background_color
        content = self._text(item) or BeautifulSoup(str(item.html or ""), "html.parser").get_text()
        if self.font_classifier is None or not any("\u4e00" <= char <= "\u9fff" for char in content):
            return result
        try:
            image = Image.open(io.BytesIO(base64.b64decode(item.image_base64))).convert("RGB")
            if item.category == "table" and item.html:
                row_count = max(len(BeautifulSoup(item.html, "html.parser").find_all("tr")), 1)
                predictions = [
                    self.font_classifier.predict_image(
                        image.crop((0, round(image.height * row / row_count), image.width, round(image.height * (row + 1) / row_count)))
                    )
                    for row in range(row_count)
                ]
            else:
                line_crops = self._text_line_crops(image, item)
                predictions = [
                    self.font_classifier.predict_image(crop)
                    for crop in line_crops or (image,)
                ]
        except Exception:
            self.font_classifier = None
            return result
        accepted = [prediction for prediction in predictions if prediction.accepted and prediction.label in FONT_FAMILY_BY_LABEL]
        votes = Counter(prediction.label for prediction in accepted)
        label, count = votes.most_common(1)[0] if votes else (None, 0)
        prediction = max((value for value in accepted if value.label == label), key=lambda value: value.confidence, default=predictions[0])
        if item.category == "table" and (count < 2 or count * 2 <= len(accepted)):
            prediction = predictions[0]
            label = None
        family = FONT_FAMILY_BY_LABEL.get(label or prediction.label)
        if prediction.accepted and family and (item.category != "table" or label):
            result["font_family"] = family
        result["font_prediction"] = {
            "label": prediction.label,
            "confidence": prediction.confidence,
            "margin": prediction.margin,
            "accepted": prediction.accepted,
        }
        return result

    @classmethod
    def _text_line_crops(cls, image: Image.Image, item: RecognitionItem):
        scale_x = image.width / max(item.bbox.width, 1.0)
        scale_y = image.height / max(item.bbox.height, 1.0)
        crops = []
        for line in item.text_lines:
            bbox = cls._polygon_rect(line.polygon)
            if bbox is None or bbox.width < bbox.height:
                continue
            left = max(bbox.x1, item.bbox.x1)
            top = max(bbox.y1, item.bbox.y1)
            right = min(bbox.x2, item.bbox.x2)
            bottom = min(bbox.y2, item.bbox.y2)
            if right <= left or bottom <= top:
                continue
            crop = image.crop(
                (
                    round((left - item.bbox.x1) * scale_x),
                    round((top - item.bbox.y1) * scale_y),
                    round((right - item.bbox.x1) * scale_x),
                    round((bottom - item.bbox.y1) * scale_y),
                )
            )
            if crop.width and crop.height:
                crops.append(crop)
        return tuple(crops)

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

    @classmethod
    def _is_shared_figure_caption(cls, caption: RecognitionItem, items, page) -> bool:
        candidates = [
            item
            for item in items
            if item.category == "figure"
            and 0 <= caption.bbox.y1 - item.bbox.y2 <= page.height_px * 0.06
            and max(0.0, min(caption.bbox.x2, item.bbox.x2) - max(caption.bbox.x1, item.bbox.x1)) > 0
        ]
        if len(candidates) < 2:
            return False
        left, right = min(candidates, key=lambda item: item.bbox.x1), max(candidates, key=lambda item: item.bbox.x2)
        if left is right or left.bbox.x2 > right.bbox.x1:
            return False
        union_center = (left.bbox.x1 + right.bbox.x2) / 2.0
        caption_center = (caption.bbox.x1 + caption.bbox.x2) / 2.0
        return abs(caption_center - union_center) <= (right.bbox.x2 - left.bbox.x1) * 0.08

    @classmethod
    def _merge_caption_text(cls, captions: Iterable[RecognitionItem]) -> str:
        parts = []
        for item in sorted(captions, key=lambda value: (value.bbox.y1, value.bbox.x1)):
            text = "\n".join(value for _line, value, _bbox in cls._visible_lines(item))
            normalized = re.sub(r"\s+", "", text).casefold()
            if not normalized:
                continue
            contained = next(
                (index for index, part in enumerate(parts) if normalized in part[0] or part[0] in normalized),
                None,
            )
            if contained is None:
                parts.append((normalized, text, item.bbox))
            elif len(normalized) > len(parts[contained][0]):
                parts[contained] = (normalized, text, item.bbox)
        rows = []
        for _normalized, text, bbox in parts:
            center = (bbox.y1 + bbox.y2) / 2.0
            if rows and abs(center - rows[-1][0]) <= min(bbox.height, rows[-1][1]) * 0.35:
                rows[-1][0] = (rows[-1][0] + center) / 2.0
                rows[-1][1] = max(rows[-1][1], bbox.height)
                rows[-1][2].append((bbox.x1, text))
            else:
                rows.append([center, bbox.height, [(bbox.x1, text)]])
        return "\n".join(
            "\t".join(text for _left, text in sorted(items))
            for _center, _height, items in rows
        )

    @classmethod
    def _caption_alignment(cls, primary: RecognitionItem, captions: Iterable[RecognitionItem]) -> str:
        items = tuple(captions)
        if not items:
            return "center"
        bbox = cls._union(item.bbox for item in items)
        tolerance = max(primary.bbox.width * 0.08, 1.0)
        if abs((bbox.x1 + bbox.x2 - primary.bbox.x1 - primary.bbox.x2) / 2.0) <= tolerance:
            return "center"
        if abs(bbox.x1 - primary.bbox.x1) <= tolerance:
            return "left"
        if abs(bbox.x2 - primary.bbox.x2) <= tolerance:
            return "right"
        return "center"

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

    @classmethod
    def _text(cls, item: RecognitionItem) -> str:
        parts = [value for _line, value, _bbox in cls._visible_lines(item)]
        output = ""
        for part in parts:
            if output.endswith("-") and part[:1].islower():
                output = output[:-1] + part
            else:
                output = _join_text_segments(output, part)
        return output

    @staticmethod
    def _single_row_text(visible_lines, fallback: str) -> str:
        if len(visible_lines) < 2 or any(bbox is None for _line, _value, bbox in visible_lines):
            return fallback
        segments = sorted(visible_lines, key=lambda item: item[2].x1)
        glyph_width = median(bbox.width / max(len(value), 1) for _line, value, bbox in segments)
        output = segments[0][1]
        for previous, current in zip(segments, segments[1:]):
            gap = current[2].x1 - previous[2].x2
            if gap >= glyph_width * 0.20:
                output += " " * max(1, round(gap / glyph_width))
            output += current[1]
        return output

    @classmethod
    def _visible_lines(cls, item: RecognitionItem):
        visible = []
        for line in item.text_lines:
            text = line.text.strip()
            line_bbox = cls._polygon_rect(line.polygon)
            if not text:
                continue
            if line_bbox is None:
                visible.append((line, text, None))
                continue
            horizontal = line_bbox.width >= line_bbox.height
            orthogonal_overlap = (
                min(line_bbox.y2, item.bbox.y2) - max(line_bbox.y1, item.bbox.y1)
                if horizontal
                else min(line_bbox.x2, item.bbox.x2) - max(line_bbox.x1, item.bbox.x1)
            )
            orthogonal_span = line_bbox.height if horizontal else line_bbox.width
            if orthogonal_overlap / max(orthogonal_span, 1.0) < 0.50:
                continue
            start = line_bbox.x1 if horizontal else line_bbox.y1
            end = line_bbox.x2 if horizontal else line_bbox.y2
            clip_start = item.bbox.x1 if horizontal else item.bbox.y1
            clip_end = item.bbox.x2 if horizontal else item.bbox.y2
            span = max(end - start, 1.0)
            glyph_span = span / max(len(text), 1)
            left = round(len(text) * max(clip_start - start, 0.0) / span) if clip_start - start >= glyph_span * 0.5 else 0
            right = round(len(text) * max(end - clip_end, 0.0) / span) if end - clip_end >= glyph_span * 0.5 else 0
            clipped = text[left : max(len(text) - right, left)].strip()
            if clipped:
                visible_bbox = Rect(
                    max(line_bbox.x1, item.bbox.x1),
                    max(line_bbox.y1, item.bbox.y1),
                    min(line_bbox.x2, item.bbox.x2),
                    min(line_bbox.y2, item.bbox.y2),
                )
                visible.append((line, clipped, visible_bbox))
        return visible

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
            score = distance + min(order_gap, 10.0) * 0.003
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
                content_bbox = element.content_bbox or element.bbox
                line_count = max(len(element.payload.get("lines") or ()), 1)
                line_heights = element.payload.get("line_heights_px") or ()
                line_tops = element.payload.get("line_tops_px") or ()
                line_pitches = element.payload.get("line_pitches_px") or ()
                if len(line_tops) == len(line_heights) and line_tops:
                    row_bottoms = []
                    for top, height in sorted(zip(line_tops, line_heights)):
                        if not row_bottoms or float(top) >= row_bottoms[-1] - float(height) * 0.10:
                            row_bottoms.append(float(top) + float(height))
                        else:
                            row_bottoms[-1] = max(row_bottoms[-1], float(top) + float(height))
                    line_count = len(row_bottoms)
                if element.text_structure.orientation == "vertical":
                    line_count = max(sum(not char.isspace() for char in element.text), 1)
                    raw_size = content_bbox.height / line_count * scale / 1.05
                elif line_heights and element.content_bbox is not None:
                    font = element.payload.get("font_family") or (
                        "黑体" if element.kind == "heading" else "宋体"
                    )
                    raw_size = median(line_heights) * scale / FONT_INK_HEIGHT_RATIO[font]
                    if line_pitches:
                        raw_size = min(raw_size, median(line_pitches) * scale / 1.05)
                else:
                    source_height = content_bbox.height / line_count
                    if line_heights:
                        source_height = min(source_height, median(line_heights) * 1.2)
                    raw_size = source_height * scale / 1.05
                raw_size = round(max(raw_size, 1.0) * 2.0) / 2.0
                base = "heading" if element.kind == "heading" else "caption" if element.kind == "caption" else "body"
                font = element.payload.get("font_family") or ("黑体" if base == "heading" else "宋体")
                color = element.payload.get("text_color") or "#000000"
                samples.append((element, raw_size, base, font, color))

        paragraph_sizes = {}
        for element, size, base, _font, color in samples:
            lines = element.payload.get("lines") or ()
            if base == "body" and element.kind == "paragraph_group" and len(lines) >= 2 and len(element.text) >= 40:
                color_bucket = tuple(int(color[index : index + 2], 16) // 32 for index in (1, 3, 5))
                paragraph_sizes.setdefault((base, color_bucket), []).append(size)
        paragraph_consensus = {}
        for key, sizes in paragraph_sizes.items():
            center = median(sizes)
            support = sum(abs(size - center) <= max(1.0, center * 0.20) for size in sizes)
            if len(sizes) >= 3 and support * 2 > len(sizes):
                paragraph_consensus[key] = center
        body_sizes = [size for sizes in paragraph_sizes.values() for size in sizes]
        body_consensus = median(body_sizes) if len(body_sizes) >= 3 else None
        if body_consensus is not None:
            support = sum(abs(size - body_consensus) <= max(1.0, body_consensus * 0.20) for size in body_sizes)
            body_consensus = body_consensus if support * 2 > len(body_sizes) else None

        normalized_samples = []
        for element, size, base, font, color in samples:
            lines = element.payload.get("lines") or ()
            if base == "body" and element.kind == "paragraph_group":
                color_bucket = tuple(int(color[index : index + 2], 16) // 32 for index in (1, 3, 5))
                local_consensus = paragraph_consensus.get((base, color_bucket))
                if local_consensus is not None and abs(size - local_consensus) <= max(1.0, local_consensus * 0.40):
                    size = local_consensus
                if body_consensus is not None and not element.payload.get("line_heights_px") and size < body_consensus * 0.60:
                    size = body_consensus
            elif (
                base == "heading"
                and body_consensus is not None
                and not element.payload.get("line_heights_px")
                and size < body_consensus * 0.60
            ):
                size = body_consensus
            normalized_samples.append((element, size, base, font, color))
        samples = normalized_samples

        clusters = {}
        assignments: dict[str, str] = {}
        for element, size, base, font, color in sorted(samples, key=lambda item: (item[2], item[4], item[1])):
            color_bucket = tuple(int(color[index : index + 2], 16) // 32 for index in (1, 3, 5))
            groups = clusters.setdefault((base, color_bucket), [])
            if not groups or abs(size - median(groups[-1]["sizes"])) > max(1.0, median(groups[-1]["sizes"]) * 0.08):
                groups.append({"sizes": [size], "samples": [(element, font, color)]})
            else:
                groups[-1]["sizes"].append(size)
                groups[-1]["samples"].append((element, font, color))

        roles = []
        role_counts = {}
        for (base, _color_bucket), groups in clusters.items():
            for group in groups:
                font_counts = {}
                for _element, font, _color in group["samples"]:
                    font_counts[font] = font_counts.get(font, 0) + 1
                default_font = "黑体" if base == "heading" else "宋体"
                dominant_font = max(font_counts, key=lambda font: (font_counts[font], font == default_font, font))
                colors = [color for _element, _font, color in group["samples"]]
                cluster_color = "#" + "".join(
                    f"{round(median(int(color[index : index + 2], 16) for color in colors)):02X}"
                    for index in (1, 3, 5)
                )
                role_counts[base] = role_counts.get(base, 0) + 1
                role_id = f"{base}_{role_counts[base]}"
                roles.append(
                    TypographicRole(
                        role_id,
                        dominant_font,
                        "Times New Roman",
                        median(group["sizes"]),
                        1.0,
                        bold=base == "heading",
                        color=cluster_color,
                    )
                )
                for element, _font, _color in group["samples"]:
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
    def _line_height(line: TextEvidence, bbox: Rect) -> float:
        points = tuple(line.polygon)
        if len(points) < 4:
            return bbox.height
        edges = [
            ((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2) ** 0.5
            for left, right in zip(points, points[1:] + points[:1])
        ]
        return min((edge for edge in edges if edge > 0), default=bbox.height)

    @staticmethod
    def _visual_row_boxes(boxes: Iterable[Rect]) -> tuple[Rect, ...]:
        rows = []
        for box in sorted(boxes, key=lambda value: (value.y1, value.x1)):
            if rows and box.y1 < rows[-1].y2 - min(box.height, rows[-1].height) * 0.10:
                rows[-1] = DocumentAnalyzer._union((rows[-1], box))
            else:
                rows.append(box)
        return tuple(rows)

    @staticmethod
    def _split_text_rows(primary: RecognitionItem, visible_lines, rows: tuple[Rect, ...]):
        grouped = []
        for row in rows:
            entries = sorted(
                (
                    (value, bbox)
                    for _line, value, bbox in visible_lines
                    if bbox is not None and row.y1 <= (bbox.y1 + bbox.y2) / 2.0 <= row.y2
                ),
                key=lambda item: item[1].x1,
            )
            grouped.append(entries)
        if len(grouped) < 2 or any(len(row) != 2 for row in grouped):
            return ()
        if any(row[1][1].x1 - row[0][1].x2 < primary.bbox.width * 0.08 for row in grouped):
            return ()
        return tuple(tuple(value for value, _bbox in row) for row in grouped)

    @staticmethod
    def _union(rectangles: Iterable[Rect]) -> Rect:
        rects = tuple(rectangles)
        return Rect(
            min(rect.x1 for rect in rects),
            min(rect.y1 for rect in rects),
            max(rect.x2 for rect in rects),
            max(rect.y2 for rect in rects),
        )
