"""Resolve immutable recognition evidence into semantic document elements."""

from __future__ import annotations

import base64
import io
import re
import unicodedata
from collections import Counter
from dataclasses import replace
from statistics import median
from typing import Iterable, Optional

from bs4 import BeautifulSoup
import numpy as np
from PIL import Image

from docflow.appearance.color_inferrer import (
    infer_background_extent,
    infer_crop_style,
    infer_table_row_fills,
    infer_table_rule_style,
)
from docflow.appearance.font_classifier import FONT_FAMILY_BY_LABEL, FONT_INK_HEIGHT_RATIO
from docflow.appearance.text_weight_inferrer import infer_text_stroke_ratio
from docflow.model.stages import (
    AnalysisDiagnostic,
    AnalysisPage,
    DocumentAnalysis,
    RecognitionEvidence,
    RecognitionItem,
    Rect,
    SemanticElement,
    TextRow,
    TextEvidence,
    TextSpan,
    TextStructure,
    TypographicRole,
    inline_formula_marker,
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
    r"^\s*(?:[\u2022\u25cf\u25aa\u25e6]|[\u2460-\u2473]|(?:\(?\d{1,3}\)?|[A-Za-z])[.)\u3001\uff0e](?!\d))"
)
_INLINE_LIST_MARKER_RE = re.compile(
    r"[;\uff1b]\s*(?:[\u2460-\u2473]|(?:\(?\d{1,3}\)?|[A-Za-z])[.)\u3001\uff0e](?!\d))"
)
_PARAGRAPH_CONTINUATION_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "because",
    "before",
    "between",
    "but",
    "by",
    "can",
    "could",
    "during",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "may",
    "might",
    "must",
    "not",
    "of",
    "on",
    "or",
    "over",
    "should",
    "than",
    "that",
    "the",
    "to",
    "under",
    "was",
    "were",
    "while",
    "who",
    "whose",
    "will",
    "with",
    "within",
    "without",
    "would",
}
_LATEX_GREEK_SYMBOLS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "lambda": "λ",
    "mu": "μ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "phi": "φ",
    "psi": "ψ",
    "omega": "ω",
    "Delta": "Δ",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
}


def _join_text_segments(left: str, right: str) -> str:
    if not left or not right:
        return left or right
    previous = left.rstrip()[-1]
    current = right.lstrip()[0]

    def cjk(char: str) -> bool:
        return "\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff"

    separator = " " if not cjk(previous) and not cjk(current) and (
        previous.isalnum() and current.isalnum()
        or previous in ",.;:!?)%]”’" and current.isalnum()
    ) else ""
    return left + separator + right


def _text_width_units(text: str) -> float:
    value = str(text or "")
    if not value:
        return 0.0
    total = 0.0
    for char in value:
        if char.isspace():
            total += 0.35
        elif ord(char) >= 0x2E80 or unicodedata.east_asian_width(char) in {"F", "W"}:
            total += 1.0
        else:
            total += 0.50
    return total


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
        page_image = self._load_page_image(page.image_path, page.page_index)
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
            pending = children[item.evidence_id] + captions[item.evidence_id]
            for source_id in pending:
                if source_id in source_ids:
                    continue
                source_ids.append(source_id)
                pending.extend(children.get(source_id, ()))
                pending.extend(captions.get(source_id, ()))
            related = [by_id[source_id] for source_id in source_ids]
            element = self._make_element(
                item,
                related,
                [by_id[source_id] for source_id in captions[item.evidence_id]],
                formula_numbers.get(item.evidence_id),
                page.page_index,
                page.width_px,
                page.height_px,
                page_image,
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
        elements = list(self._merge_continuation_paragraphs(elements))
        elements = list(self._split_body_text_paragraphs(elements))
        return AnalysisPage(page.page_index, page.width_px, page.height_px, tuple(elements), tuple(diagnostics))

    @classmethod
    def _split_body_text_paragraphs(cls, elements: list[SemanticElement]) -> tuple[SemanticElement, ...]:
        split: list[SemanticElement] = []
        for element in elements:
            if (
                element.kind != "paragraph_group"
                or len(element.text_rows) < 2
                or element.text_structure.preserve_source_lines
                or element.text_structure.is_list
                or element.text_structure.tabular_rows
                or element.text_structure.orientation != "horizontal"
            ):
                split.append(element)
                continue
            lines = [
                {
                    "row": row,
                    "text": row.text.strip(),
                    "left": row.bbox.x1,
                    "top": row.bbox.y1,
                    "right": row.bbox.x2,
                    "bottom": row.bbox.y2,
                    "center_y": (row.bbox.y1 + row.bbox.y2) / 2.0,
                    "height": row.bbox.height,
                    "bbox": (row.bbox.x1, row.bbox.y1, row.bbox.x2, row.bbox.y2),
                    "is_list": bool(_LIST_MARKER_RE.match(row.text.strip())),
                }
                for row in element.text_rows
                if row.text.strip() and row.bbox.width > 0 and row.bbox.height > 0
            ]
            groups = cls._group_logical_lines(lines, max(element.bbox.width, 1.0), element.bbox)
            if len(groups) < 2:
                split.append(element)
                continue
            for index, group in enumerate(groups):
                bbox = Rect(
                    min(line["left"] for line in group),
                    min(line["top"] for line in group),
                    max(line["right"] for line in group),
                    max(line["bottom"] for line in group),
                )
                text = ""
                for line in group:
                    text = _join_text_segments(text, line["text"])
                payload = dict(element.payload)
                payload["paragraph_break_before"] = index > 0
                payload["typographic_group"] = element.element_id
                payload["primary_bbox"] = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
                split.append(
                    replace(
                        element,
                        element_id=f"{element.element_id}_p{index + 1}",
                        bbox=bbox,
                        model_order=element.model_order + index * 0.001,
                        text=text,
                        payload=payload,
                        text_structure=replace(
                            element.text_structure,
                            preserve_source_lines=False,
                            hanging_indent_px=0.0,
                        ),
                        content_bbox=bbox,
                        text_rows=tuple(line["row"] for line in group),
                    )
                )
        return tuple(split)

    @classmethod
    def _merge_continuation_paragraphs(cls, elements: list[SemanticElement]) -> tuple[SemanticElement, ...]:
        merged: list[SemanticElement] = []
        for element in elements:
            if merged and cls._continues_paragraph(merged[-1], element):
                merged[-1] = cls._merge_paragraph_elements(merged[-1], element)
            else:
                merged.append(element)
        return tuple(merged)

    @staticmethod
    def _continues_paragraph(previous: SemanticElement, current: SemanticElement) -> bool:
        if previous.kind != "paragraph_group" or current.kind != "paragraph_group":
            return False
        if current.payload.get("paragraph_break_before"):
            return False
        if previous.text_structure.is_list or current.text_structure.is_list:
            return False
        if previous.text_structure.orientation != current.text_structure.orientation:
            return False
        left_text = previous.text.strip()
        right_text = current.text.strip()
        if not left_text or not right_text or _LIST_MARKER_RE.match(right_text):
            return False
        if re.search(r"[.!?。！？][\"'”’」』）)]*$", left_text):
            return False

        overlap = max(0.0, min(previous.bbox.x2, current.bbox.x2) - max(previous.bbox.x1, current.bbox.x1))
        if overlap / max(min(previous.bbox.width, current.bbox.width), 1.0) < 0.60:
            return False

        row_heights = [row.bbox.height for row in previous.text_rows if row.bbox.height > 0]
        row_tops = [row.bbox.y1 for row in previous.text_rows]
        line_height = median(row_heights) if row_heights else max(previous.bbox.height, 1.0)
        line_pitch = median(right - left for left, right in zip(row_tops, row_tops[1:])) if len(row_tops) >= 2 else line_height * 1.2
        gap = current.bbox.y1 - previous.bbox.y2
        if gap < -line_height * 0.25 or gap > max(line_height * 1.5, line_pitch * 1.15):
            return False

        if re.search(r"[,;:，；：、(\[{]$", left_text):
            return True
        last_word = re.search(r"([A-Za-z]+)\s*$", left_text)
        if last_word and last_word.group(1).lower() in _PARAGRAPH_CONTINUATION_WORDS:
            return True
        return right_text[0].islower() and left_text[-1].isalnum()

    @classmethod
    def _merge_paragraph_elements(cls, previous: SemanticElement, current: SemanticElement) -> SemanticElement:
        bbox = cls._union((previous.bbox, current.bbox))
        content_boxes = tuple(
            value for value in (previous.content_bbox, current.content_bbox) if value is not None
        )
        payload = dict(previous.payload)
        payload["confidence"] = min(previous.payload.get("confidence", 1.0), current.payload.get("confidence", 1.0))
        payload["primary_bbox"] = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        payload["embedded_source_ids"] = tuple(
            dict.fromkeys(
                tuple(previous.payload.get("embedded_source_ids") or ())
                + tuple(current.payload.get("embedded_source_ids") or ())
            )
        )
        payload["inline_formulas"] = tuple(previous.payload.get("inline_formulas") or ()) + tuple(
            current.payload.get("inline_formulas") or ()
        )
        return replace(
            previous,
            bbox=bbox,
            source_ids=tuple(dict.fromkeys((*previous.source_ids, *current.source_ids))),
            text=_join_text_segments(previous.text, current.text),
            child_ids=tuple(dict.fromkeys((*previous.child_ids, *current.child_ids, current.element_id))),
            payload=payload,
            text_structure=replace(previous.text_structure, preserve_source_lines=False, hanging_indent_px=0.0),
            content_bbox=cls._union(content_boxes) if content_boxes else None,
            text_rows=previous.text_rows + current.text_rows,
        )

    def _make_element(
        self,
        primary: RecognitionItem,
        related: list[RecognitionItem],
        captions: list[RecognitionItem],
        formula_number: Optional[str],
        page_index: int,
        page_width_px: int,
        page_height_px: int,
        page_image: Optional[Image.Image] = None,
    ) -> SemanticElement:
        kind = self._kind(primary)
        footer_table = kind == "table_group" and self._is_page_furniture_table(
            primary,
            page_width_px,
            page_height_px,
        )
        if footer_table:
            kind = "footer"
        visible_lines = self._visible_lines(primary)
        if kind == "header" and self._is_substantive_header(primary, visible_lines, page_height_px):
            kind = "paragraph_group"
        line_boxes = tuple(bbox for _line, _value, bbox in visible_lines if bbox is not None)
        visual_rows = self._visual_row_boxes(line_boxes)
        text_rows = self._text_rows(visible_lines, visual_rows)
        inline_formulas = tuple(
            sorted(
                (
                    item
                    for item in related[1:]
                    if item.raw_type == "inline_formula" and item.latex
                ),
                key=lambda item: (item.bbox.y1, item.bbox.x1),
            )
        )
        if inline_formulas:
            text_rows = self._merge_inline_formulas(text_rows, inline_formulas)
        if footer_table:
            text = self._table_text(primary.html)
        else:
            text = "" if kind in {"figure_group", "table_group", "equation_group"} else self._text(primary)
        if inline_formulas and text_rows:
            text = ""
            for row in text_rows:
                text = _join_text_segments(text, row.text)
        elif text and len(text_rows) == 1:
            text = text_rows[0].text
        if kind == "heading":
            text = self._normalize_heading(text)
        content_bbox = self._union(visual_rows) if visual_rows else primary.bbox
        tabular_rows = self._has_tabular_rows(primary, text_rows)
        caption_lines = [
            line
            for caption in captions
            for line in self._visible_lines(caption)
        ]
        payload = {
            "confidence": primary.confidence,
            "image_base64": primary.image_base64,
            "html": primary.html if kind == "table_group" else None,
            "latex": primary.latex,
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
            "inline_formulas": tuple(
                {
                    "latex": item.latex,
                    "image_base64": item.image_base64,
                    "bbox": (item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2),
                }
                for item in inline_formulas
            ),
        }
        payload.update(self._infer_visual_style(primary))
        if page_image is not None and payload.get("background_color"):
            background_bbox = infer_background_extent(
                np.asarray(page_image),
                (primary.bbox.x1, primary.bbox.y1, primary.bbox.x2, primary.bbox.y2),
                payload["background_color"],
            )
            if background_bbox is not None:
                payload["background_bbox"] = background_bbox
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
        if kind == "table_group" and not primary.html and not primary.attributes.get("table_cells"):
            payload["structure_missing"] = True
        if kind == "table_group" and primary.html:
            soup = BeautifulSoup(primary.html, "html.parser")
            rows = soup.find_all("tr")
            row_count = len(rows)
            column_count = max(
                (
                    sum(max(int(cell.get("colspan", 1)), 1) for cell in row.find_all(["td", "th"], recursive=False))
                    for row in rows
                ),
                default=0,
            )
            source_attributes = primary.attributes.get("source_attributes") or {}
            if source_attributes.get("table_content_fit"):
                payload["table_content_fit"] = True
            if source_attributes.get("table_auto_height"):
                payload["table_auto_height"] = True
            cells = primary.attributes.get("table_cells") or ()
            payload["table_line_heights_px"] = tuple(
                float(obj["bbox"][3]) - float(obj["bbox"][1])
                for cell in cells
                for obj in cell.get("ocr_objects") or ()
                if str(obj.get("text") or "").strip()
                and len(obj.get("bbox") or ()) >= 4
                and float(obj["bbox"][3]) > float(obj["bbox"][1])
            )
            row_ratios = self._table_axis_ratios(cells, row_count, "row", "rowspan", 1, 3)
            column_edges = self._table_column_edges(cells, column_count)
            column_ratios = self._ratios_from_edges(column_edges)
            if row_ratios:
                payload["table_row_height_ratios"] = row_ratios
            if column_ratios:
                payload["table_column_width_ratios"] = column_ratios
            try:
                table_image = Image.open(io.BytesIO(base64.b64decode(primary.image_base64))).convert("RGB")
            except Exception:
                table_image = None
            cell_styles = self._table_cell_styles(cells, column_edges, table_image)
            if cell_styles:
                payload["table_cell_styles"] = cell_styles
            paragraph_html = self._table_html_with_paragraphs(
                primary.html,
                cells,
                column_edges,
                cell_styles,
            )
            if paragraph_html:
                payload["html"] = paragraph_html
            cell_geometry = self._table_cell_geometry(cells)
            if cell_geometry:
                payload["table_cell_geometry"] = cell_geometry
            payload["table_row_styles"] = infer_table_row_fills(primary.image_base64, row_count, row_ratios)
            payload["table_rule_style"] = infer_table_rule_style(primary.image_base64)
        return SemanticElement(
            element_id=f"p{page_index:04d}_{kind}_{primary.evidence_id}",
            kind=kind,
            bbox=self._union(item.bbox for item in related),
            model_order=primary.model_order,
            source_ids=tuple(item.evidence_id for item in related),
            text=text,
            payload=payload,
            text_structure=self._text_structure(primary, visible_lines, text_rows, tabular_rows),
            content_bbox=content_bbox,
            text_rows=text_rows,
        )

    @staticmethod
    def _table_text(html_text: str | None) -> str:
        soup = BeautifulSoup(str(html_text or ""), "html.parser")
        table = soup.find("table")
        if table is None:
            return ""
        return " ".join(
            text
            for text in (cell.get_text(" ", strip=True) for cell in table.find_all(["td", "th"]))
            if text
        )

    @classmethod
    def _is_page_furniture_table(cls, item: RecognitionItem, page_width: int, page_height: int) -> bool:
        if not item.html or page_width <= 0 or page_height <= 0:
            return False
        bbox = item.bbox
        if bbox.y1 < page_height * 0.84 or bbox.height > page_height * 0.20:
            return False
        soup = BeautifulSoup(str(item.html), "html.parser")
        table = soup.find("table")
        if table is None:
            return False
        cells = table.find_all(["td", "th"])
        texts = [cell.get_text(" ", strip=True) for cell in cells]
        texts = [text for text in texts if text]
        return 0 < len(texts) <= 4 and len(cells) <= 8

    @staticmethod
    def _load_page_image(path: Optional[str], page_index: int) -> Optional[Image.Image]:
        if not path:
            return None
        try:
            image = Image.open(path)
            image.seek(page_index)
            return image.convert("RGB")
        except (OSError, EOFError):
            return None

    @staticmethod
    def _text_structure(
        primary: RecognitionItem,
        visible_lines,
        text_rows: tuple[TextRow, ...],
        tabular_rows: bool = False,
    ) -> TextStructure:
        lines = tuple(row.text for row in text_rows) or tuple(value for _line, value, _bbox in visible_lines)
        marked = tuple(bool(_LIST_MARKER_RE.match(str(line))) for line in lines)
        inline_list = any(_INLINE_LIST_MARKER_RE.search(str(line)) for line in lines)
        preserve_lines = tabular_rows or (
            not inline_list
            and len(lines) >= 2
            and sum(marked) >= 2
            and (all(marked) or not marked[0] and all(marked[1:]))
        )
        boxes = tuple(row.bbox for row in text_rows)
        if len(boxes) == len(lines) and len(boxes) >= 2:
            center = (primary.bbox.x1 + primary.bbox.x2) / 2.0
            median_width = median(box.width for box in boxes)
            preserve_lines = preserve_lines or (
                max(box.x1 for box in boxes) - min(box.x1 for box in boxes) > primary.bbox.width * 0.05
                and all(abs((box.x1 + box.x2) / 2.0 - center) <= primary.bbox.width * 0.025 for box in boxes)
                and median_width <= primary.bbox.width * 0.92
            )
        lefts = tuple(box.x1 for box in boxes)
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
            tabular_rows=tabular_rows,
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
        try:
            image = Image.open(io.BytesIO(base64.b64decode(item.image_base64))).convert("RGB")
        except Exception:
            return result
        line_crops = self._text_line_crops(image, item) if item.category in _TEXT_CATEGORIES else ()
        stroke_ratio = infer_text_stroke_ratio(line_crops or (image,))
        if stroke_ratio is not None:
            result["stroke_ratio"] = stroke_ratio
        content = self._text(item) or BeautifulSoup(str(item.html or ""), "html.parser").get_text()
        if self.font_classifier is None or not any("\u4e00" <= char <= "\u9fff" for char in content):
            return result
        try:
            if item.category == "table" and item.html:
                row_count = max(len(BeautifulSoup(item.html, "html.parser").find_all("tr")), 1)
                predictions = [
                    self.font_classifier.predict_image(
                        image.crop((0, round(image.height * row / row_count), image.width, round(image.height * (row + 1) / row_count)))
                    )
                    for row in range(row_count)
                ]
            else:
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
                    if self._duplicate_categories(kept, item)
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
    def _duplicate_categories(left: RecognitionItem, right: RecognitionItem) -> bool:
        if left.category == right.category:
            return True
        return {left.category, right.category} == {"header", "text"}

    @staticmethod
    def _is_substantive_header(primary: RecognitionItem, visible_lines, page_height_px: int) -> bool:
        if primary.raw_type == "header_image":
            return False
        texts = [str(text).strip() for _line, text, _bbox in visible_lines if str(text).strip()]
        return (
            len(texts) >= 3
            and sum(len(text) for text in texts) >= 80
            and primary.bbox.height >= page_height_px * 0.04
        )

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
                if element.kind in {
                    "figure_group",
                    "table_group",
                    "equation_group",
                    "header",
                    "footer",
                    "page_number",
                }:
                    continue
                content_bbox = element.content_bbox or element.bbox
                line_count = max(len(element.text_rows) or len(element.payload.get("lines") or ()), 1)
                line_heights = (
                    tuple(row.ink_height_px or row.bbox.height for row in element.text_rows)
                    or element.payload.get("line_heights_px")
                    or ()
                )
                line_tops = (
                    tuple(row.bbox.y1 for row in element.text_rows)
                    or element.payload.get("line_tops_px")
                    or ()
                )
                line_pitches = (
                    tuple(
                        right.bbox.y1 - left.bbox.y1
                        for left, right in zip(element.text_rows, element.text_rows[1:])
                    )
                    or element.payload.get("line_pitches_px")
                    or ()
                )
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
                elif element.kind == "heading" and line_count == 1:
                    primary_bbox = (
                        Rect.from_sequence(element.payload["primary_bbox"])
                        if element.payload.get("primary_bbox")
                        else element.bbox
                    )
                    raw_size = primary_bbox.height * scale / 1.05
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
                if element.text_structure.orientation != "vertical":
                    width_size = self._line_width_font_size(element, scale)
                    if width_size is not None:
                        raw_size = min(raw_size, width_size)
                raw_size = round(max(raw_size, 1.0) * 2.0) / 2.0
                base = "heading" if element.kind == "heading" else "caption" if element.kind == "caption" else "body"
                font = element.payload.get("font_family") or ("黑体" if base == "heading" else "宋体")
                color = element.payload.get("text_color") or "#000000"
                samples.append((element, raw_size, base, font, color, element.payload.get("stroke_ratio")))

        typographic_sizes: dict[str, list[float]] = {}
        for element, size, _base, _font, _color, _stroke in samples:
            group = element.payload.get("typographic_group")
            if group:
                typographic_sizes.setdefault(str(group), []).append(size)
        if typographic_sizes:
            samples = [
                (
                    element,
                    median(typographic_sizes[str(element.payload["typographic_group"])])
                    if element.payload.get("typographic_group") in typographic_sizes
                    else size,
                    base,
                    font,
                    color,
                    stroke,
                )
                for element, size, base, font, color, stroke in samples
            ]

        body_strokes = [
            float(stroke_ratio)
            for element, _size, base, _font, _color, stroke_ratio in samples
            if base == "body"
            and stroke_ratio is not None
            and element.kind == "paragraph_group"
            and len(element.text) >= 40
        ]
        regular_stroke = median(body_strokes) if len(body_strokes) >= 3 else None
        stroke_spread = (
            median(abs(value - regular_stroke) for value in body_strokes)
            if regular_stroke is not None
            else None
        )

        body_font_counts = Counter(
            font
            for element, _size, base, font, _color, _stroke_ratio in samples
            if base == "body"
            and element.kind == "paragraph_group"
            and len(element.text) >= 40
            and element.payload.get("font_family")
        )
        body_font = None
        if sum(body_font_counts.values()) >= 3:
            candidate, support = body_font_counts.most_common(1)[0]
            if support * 2 > sum(body_font_counts.values()):
                body_font = candidate

        weighted_samples = []
        for element, size, base, font, color, stroke_ratio in samples:
            bold = bool(
                base == "heading"
                and regular_stroke is not None
                and stroke_ratio is not None
                and float(stroke_ratio) >= regular_stroke * 1.08
                and float(stroke_ratio) - regular_stroke >= max(0.006, float(stroke_spread) * 2.5)
            )
            if base == "heading" and not bold and not element.payload.get("font_family") and body_font:
                font = body_font
            weighted_samples.append((element, size, base, font, color, bold))
        samples = weighted_samples

        paragraph_sizes = {}
        for element, size, base, _font, color, _bold in samples:
            lines = element.text_rows or element.payload.get("lines") or ()
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
        for element, size, base, font, color, bold in samples:
            lines = element.text_rows or element.payload.get("lines") or ()
            if base == "body" and element.kind == "paragraph_group":
                color_bucket = tuple(int(color[index : index + 2], 16) // 32 for index in (1, 3, 5))
                local_consensus = paragraph_consensus.get((base, color_bucket))
                if local_consensus is not None and abs(size - local_consensus) <= max(1.0, local_consensus * 0.40):
                    size = local_consensus
                if (
                    body_consensus is not None
                    and not element.text_rows
                    and not element.payload.get("line_heights_px")
                    and size < body_consensus * 0.60
                ):
                    size = body_consensus
            elif (
                base == "heading"
                and body_consensus is not None
                and not element.text_rows
                and not element.payload.get("line_heights_px")
                and size < body_consensus * 0.60
            ):
                size = body_consensus
            normalized_samples.append((element, size, base, font, color, bold))
        samples = normalized_samples

        clusters = {}
        assignments: dict[str, str] = {}
        for element, size, base, font, color, bold in sorted(samples, key=lambda item: (item[2], item[4], item[5], item[1])):
            color_bucket = tuple(int(color[index : index + 2], 16) // 32 for index in (1, 3, 5))
            groups = clusters.setdefault((base, color_bucket, bold), [])
            if not groups or abs(size - median(groups[-1]["sizes"])) > max(1.0, median(groups[-1]["sizes"]) * 0.08):
                groups.append({"sizes": [size], "samples": [(element, font, color)]})
            else:
                groups[-1]["sizes"].append(size)
                groups[-1]["samples"].append((element, font, color))

        roles = []
        role_counts = {}
        for (base, _color_bucket, bold), groups in clusters.items():
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
                        bold=bold,
                        color=cluster_color,
                    )
                )
                for element, _font, _color in group["samples"]:
                    assignments[element.element_id] = role_id
        return tuple(roles), assignments

    @staticmethod
    def _line_width_font_size(element: SemanticElement, source_scale: float) -> float | None:
        sizes = []
        for row in element.text_rows:
            units = _text_width_units(row.text)
            if units < 1.5 or row.bbox.width <= 0:
                continue
            sizes.append(row.bbox.width * source_scale / units)
        return median(sizes) if sizes else None

    @staticmethod
    def _area(rect: Rect) -> float:
        return max(rect.x2 - rect.x1, 0.0) * max(rect.y2 - rect.y1, 0.0)

    @staticmethod
    def _table_axis_ratios(cells, count: int, position_key: str, span_key: str, start: int, end: int):
        samples = [[] for _ in range(count)]
        for cell in cells:
            try:
                position = int(cell.get(position_key, 0))
                span = max(int(cell.get(span_key, 1)), 1)
                bbox = cell["bbox"]
                size = (float(bbox[end]) - float(bbox[start])) / span
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if size <= 0:
                continue
            for target in range(max(position, 0), min(position + span, count)):
                samples[target].append(size)
        if not samples or not all(samples):
            return ()
        sizes = [median(values) for values in samples]
        total = sum(sizes)
        return tuple(size / total for size in sizes) if total > 0 else ()

    @staticmethod
    def _table_column_edges(cells, count: int):
        columns = [[] for _ in range(count)]
        for cell in cells:
            try:
                if max(int(cell.get("colspan", 1)), 1) != 1:
                    continue
                column = int(cell.get("col", 0))
                x1, y1, x2, y2 = map(float, cell["bbox"][:4])
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            text_boxes = []
            for obj in cell.get("ocr_objects") or ():
                try:
                    tx1, ty1, tx2, ty2 = map(float, obj["bbox"][:4])
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
                if tx2 > tx1 and ty2 > ty1 and str(obj.get("text") or "").strip():
                    text_boxes.append((tx1, tx2))
            if text_boxes:
                text_left = min(box[0] for box in text_boxes)
                text_right = max(box[1] for box in text_boxes)
                if (text_right - text_left) >= (x2 - x1) * 0.60:
                    x1, x2 = text_left, text_right
            if 0 <= column < count and x2 > x1 and y2 > y1:
                columns[column].append((x1, x2))
        if not columns or not all(columns):
            return ()
        lefts = [min(box[0] for box in boxes) for boxes in columns]
        rights = [max(box[1] for box in boxes) for boxes in columns]
        edges = [lefts[0], *((rights[index] + lefts[index + 1]) / 2.0 for index in range(count - 1)), rights[-1]]
        return tuple(edges) if all(right > left for left, right in zip(edges, edges[1:])) else ()

    @staticmethod
    def _ratios_from_edges(edges):
        if len(edges) < 2:
            return ()
        widths = [right - left for left, right in zip(edges, edges[1:])]
        total = sum(widths)
        return tuple(width / total for width in widths) if total > 0 else ()

    @staticmethod
    def _table_cell_styles(cells, column_edges, table_image=None):
        if not column_edges:
            return ()
        styles = []
        column_samples = {}
        column_votes = {}
        formula_columns = set()
        for cell in cells:
            try:
                row = int(cell.get("row", 0))
                column = int(cell.get("col", 0))
                span = max(int(cell.get("colspan", 1)), 1)
                left, right = column_edges[column], column_edges[column + span]
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            text_boxes = []
            for obj in cell.get("ocr_objects") or ():
                try:
                    if str(obj.get("text") or "").strip():
                        x1, y1, x2, y2 = map(float, obj["bbox"][:4])
                        if x2 > x1 and y2 > y1:
                            text_boxes.append((x1, y1, x2, y2))
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
            if text_boxes:
                text_left = min(box[0] for box in text_boxes)
                text_right = max(box[2] for box in text_boxes)
            width = max(right - left, 1.0)
            alignment = DocumentAnalyzer._table_text_alignment(
                text_boxes,
                (left, right),
                width,
            )
            has_formula = any(
                str(obj.get("label") or "") in {"display_formula", "equation", "formula", "inline_formula"}
                for obj in cell.get("layout_objects") or ()
            )
            if has_formula:
                alignment = "center"
                formula_columns.add(column)
            role = str(cell.get("role") or "body")
            if role not in {"corner", "column_header"}:
                column_samples.setdefault(column, []).extend(
                    (box, left, right, width)
                    for box in text_boxes
                )
                column_votes.setdefault(column, Counter())[alignment] += 1
            stroke_ratio = None
            if table_image is not None:
                try:
                    y1 = min(float(obj["bbox"][1]) for obj in cell.get("ocr_objects") or () if str(obj.get("text") or "").strip())
                    y2 = max(float(obj["bbox"][3]) for obj in cell.get("ocr_objects") or () if str(obj.get("text") or "").strip())
                    crop = table_image.crop((max(text_left, 0), max(y1, 0), min(text_right, table_image.width), min(y2, table_image.height)))
                    stroke_ratio = infer_text_stroke_ratio((crop,))
                except (KeyError, TypeError, ValueError):
                    pass
            styles.append((row, column, alignment, role, stroke_ratio))
        column_consensus = {}
        for column, samples in column_samples.items():
            alignment = DocumentAnalyzer._table_column_alignment(samples)
            if alignment:
                column_consensus[column] = alignment
            elif column in column_votes:
                votes = column_votes[column]
                alignment, count = votes.most_common(1)[0]
                if count / sum(votes.values()) >= 0.60:
                    column_consensus[column] = alignment
        for column in formula_columns:
            column_consensus[column] = "center"
        if column_consensus:
            styles = [
                (row, column, column_consensus.get(column, alignment), role, stroke)
                for row, column, alignment, role, stroke in styles
            ]
        body_strokes = [stroke for _row, _column, _alignment, role, stroke in styles if role not in {"corner", "column_header"} and stroke is not None]
        regular = median(body_strokes) if len(body_strokes) >= 3 else None
        bold_candidates = {
            (row, column): bool(
                role in {"corner", "column_header"}
                and regular is not None
                and stroke is not None
                and stroke >= regular * 1.08
                and stroke - regular >= 0.006
            )
            for row, column, _alignment, role, stroke in styles
        }
        header_rows = {
            row
            for row in {row for row, _column, _alignment, role, _stroke in styles if role in {"corner", "column_header"}}
            if (
                sum(value for (candidate_row, _column), value in bold_candidates.items() if candidate_row == row)
                / max(sum(candidate_row == row for candidate_row, _column in bold_candidates), 1)
                >= 0.75
            )
        }
        return tuple(
            (
                row,
                column,
                alignment,
                row in header_rows and role in {"corner", "column_header"},
            )
            for row, column, alignment, role, stroke in styles
        )

    @staticmethod
    def _table_cell_geometry(cells):
        """Store scale-independent vertical alignment and padding ratios."""
        geometry = []
        for cell in cells:
            try:
                row = int(cell.get("row", 0))
                column = int(cell.get("col", 0))
                x1, y1, x2, y2 = map(float, cell["bbox"][:4])
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            boxes = []
            for obj in [*(cell.get("ocr_objects") or ()), *(cell.get("layout_objects") or ())]:
                if str(obj.get("label") or "") == "table":
                    continue
                try:
                    ox1, oy1, ox2, oy2 = map(float, obj["bbox"][:4])
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
                if ox2 > ox1 and oy2 > oy1:
                    boxes.append((ox1, oy1, ox2, oy2))
            if not boxes:
                continue
            content_y1 = min(box[1] for box in boxes)
            content_y2 = max(box[3] for box in boxes)
            content_x1 = min(box[0] for box in boxes)
            content_x2 = max(box[2] for box in boxes)
            height = y2 - y1
            width = x2 - x1
            top = max(content_y1 - y1, 0.0) / height
            bottom = max(y2 - content_y2, 0.0) / height
            start = max(content_x1 - x1, 0.0) / width
            end = max(x2 - content_x2, 0.0) / width
            if top <= 0.05:
                vertical = "top"
            elif bottom <= 0.05:
                vertical = "bottom"
            elif abs(top - bottom) <= 0.12:
                vertical = "center"
            elif top <= 0.15:
                vertical = "top"
            elif bottom <= 0.15:
                vertical = "bottom"
            else:
                vertical = "center"
            geometry.append(
                (
                    row,
                    column,
                    vertical,
                    round(min(top, 0.25), 4),
                    round(min(bottom, 0.25), 4),
                    round(min(start, 0.25), 4),
                    round(min(end, 0.25), 4),
                )
            )
        return tuple(geometry)

    @staticmethod
    def _table_text_alignment(text_boxes, column_span, width: float) -> str:
        if len(text_boxes) >= 2:
            ordered = sorted(text_boxes, key=lambda box: (box[1] + box[3]) / 2.0)
            heights = [box[3] - box[1] for box in ordered]
            tolerance = max(8.0, median(heights) * 0.65)
            lines = []
            for box in ordered:
                center_y = (box[1] + box[3]) / 2.0
                if not lines or abs(center_y - median((item[1] + item[3]) / 2.0 for item in lines[-1])) > tolerance:
                    lines.append([box])
                else:
                    lines[-1].append(box)
            if len(lines) >= 2:
                lefts = [min(box[0] for box in line) for line in lines]
                rights = [max(box[2] for box in line) for line in lines]
                centers = [(left + right) / 2.0 for left, right in zip(lefts, rights)]
                deviations = {
                    "left": median(abs(value - median(lefts)) for value in lefts),
                    "right": median(abs(value - median(rights)) for value in rights),
                    "center": median(abs(value - median(centers)) for value in centers),
                }
                competing = min(deviations["right"], deviations["center"])
                if deviations["left"] <= max(3.0, competing * 1.5):
                    return "left"
                ordered_deviations = sorted(deviations.items(), key=lambda item: item[1])
                best, best_value = ordered_deviations[0]
                second_value = ordered_deviations[1][1]
                if best_value <= max(second_value * 0.8, 1.0):
                    return best

        left, right = column_span
        text_left = min(box[0] for box in text_boxes) if text_boxes else left
        text_right = max(box[2] for box in text_boxes) if text_boxes else right
        left_gap, right_gap = max(text_left - left, 0.0), max(right - text_right, 0.0)
        if abs(left_gap - right_gap) <= width * 0.12:
            return "center"
        if right_gap <= width * 0.06 and left_gap >= width * 0.12:
            return "right"
        return "left"

    @staticmethod
    def _table_column_alignment(samples) -> Optional[str]:
        if len(samples) < 2:
            return None
        left_offsets = []
        right_offsets = []
        center_offsets = []
        for box, cell_left, cell_right, cell_width in samples:
            x1, _y1, x2, _y2 = box
            left_offsets.append(max(x1 - cell_left, 0.0) / cell_width)
            right_offsets.append(max(cell_right - x2, 0.0) / cell_width)
            center_offsets.append(((x1 + x2) / 2.0 - (cell_left + cell_right) / 2.0) / cell_width)

        def deviation(values):
            center = median(values)
            return median(abs(value - center) for value in values)

        candidates = {
            "left": deviation(left_offsets) + max(median(left_offsets) - 0.25, 0.0) * 0.5,
            "center": deviation(center_offsets) + abs(median(center_offsets)) * 0.5,
            "right": deviation(right_offsets) + max(median(right_offsets) - 0.25, 0.0) * 0.5,
        }
        ordered = sorted(candidates.items(), key=lambda item: item[1])
        best, best_score = ordered[0]
        second_score = ordered[1][1]
        if best_score <= max(second_score * 0.75, 0.02):
            return best
        return None

    @classmethod
    def _table_html_with_paragraphs(cls, html_text, cells, column_edges, cell_styles):
        if not html_text or not cells or not column_edges:
            return None
        soup = BeautifulSoup(str(html_text), "html.parser")
        table = soup.find("table")
        if table is None:
            return None
        by_position = {
            (int(cell.get("row", 0)), int(cell.get("col", 0))): cell
            for cell in cells
        }
        styles = {
            (int(row), int(column)): (str(alignment), bool(bold))
            for row, column, alignment, bold in cell_styles
        }
        changed = False
        for cell_node, row, column, row_span, column_span in cls._table_nodes_with_positions(table):
            cell = by_position.get((row, column))
            if not cell:
                continue
            try:
                span = (column_edges[column], column_edges[column + column_span])
            except IndexError:
                continue
            style = styles.get((row, column), ("center" if cell_node.name == "th" else "left", cell_node.name == "th"))
            paragraphs = cls._table_cell_paragraphs(cell, span, style)
            if not paragraphs:
                continue
            visual_nodes = []
            for visual in cell_node.find_all(["img", "span"], recursive=True):
                classes = visual.get("class") or ()
                if visual.name != "img" and "docflow-formula" not in classes:
                    continue
                bbox = cls._table_node_bbox(visual)
                visual.extract()
                visual_nodes.append((bbox, visual))
            blocks = [
                (paragraph.get("bbox") or (float("inf"),) * 4, "paragraph", paragraph)
                for paragraph in paragraphs
            ]
            blocks.extend(
                (bbox or (float("inf"),) * 4, "visual", node)
                for bbox, node in visual_nodes
            )
            blocks.sort(key=lambda item: (float(item[0][1]), float(item[0][0])))
            cell_node.clear()
            for _bbox, kind, value in blocks:
                if kind == "visual":
                    cell_node.append(value)
                    continue
                paragraph = soup.new_tag("p")
                paragraph["data-align"] = str(value["alignment"])
                paragraph["data-space-before"] = str(value["space_before"])
                if value["bold"]:
                    paragraph["data-bold"] = "true"
                if value.get("bbox"):
                    paragraph["data-bbox"] = ",".join(str(round(float(number), 3)) for number in value["bbox"])
                lines = tuple(value.get("lines") or (value["text"],))
                if len(lines) == 1:
                    paragraph.string = str(lines[0])
                else:
                    for index, line in enumerate(lines):
                        if index:
                            paragraph.append(soup.new_tag("br"))
                        paragraph.append(str(line))
                cell_node.append(paragraph)
            changed = True
        return str(soup) if changed else None

    @staticmethod
    def _table_nodes_with_positions(table):
        occupied = set()
        nodes = []
        for row_index, row in enumerate(table.find_all("tr")):
            column_index = 0
            for node in row.find_all(["td", "th"], recursive=False):
                while (row_index, column_index) in occupied:
                    column_index += 1
                try:
                    row_span = max(int(node.get("rowspan", 1)), 1)
                    column_span = max(int(node.get("colspan", 1)), 1)
                except (TypeError, ValueError):
                    row_span = column_span = 1
                nodes.append((node, row_index, column_index, row_span, column_span))
                for occupied_row in range(row_index, row_index + row_span):
                    for occupied_column in range(column_index, column_index + column_span):
                        if (occupied_row, occupied_column) != (row_index, column_index):
                            occupied.add((occupied_row, occupied_column))
                column_index += column_span
        return nodes

    @staticmethod
    def _table_node_bbox(node):
        value = node.get("data-bbox") or node.get("data-page-bbox") or ""
        try:
            bbox = tuple(float(number) for number in value.split(","))
        except (AttributeError, TypeError, ValueError):
            return None
        return bbox if len(bbox) == 4 else None

    @classmethod
    def _table_cell_paragraphs(cls, cell, column_span, style):
        fallback_alignment, bold = style
        lines = cls._table_cell_lines(cell)
        if not lines:
            text_lines = tuple(line.strip() for line in str(cell.get("text") or "").splitlines() if line.strip())
            if not text_lines:
                return ()
            return tuple(
                {
                    "text": text,
                    "lines": (text,),
                    "alignment": fallback_alignment,
                    "bold": bold,
                    "space_before": 0.0,
                    "bbox": tuple(cell.get("bbox") or ()),
                }
                for text in text_lines
            )

        cell_bbox = tuple(cell.get("bbox") or (0, 0, 0, column_span[1]))
        cell_width = max(column_span[1] - column_span[0], 1.0)
        groups = cls._group_logical_lines(lines, cell_width, cell_bbox)
        line_height = median(line["height"] for line in lines)
        pitches = [right["center_y"] - left["center_y"] for left, right in zip(lines, lines[1:])]
        pitch = median(pitches) if pitches else line_height * 1.2
        normal_gap = max(pitch - line_height, 0.0)
        has_formula = any(
            str(obj.get("label") or "") in {"display_formula", "equation", "formula", "inline_formula"}
            for obj in cell.get("layout_objects") or ()
        )

        paragraphs = []
        previous_bottom = None
        cell_top = float(cell_bbox[1])
        cell_bottom = float(cell_bbox[3])
        base_left = median(line["left"] for line in lines)
        for group in groups:
            text = ""
            for line in group:
                text = _join_text_segments(text, line["text"])
            line_texts = tuple(line["text"] for line in group)
            bbox = (
                min(line["left"] for line in group),
                min(line["top"] for line in group),
                max(line["right"] for line in group),
                max(line["bottom"] for line in group),
            )
            if has_formula:
                alignment = "center"
            elif len(groups) == 1:
                alignment = fallback_alignment
            else:
                alignment = cls._table_text_alignment(
                    [line["bbox"] for line in group],
                    column_span,
                    cell_width,
                )
            short_lines = all(line["right"] - line["left"] <= cell_width * 0.65 for line in group)
            line_alignments = [
                cls._table_line_alignment(line["bbox"], column_span)
                for line in group
            ]
            group_center = (bbox[0] + bbox[2]) / 2.0
            cell_center = (cell_bbox[0] + cell_bbox[2]) / 2.0
            cell_box_width = max(cell_bbox[2] - cell_bbox[0], 1.0)
            if (
                len(group) == 1
                and bbox[2] - bbox[0] <= cell_box_width * 0.40
                and abs(group_center - cell_center) <= cell_box_width * 0.15
            ):
                alignment = "center"
            elif (
                short_lines
                and bbox[1] >= cell_top + (cell_bottom - cell_top) * 0.65
                and (
                    bbox[0] >= base_left + cell_width * 0.12
                    or all(value == "right" for value in line_alignments)
                )
            ):
                alignment = "right"
            elif len(group) == 1 and short_lines and fallback_alignment == "center":
                alignment = "center"
            space_before = 0.0
            if previous_bottom is not None:
                extra_gap = max(bbox[1] - previous_bottom - normal_gap, 0.0)
                space_before = round(min(extra_gap / max(line_height, 1.0), 1.25), 3)
            paragraphs.append(
                {
                    "text": text,
                    "lines": (
                        line_texts
                        if (
                            alignment in {"center", "right"}
                            and len(line_texts) > 1
                            and all(
                                line["right"] - line["left"] <= (column_span[1] - column_span[0]) * 0.65
                                for line in group
                            )
                        )
                        else (text,)
                    ),
                    "alignment": alignment,
                    "bold": bold,
                    "space_before": space_before,
                    "bbox": bbox,
                }
            )
            previous_bottom = bbox[3]
        return tuple(paragraphs)

    @staticmethod
    def _table_cell_lines(cell):
        formula_boxes = []
        for obj in cell.get("layout_objects") or ():
            if str(obj.get("label") or "") not in {"display_formula", "equation", "formula", "inline_formula"}:
                continue
            try:
                formula_boxes.append(tuple(float(value) for value in obj["bbox"][:4]))
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        boxes = []
        for obj in cell.get("ocr_objects") or ():
            text = str(obj.get("text") or "").strip()
            if not text:
                continue
            try:
                x1, y1, x2, y2 = map(float, obj["bbox"][:4])
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            if any(DocumentAnalyzer._formula_bbox_overlaps_text(box, (x1, y1, x2, y2)) for box in formula_boxes):
                continue
            boxes.append(
                {
                    "text": text,
                    "left": x1,
                    "top": y1,
                    "right": x2,
                    "bottom": y2,
                    "center_y": (y1 + y2) / 2.0,
                    "height": y2 - y1,
                }
            )
        if not boxes:
            return []
        boxes.sort(key=lambda item: (item["center_y"], item["left"]))
        tolerance = max(8.0, median(box["height"] for box in boxes) * 0.65)
        lines = []
        for box in boxes:
            if not lines or abs(box["center_y"] - lines[-1]["center_y"]) > tolerance:
                lines.append({"items": [box], "center_y": box["center_y"]})
            else:
                lines[-1]["items"].append(box)
        for line in lines:
            items = line.pop("items")
            text = ""
            for item in sorted(items, key=lambda value: value["left"]):
                text = _join_text_segments(text, item["text"])
            line.update(
                {
                    "text": text,
                    "left": min(item["left"] for item in items),
                    "top": min(item["top"] for item in items),
                    "right": max(item["right"] for item in items),
                    "bottom": max(item["bottom"] for item in items),
                    "center_y": median(item["center_y"] for item in items),
                    "height": median(item["height"] for item in items),
                }
            )
            line["bbox"] = (line["left"], line["top"], line["right"], line["bottom"])
            line["is_list"] = bool(_LIST_MARKER_RE.match(text))
        return lines

    @staticmethod
    def _formula_bbox_overlaps_text(formula_bbox, text_bbox) -> bool:
        ix1 = max(formula_bbox[0], text_bbox[0])
        iy1 = max(formula_bbox[1], text_bbox[1])
        ix2 = min(formula_bbox[2], text_bbox[2])
        iy2 = min(formula_bbox[3], text_bbox[3])
        overlap = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
        area = max((text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1]), 1.0)
        if overlap / area >= 0.35:
            return True
        center_x = (text_bbox[0] + text_bbox[2]) / 2.0
        center_y = (text_bbox[1] + text_bbox[3]) / 2.0
        return formula_bbox[0] <= center_x <= formula_bbox[2] and formula_bbox[1] <= center_y <= formula_bbox[3]

    @classmethod
    def _group_logical_lines(cls, lines, frame_width: float, frame_bbox):
        if not lines:
            return ()
        frame = (
            (float(frame_bbox.x1), float(frame_bbox.x2))
            if hasattr(frame_bbox, "x1")
            else (float(frame_bbox[0]), float(frame_bbox[2]))
        )
        line_height = median(line["height"] for line in lines)
        pitches = [right["center_y"] - left["center_y"] for left, right in zip(lines, lines[1:])]
        pitch = median(pitches) if pitches else line_height * 1.2
        start_tolerance = max(6.0, line_height * 0.45)
        for line in lines:
            line["alignment"] = cls._table_line_alignment(line["bbox"], frame)

        groups = [[lines[0]]]
        for line in lines[1:]:
            previous = groups[-1][-1]
            gap = line["top"] - previous["bottom"]
            alignment = line.get("alignment", "left")
            previous_alignment = previous.get("alignment", "left")
            starts_new_group = (
                (line.get("is_list") and cls._table_line_ends_sentence(previous["text"]))
                or (
                    re.match(r"^\s*[a-z]\s+\S", line["text"])
                    and not cls._line_continues_text(previous["text"])
                )
                or line["center_y"] - previous["center_y"] > pitch * 1.35
                or gap > max(line_height * 1.45, pitch * 1.15)
                or (
                    alignment not in {"center", "right"}
                    and previous_alignment not in {"center", "right"}
                    and (
                        (
                            not cls._line_continues_text(previous["text"])
                            and re.match(r"^[A-Z][A-Za-z ]{0,24}:", line["text"])
                        )
                        or previous["text"].rstrip().endswith(":")
                        or (
                            re.match(r"^[A-Z][A-Za-z ]{0,24}:", previous["text"])
                            and re.match(r"^[A-Z0-9]", line["text"])
                        )
                        or (
                            re.match(r"^[A-Z0-9]", line["text"])
                            and not cls._line_continues_text(previous["text"])
                            and len(previous["text"].strip()) <= 60
                            and previous["right"] - previous["left"] <= frame_width * 0.75
                            and line["right"] - line["left"] <= frame_width * 0.75
                        )
                    )
                )
            )
            if starts_new_group:
                groups.append([line])
            else:
                groups[-1].append(line)

        split_groups = []
        for group in groups:
            if (
                len(group) > 1
                and group[0]["right"] - group[0]["left"] <= frame_width * 0.75
                and (
                    group[1]["left"] - group[0]["left"] > start_tolerance
                    or (
                        group[0]["right"] - group[0]["left"] <= frame_width * 0.50
                        and group[1]["right"] - group[1]["left"]
                        >= (group[0]["right"] - group[0]["left"]) * 1.50
                    )
                )
            ):
                split_groups.append([group[0]])
                split_groups.append(group[1:])
            else:
                split_groups.append(group)

        suffix_split_groups = []
        for group in split_groups:
            split_at = None
            for index in range(1, len(group) - 1):
                head_left = median(line["left"] for line in group[:index])
                tail_left = median(line["left"] for line in group[index:])
                if (
                    tail_left > head_left + start_tolerance * 1.5
                    and all(
                        abs(line["left"] - tail_left) <= start_tolerance
                        for line in group[index:]
                    )
                ):
                    split_at = index
                    break
            if split_at is None:
                suffix_split_groups.append(group)
            else:
                suffix_split_groups.append(group[:split_at])
                suffix_split_groups.append(group[split_at:])

        groups = []
        rail_tolerance = max(start_tolerance * 2.0, frame_width * 0.08)
        for group in suffix_split_groups:
            base_left = median(line["left"] for line in group)
            run_start = None
            split_at = None
            for index in range(len(group) + 1):
                shifted = index < len(group) and abs(group[index]["left"] - base_left) > rail_tolerance
                if shifted and run_start is None:
                    run_start = index
                if not shifted and run_start is not None:
                    if index - run_start >= 2:
                        split_at = (run_start, index)
                        break
                    run_start = None
            if split_at is None:
                groups.append(group)
                continue
            run_start, run_end = split_at
            if run_start:
                groups.append(group[:run_start])
            groups.append(group[run_start:run_end])
            if run_end < len(group):
                groups.append(group[run_end:])
        return tuple(tuple(group) for group in groups)

    @staticmethod
    def _table_line_alignment(bbox, column_span) -> str:
        left, right = column_span
        width = max(right - left, 1.0)
        text_left, text_right = bbox[0], bbox[2]
        left_gap = max(text_left - left, 0.0)
        right_gap = max(right - text_right, 0.0)
        edge_tolerance = max(4.0, width * 0.04)
        if left_gap <= edge_tolerance and right_gap <= edge_tolerance:
            return "left"
        center_tolerance = max(8.0, width * 0.06)
        if abs(left_gap - right_gap) <= center_tolerance:
            return "center"
        return "right" if right_gap < left_gap else "left"

    @staticmethod
    def _table_line_ends_sentence(text: str) -> bool:
        return bool(re.search(r"[.!?。！？][\"'”’」』）)]*$", str(text).strip()))

    @staticmethod
    def _line_continues_text(text: str) -> bool:
        value = str(text).strip()
        if re.search(r"[,;:，；：、(\[{]$", value):
            return True
        last_word = re.search(r"([A-Za-z]+)\s*$", value)
        return bool(last_word and last_word.group(1).lower() in _PARAGRAPH_CONTINUATION_WORDS)

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
            overlap = min(box.y2, rows[-1].y2) - max(box.y1, rows[-1].y1) if rows else 0.0
            if rows and overlap >= min(box.height, rows[-1].height) * 0.45:
                rows[-1] = DocumentAnalyzer._union((rows[-1], box))
            else:
                rows.append(box)
        return tuple(rows)

    @staticmethod
    def _text_rows(visible_lines, rows: tuple[Rect, ...]) -> tuple[TextRow, ...]:
        output = []
        for row in rows:
            entries = tuple(
                (line, value, bbox)
                for line, value, bbox in sorted(
                    (
                        (line, value, bbox)
                        for line, value, bbox in visible_lines
                        if bbox is not None and row.y1 <= (bbox.y1 + bbox.y2) / 2.0 <= row.y2
                    ),
                    key=lambda item: item[2].x1,
                )
            )
            spans = tuple(TextSpan(value, bbox) for _line, value, bbox in entries)
            if not spans:
                continue
            glyph_width = median(span.bbox.width / max(len(span.text), 1) for span in spans)
            text = spans[0].text
            for previous, current in zip(spans, spans[1:]):
                gap = current.bbox.x1 - previous.bbox.x2
                if gap >= glyph_width * 0.20:
                    text += " " * max(1, round(gap / glyph_width))
                text += current.text
            output.append(
                TextRow(
                    text,
                    row,
                    spans,
                    median(DocumentAnalyzer._line_height(line, bbox) for line, _value, bbox in entries),
                )
            )
        return tuple(output)

    @classmethod
    def _merge_inline_formulas(
        cls,
        rows: tuple[TextRow, ...],
        formulas: tuple[RecognitionItem, ...],
    ) -> tuple[TextRow, ...]:
        """按行列坐标将行内公式替换为正文中的唯一占位符。"""
        assignments: dict[int, list[tuple[int, RecognitionItem]]] = {index: [] for index in range(len(rows))}
        for formula_index, formula in enumerate(formulas):
            candidates = []
            for row_index, row in enumerate(rows):
                overlap = max(0.0, min(row.bbox.y2, formula.bbox.y2) - max(row.bbox.y1, formula.bbox.y1))
                if overlap > 0:
                    candidates.append((overlap, row_index))
            if not candidates:
                continue
            _overlap, row_index = max(candidates)
            assignments[row_index].append((formula_index, formula))

        merged = []
        for row_index, row in enumerate(rows):
            entries = sorted(assignments[row_index], key=lambda item: item[1].bbox.x1)
            if not entries or row.bbox.width <= 0:
                merged.append(row)
                continue
            segments = []
            cursor = 0
            for formula_index, formula in entries:
                start = round(len(row.text) * (formula.bbox.x1 - row.bbox.x1) / row.bbox.width)
                end = round(len(row.text) * (formula.bbox.x2 - row.bbox.x1) / row.bbox.width)
                start = min(max(start, cursor), len(row.text))
                end = min(max(end, start + 1), len(row.text))
                latex = str(formula.latex or "")
                if start > cursor and cls._formula_adjacent_char(row.text[start - 1], latex, leading=True):
                    start -= 1
                if end < len(row.text) and cls._formula_adjacent_char(row.text[end], latex, leading=False):
                    end += 1
                segments.append(row.text[cursor:start])
                segments.append(inline_formula_marker(formula_index))
                cursor = end
            segments.append(row.text[cursor:])
            merged.append(replace(row, text="".join(segments)))
        return tuple(merged)

    @staticmethod
    def _formula_adjacent_char(char: str, latex: str, leading: bool) -> bool:
        """判断公式框外的相邻字符是否仍是同一公式符号。"""
        if char.isspace() or not char:
            return False
        if char == "∞" and r"\infty" in latex:
            return True
        for command, symbol in _LATEX_GREEK_SYMBOLS.items():
            if char == symbol and re.search(rf"\\{command}(?![A-Za-z])", latex):
                return True
        if not leading or not char.isalpha():
            return False
        prefix = latex.lstrip()
        return bool(
            re.match(rf"{re.escape(char)}(?![A-Za-z])", prefix)
            or re.search(rf"\\(?:mathrm|mathbf|mathcal)\{{\s*{re.escape(char)}\s*\}}", prefix)
        )

    @staticmethod
    def _has_tabular_rows(primary: RecognitionItem, rows: tuple[TextRow, ...]) -> bool:
        if len(rows) < 2 or len(rows[0].spans) < 2:
            return False
        field_count = len(rows[0].spans)
        return all(
            len(row.spans) == field_count
            and all(
                right.bbox.x1 - left.bbox.x2 >= primary.bbox.width * 0.08
                for left, right in zip(row.spans, row.spans[1:])
            )
            for row in rows
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
