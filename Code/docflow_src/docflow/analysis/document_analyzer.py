"""Resolve immutable recognition evidence into semantic document elements."""

from __future__ import annotations

import re
from dataclasses import replace
from statistics import median
from typing import Iterable, Optional

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
            target_category = _CAPTION_TARGET.get(caption.category)
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
            target = self._nearest_relation(number, items, "formula", page.width_px, page.height_px)
            if target is None or target.evidence_id == number.evidence_id:
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
        kind = self._kind(primary.category)
        text = self._text(primary)
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
            "caption": " ".join(filter(None, (self._text(item) for item in captions))),
            "embedded_source_ids": tuple(
                item.evidence_id
                for item in related[1:]
                if item not in captions and not self._is_formula_number(item)
            ),
        }
        if primary.category == "formula":
            detected_number = next(
                (line.text.strip() for line in primary.text_lines if _FORMULA_NUMBER_RE.match(line.text or "")),
                "",
            )
            payload["number"] = formula_number or detected_number
        if primary.category == "table" and not primary.html and not primary.attributes.get("table_cells"):
            payload["structure_missing"] = True
        return SemanticElement(
            element_id=f"p{page_index:04d}_{kind}_{primary.evidence_id}",
            kind=kind,
            bbox=self._union(item.bbox for item in related),
            model_order=min(item.model_order for item in related),
            source_ids=tuple(item.evidence_id for item in related),
            text=text,
            payload=payload,
        )

    @staticmethod
    def _kind(category: str) -> str:
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
        return join_text_segments([line.text.strip() for line in item.text_lines if line.text.strip()])

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

    def _infer_roles(
        self, pages: tuple[AnalysisPage, ...]
    ) -> tuple[tuple[TypographicRole, ...], dict[str, str]]:
        samples: list[tuple[SemanticElement, float, str]] = []
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
                samples.append((element, raw_size, base))

        clusters: dict[str, list[list[float]]] = {}
        assignments: dict[str, str] = {}
        for _element, size, base in sorted(samples, key=lambda item: (item[2], item[1])):
            groups = clusters.setdefault(base, [])
            if not groups or abs(size - median(groups[-1])) > max(1.25, median(groups[-1]) * 0.15):
                groups.append([size])
            else:
                groups[-1].append(size)

        roles = []
        for base, groups in clusters.items():
            for index, sizes in enumerate(groups, 1):
                role_id = f"{base}_{index}"
                roles.append(
                    TypographicRole(
                        role_id,
                        "黑体" if base == "heading" else "宋体",
                        "Times New Roman",
                        median(sizes),
                        1.0,
                        bold=base == "heading",
                    )
                )
                for element, size, sample_base in samples:
                    if sample_base == base and size in sizes and element.element_id not in assignments:
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
    def _union(rectangles: Iterable[Rect]) -> Rect:
        rects = tuple(rectangles)
        return Rect(
            min(rect.x1 for rect in rects),
            min(rect.y1 for rect in rects),
            max(rect.x2 for rect in rects),
            max(rect.y2 for rect in rects),
        )
