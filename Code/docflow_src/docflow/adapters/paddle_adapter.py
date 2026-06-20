"""PaddleOCR / ppstructure 输出的适配器。

将 PaddleOCR 的版面分析与结构识别结果转换为 DocFlow v2.0 标准 JSON 格式。
"""

from __future__ import annotations

import base64
import re
from difflib import SequenceMatcher
from collections import Counter
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from docflow.adapters.base_adapter import BaseAdapter


class PaddleAdapter(BaseAdapter):
    """将 PaddleOCR :class:`StructureSystem` 输出转换为 v2.0 JSON。

    预期的 *paddle_results* 格式
    ---------------------------------
    一个区域字典列表，每个字典包含：

    - ``type``    -- str，如 ``"text"``、``"table"``、``"figure"``
    - ``bbox``    -- ``[x1, y1, x2, y2]``
    - ``img``     -- numpy ndarray（页面图像的 ROI 裁剪）
    - ``res``     -- 根据类型不同（OCR 行、表格 HTML、LaTeX 等）
    - ``img_idx`` -- int（多页处理时的页面索引）
    - ``score``   -- float，检测置信度
    """

    # PaddleOCR 类型 → v2.0 类别映射
    _CATEGORY_MAP: Dict[str, str] = {
        "abstract": "abstract",
        "algorithm": "code",
        "aside_text": "text",
        "chart": "figure",
        "content": "text",
        "display_formula": "formula",
        "doc_title": "title",
        "figure_title": "figure_caption",
        "footer_image": "figure",
        "footnote": "footnote",
        "formula_number": "formula",
        "header_image": "figure",
        "image": "figure",
        "inline_formula": "formula",
        "number": "page_number",
        "paragraph_title": "title",
        "reference_content": "reference",
        "seal": "figure",
        "vertical_text": "text",
        "vision_footnote": "footnote",
        "text": "text",
        "title": "title",
        "table": "table",
        "figure": "figure",
        "equation": "formula",
        "header": "header",
        "footer": "footer",
        "figure_caption": "figure_caption",
        "table_caption": "table_caption",
        "reference": "reference",
    }

    # 携带 OCR 行结果的文本类区块类型
    _TEXT_TYPES = frozenset({
        "text", "title", "reference", "header", "footer",
        "figure_caption", "table_caption", "abstract",
        "table_footnote", "formula_caption", "footnote",
        "algorithm", "aside_text", "content", "doc_title",
        "paragraph_title", "figure_title", "reference_content",
        "vertical_text", "vision_footnote",
    })
    _DEDUP_CATEGORIES = frozenset({
        "text",
        "title",
        "reference",
        "formula",
        "figure",
        "table",
        "figure_caption",
        "table_caption",
        "table_footnote",
        "formula_caption",
        "abstract",
        "code",
        "footnote",
        "page_number",
    })
    _CAPTION_FAMILY = frozenset({
        "figure_caption",
        "table_caption",
        "formula_caption",
        "table_footnote",
    })
    _TEXTLIKE_DEDUP_CATEGORIES = frozenset({
        "text",
        "title",
        "reference",
        "figure_caption",
        "table_caption",
        "table_footnote",
        "formula_caption",
        "abstract",
        "code",
        "footnote",
        "page_number",
    })
    _GENERIC_PARENT_RAW_TYPES = frozenset({
        "content",
        "text",
        "aside_text",
        "vertical_text",
        "algorithm",
    })
    _GENERIC_PARENT_CATEGORIES = frozenset({
        "text",
        "code",
    })
    _SEMANTIC_CONTAINER_CATEGORIES = frozenset({
        "table",
        "figure",
        "header",
        "footer",
        "reference",
        "abstract",
    })
    _SEMANTIC_CONTAINER_RAW_TYPES = frozenset({
        "table",
        "image",
        "chart",
        "header",
        "header_image",
        "footer",
        "footer_image",
        "reference",
        "abstract",
        "seal",
    })
    _TITLE_LIKE_RAW_TYPES = frozenset({
        "doc_title",
        "paragraph_title",
        "figure_title",
        "table_caption",
        "figure_caption",
    })
    _FORMULA_RAW_TYPES = frozenset({
        "display_formula",
        "inline_formula",
        "formula_number",
        "equation",
        "formula",
    })
    _TOP_LEVEL_PRESERVE_RAW_TYPES = frozenset({
        "doc_title",
        "paragraph_title",
        "figure_title",
        "table_caption",
        "figure_caption",
    })
    _FIGURE_CAPTION_RE = re.compile(
        r"^\s*(?:图|fig(?:ure)?\.?)\s*[\d一二三四五六七八九十]+(?:[-－—]\d+)?\s*\S*",
        re.IGNORECASE,
    )

    def convert(
        self,
        results: list,
        image: np.ndarray,
        img_idx: int = 0,
        **kwargs,
    ) -> dict:
        """将 PaddleOCR 结果转换为 v2.0 标准 JSON。

        Parameters
        ----------
        results:
            来自 ``StructureSystem.__call__`` 的区域字典列表。
        image:
            页面图像，numpy ndarray (H x W x C)。
        img_idx:
            零基页面索引（多页文档用）。

        Returns
        -------
        符合 DocFlow v2.0 JSON Schema 的字典，包含单页条目。
        """
        h, w = image.shape[:2]
        blocks: List[Dict[str, Any]] = []
        results = self._recall_missing_figures_from_captions(results, image)
        filtered_results, cleanup_report = self._suppress_nested_duplicates(results)
        filtered_results, trim_report = self._trim_carry_over_text_regions(filtered_results)
        filtered_results = [
            self._trim_title_leading_formula_number(region)
            for region in filtered_results
        ]
        cleanup_report.extend(trim_report)
        page_attributes = None
        if cleanup_report:
            page_attributes = {
                "cleanup_removed_count": len(cleanup_report),
                "cleanup_rule_counts": dict(Counter(item["reason"] for item in cleanup_report)),
            }

        filtered_results = self._ensure_model_order(filtered_results)

        for idx, region in enumerate(filtered_results):
            block = self._convert_region(region, w, h, idx)
            blocks.append(block)

        return {
            "version": "2.0",
            "metadata": {
                "engine": "PaddleOCR",
                "source_file": None,
            },
            "pages": [
                {
                    "page_index": img_idx,
                    "width": w,
                    "height": h,
                    "attributes": page_attributes,
                    "blocks": blocks,
                }
            ],
        }

    @staticmethod
    def _ensure_model_order(results: list) -> list:
        """Keep PP-DocLayoutV3 reading order stable after cleanup passes."""
        ordered = list(results or [])
        if not ordered:
            return ordered
        order_values = []
        for index, region in enumerate(ordered):
            value = region.get("model_order") if isinstance(region, dict) else None
            try:
                order_values.append((int(value), index, region))
            except (TypeError, ValueError):
                return ordered
        order_values.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in order_values]

    def _suppress_nested_duplicates(self, results: list) -> tuple[list, list[dict]]:
        """抑制大框包小框的重复区域。

        典型场景：
        - 大 figure/table 框里又有一个同类小框
        - 大 caption 框完整包含一个更短的同类 caption

        只处理“同类重复”，不移除 figure 内部的小 caption 等异类对象。
        """
        if not isinstance(results, list) or len(results) < 2:
            return list(results or []), []

        enriched = []
        for index, region in enumerate(results):
            type_name = str(region.get("type", "text")).lower()
            mapped = self._CATEGORY_MAP.get(type_name, type_name)
            bbox = self._safe_bbox(region.get("bbox"))
            text = self._normalize_text(self._extract_region_text(region))
            score = float(region.get("score", 0.0) or 0.0)
            area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            enriched.append(
                {
                    "index": index,
                    "region": region,
                    "raw_type": type_name,
                    "category": mapped,
                    "bbox": bbox,
                    "text": text,
                    "score": score,
                    "area": area,
                }
            )
        self._annotate_contained_child_counts(enriched)

        # 同类中优先保留信息量更大、得分更高的块
        ranked = sorted(enriched, key=self._dedup_sort_key, reverse=True)

        kept: List[dict] = []
        report: List[dict] = []
        for candidate in ranked:
            if candidate["category"] not in self._DEDUP_CATEGORIES:
                kept.append(candidate)
                continue
            duplicate = False
            duplicate_reason = None
            duplicate_existing = None
            for existing in kept:
                duplicate_reason = self._is_nested_duplicate(candidate, existing)
                if duplicate_reason is not None:
                    duplicate = True
                    duplicate_existing = existing
                    break
            if not duplicate:
                kept.append(candidate)
            else:
                if duplicate_existing is not None and duplicate_reason in {
                    "semantic_container_child",
                    "table_container_child",
                    "page_strip_container_child",
                }:
                    self._attach_nested_child(
                        duplicate_existing["region"],
                        candidate["region"],
                        reason=duplicate_reason,
                    )
                report.append(
                    {
                        "reason": duplicate_reason,
                        "removed_index": candidate["index"],
                        "removed_category": candidate["category"],
                        "parent_index": duplicate_existing["index"] if duplicate_existing is not None else None,
                        "parent_category": duplicate_existing["category"] if duplicate_existing is not None else None,
                    }
                )

        kept.sort(key=lambda item: item["index"])
        return [item["region"] for item in kept], report

    @classmethod
    def _recall_missing_figures_from_captions(
        cls,
        results: list,
        image: np.ndarray,
    ) -> list:
        """Recall large visual regions that are structurally anchored by captions.

        Layout detectors sometimes keep the caption but miss the associated
        figure body. This is a detection-recall step, not a reading-order
        fallback: a candidate is accepted only when a figure caption has no
        same-column figure above it and image projection finds a large visual
        mass directly above the caption.
        """
        if not isinstance(results, list) or image is None or not hasattr(image, "shape"):
            return list(results or [])
        if image.ndim < 2:
            return list(results)

        page_h, page_w = image.shape[:2]
        if page_w <= 0 or page_h <= 0:
            return list(results)

        recalled: List[dict] = []
        existing_figures = [
            cls._safe_bbox(region.get("bbox"))
            for region in results
            if str(region.get("type", "")).lower() == "figure"
        ]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        for region in results:
            if not cls._is_figure_caption_like(region):
                continue
            caption_box = cls._safe_bbox(region.get("bbox"))
            if cls._has_paired_figure_above(caption_box, existing_figures, page_w, page_h):
                continue
            candidate = cls._find_caption_anchored_visual_region(gray, caption_box, page_w, page_h)
            if candidate is None:
                continue
            if any(cls._overlap_ratio(candidate, box) >= 0.65 for box in existing_figures):
                continue
            if any(cls._overlap_ratio(candidate, cls._safe_bbox(item.get("bbox"))) >= 0.65 for item in recalled):
                continue

            x1, y1, x2, y2 = [int(round(value)) for value in candidate]
            x1 = max(0, min(page_w, x1))
            x2 = max(0, min(page_w, x2))
            y1 = max(0, min(page_h, y1))
            y2 = max(0, min(page_h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            recalled_region = {
                "type": "figure",
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": 0.42,
                "res": [],
                "img": image[y1:y2, x1:x2].copy(),
            }
            recalled.append(recalled_region)
            existing_figures.append(recalled_region["bbox"])

        if not recalled:
            return list(results)
        return list(results) + recalled

    @classmethod
    def _is_figure_caption_like(cls, region: dict) -> bool:
        type_name = str(region.get("type", "")).lower()
        if type_name not in {"figure_caption", "title", "text"}:
            return False
        text = re.sub(r"\s+", "", cls._extract_region_text(region))
        return bool(cls._FIGURE_CAPTION_RE.match(text))

    @classmethod
    def _has_paired_figure_above(
        cls,
        caption_box: List[float],
        figure_boxes: List[List[float]],
        page_w: int,
        page_h: int,
    ) -> bool:
        max_gap = max(80.0, page_h * 0.08)
        for figure_box in figure_boxes:
            if cls._axis_overlap_ratio(caption_box, figure_box, axis="x") < 0.25:
                continue
            vertical_gap = caption_box[1] - figure_box[3]
            contains_caption = cls._contain_ratio(caption_box, figure_box) >= 0.75
            if contains_caption:
                return True
            if -max(24.0, page_h * 0.012) <= vertical_gap <= max_gap:
                return True
        return False

    @classmethod
    def _find_caption_anchored_visual_region(
        cls,
        gray: np.ndarray,
        caption_box: List[float],
        page_w: int,
        page_h: int,
    ) -> Optional[List[float]]:
        caption_cx = (caption_box[0] + caption_box[2]) * 0.5
        caption_w = max(1.0, caption_box[2] - caption_box[0])
        search_half_w = max(page_w * 0.22, caption_w * 1.6, 180.0)
        search_h = max(page_h * 0.34, 520.0)
        x1 = int(max(0.0, caption_cx - search_half_w))
        x2 = int(min(float(page_w), caption_cx + search_half_w))
        y1 = int(max(0.0, caption_box[1] - search_h))
        y2 = int(max(0.0, caption_box[1] - 12.0))
        if x2 <= x1 or y2 <= y1:
            return None

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        mask = np.where(roi < 245, 255, 0).astype("uint8")
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_w = page_w * 0.18
        min_h = page_h * 0.10
        min_area = page_w * page_h * 0.025
        max_gap = max(120.0, page_h * 0.08)
        candidates: List[tuple[float, float, List[float]]] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            box = [
                float(x1 + x),
                float(y1 + y),
                float(x1 + x + width),
                float(y1 + y + height),
            ]
            area = float(width * height)
            gap = caption_box[1] - box[3]
            center_delta = abs(((box[0] + box[2]) * 0.5) - caption_cx)
            if width < min_w or height < min_h or area < min_area:
                continue
            if gap < -max(16.0, page_h * 0.008) or gap > max_gap:
                continue
            if center_delta > max(width * 0.58, page_w * 0.16):
                continue
            candidates.append((max(0.0, gap), -area, box))

        if not candidates:
            return None
        candidates.sort()
        return candidates[0][2]

    @classmethod
    def _dedup_sort_key(cls, item: dict) -> tuple[float, ...]:
        """Category-aware preference for nested-duplicate resolution.

        For text-like regions, detection confidence should dominate over raw
        bbox area / concatenated text length. This prevents a single low-score
        merged text block from suppressing multiple higher-confidence fine-grain
        text blocks, which became more common after lowering layout thresholds.
        For visual regions we still prefer larger area first.
        """
        score = float(item.get("score", 0.0) or 0.0)
        text_len = float(len(item.get("text", "")))
        area = float(item.get("area", 0.0) or 0.0)
        index_bias = -float(item.get("index", 0))
        category = str(item.get("category", ""))
        raw_type = str(item.get("raw_type", ""))
        if cls._is_semantic_container_item(item):
            return (3.0, score, area, text_len, index_bias)
        if cls._is_generic_parent_item(item):
            # Coarse V3 parent regions are useful only when no finer children
            # exist, so let specific child boxes win containment decisions.
            return (0.25, score, text_len, area, index_bias)
        if category in cls._TEXTLIKE_DEDUP_CATEGORIES:
            text = str(item.get("text", "") or "").strip()
            if raw_type in cls._TITLE_LIKE_RAW_TYPES:
                return (2.4, score, text_len, area, index_bias)
            if category == "title" and text.endswith(("。", "！", "？", ".", "!", "?")):
                return (1.95, score, text_len, area, index_bias)
            return (2.0, score, text_len, area, index_bias)
        if category == "formula":
            return (1.0, score, area, text_len, index_bias)
        return (1.0, area, score, text_len, index_bias)

    @classmethod
    def _annotate_contained_child_counts(cls, items: List[dict]) -> None:
        for parent in items:
            count = 0
            specific_count = 0
            for child in items:
                if child is parent:
                    continue
                if cls._is_containment_pair(parent, child):
                    count += 1
                    if cls._is_specific_child_item(child):
                        specific_count += 1
            parent["contained_child_count"] = count
            parent["contained_specific_child_count"] = specific_count

    @classmethod
    def _is_containment_pair(
        cls,
        parent: dict,
        child: dict,
        *,
        child_cover_threshold: float = 0.85,
        area_ratio_threshold: float = 1.5,
    ) -> bool:
        parent_area = max(1.0, float(parent.get("area", 0.0) or 0.0))
        child_area = max(1.0, float(child.get("area", 0.0) or 0.0))
        if parent_area <= child_area * area_ratio_threshold:
            return False
        return cls._contain_ratio(child["bbox"], parent["bbox"]) >= child_cover_threshold

    @classmethod
    def _is_semantic_container_item(cls, item: dict) -> bool:
        raw_type = str(item.get("raw_type", ""))
        category = str(item.get("category", ""))
        if raw_type in cls._TOP_LEVEL_PRESERVE_RAW_TYPES:
            return False
        return (
            raw_type in cls._SEMANTIC_CONTAINER_RAW_TYPES
            or category in cls._SEMANTIC_CONTAINER_CATEGORIES
        )

    @classmethod
    def _is_generic_parent_item(cls, item: dict) -> bool:
        raw_type = str(item.get("raw_type", ""))
        category = str(item.get("category", ""))
        if raw_type in cls._TOP_LEVEL_PRESERVE_RAW_TYPES:
            return False
        if raw_type == "content":
            return True
        if raw_type in {"text", "aside_text", "vertical_text", "algorithm"}:
            return int(item.get("contained_specific_child_count", 0) or 0) >= 2
        return (
            category in cls._GENERIC_PARENT_CATEGORIES
            and int(item.get("contained_specific_child_count", 0) or 0) >= 2
        )

    @classmethod
    def _is_specific_child_item(cls, item: dict) -> bool:
        raw_type = str(item.get("raw_type", ""))
        category = str(item.get("category", ""))
        if raw_type in cls._TOP_LEVEL_PRESERVE_RAW_TYPES or raw_type in cls._FORMULA_RAW_TYPES:
            return True
        return category in {
            "text",
            "title",
            "code",
            "formula",
            "figure_caption",
            "table_caption",
            "table_footnote",
            "footnote",
            "page_number",
            "reference",
            "table",
            "figure",
        }

    @classmethod
    def _attach_nested_child(cls, parent_region: dict, child_region: dict, *, reason: str) -> None:
        attributes = parent_region.setdefault("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}
            parent_region["attributes"] = attributes
        children = attributes.setdefault("nested_children", [])
        if not isinstance(children, list):
            children = []
            attributes["nested_children"] = children
        child_type = str(child_region.get("type", ""))
        child_summary: Dict[str, Any] = {
            "type": child_type,
            "category": cls._CATEGORY_MAP.get(child_type.lower(), child_type.lower()),
            "bbox": [float(v) for v in cls._safe_bbox(child_region.get("bbox"))],
            "score": float(child_region.get("score", 0.0) or 0.0),
            "reason": reason,
        }
        text = cls._normalize_text(cls._extract_region_text(child_region))
        if text:
            child_summary["text"] = text[:240]
        for key in ("model_order", "raw_type", "layout_model"):
            if child_region.get(key) is not None:
                child_summary[key] = child_region.get(key)
        children.append(child_summary)
        attributes["nested_child_count"] = len(children)

    @classmethod
    def _trim_carry_over_text_regions(
        cls,
        results: List[dict],
    ) -> tuple[List[dict], List[Dict[str, Any]]]:
        if len(results) < 2:
            return results, []

        entries = [
            {
                "order": index,
                "region": cls._clone_region(region),
            }
            for index, region in enumerate(results)
        ]
        report: List[Dict[str, Any]] = []

        spatial_entries = sorted(
            entries,
            key=lambda item: (
                cls._safe_bbox(item["region"].get("bbox"))[1],
                cls._safe_bbox(item["region"].get("bbox"))[0],
                item["order"],
            ),
        )

        for entry_index, entry in enumerate(spatial_entries):
            current = entry["region"]
            best_shared = 0
            for previous_entry in reversed(spatial_entries[:entry_index]):
                previous_region = previous_entry["region"]
                if previous_region is None:
                    continue
                shared_lines = cls._shared_boundary_line_count(previous_region, current)
                if shared_lines > best_shared:
                    best_shared = shared_lines
                if best_shared > 0:
                    break
            if best_shared > 0:
                current = cls._trim_region_prefix_lines(current, best_shared)
                entry["region"] = current
                report.append(
                    {
                        "reason": "text_carry_over_trim",
                        "removed_lines": best_shared,
                    }
                )
                if current is None:
                    continue

        trimmed = [
            item["region"]
            for item in sorted(entries, key=lambda item: item["order"])
            if item["region"] is not None
        ]
        return trimmed, report

    def _is_nested_duplicate(self, candidate: dict, existing: dict) -> Optional[str]:
        if self._is_containment_pair(existing, candidate):
            if (
                self._is_semantic_container_item(existing)
                and not self._is_generic_parent_item(existing)
                and candidate["raw_type"] not in self._TOP_LEVEL_PRESERVE_RAW_TYPES
            ):
                if existing["category"] == "table":
                    return "table_container_child"
                if existing["category"] in {"header", "footer"} or existing["raw_type"] in {"header", "footer"}:
                    return "page_strip_container_child"
                return "semantic_container_child"
            if self._is_generic_parent_item(existing):
                return None

        if self._is_containment_pair(candidate, existing):
            if self._is_generic_parent_item(candidate):
                return "generic_parent_suppressed"

        same_category = candidate["category"] == existing["category"]
        same_caption_family = (
            candidate["category"] in self._CAPTION_FAMILY
            and existing["category"] in self._CAPTION_FAMILY
        )
        if not same_category and not same_caption_family:
            candidate_is_visual = candidate["category"] in {"figure", "formula", "table"}
            existing_is_textlike = existing["category"] in self._TEXTLIKE_DEDUP_CATEGORIES
            if candidate_is_visual and existing_is_textlike and existing["text"]:
                overlap_ratio = self._overlap_ratio(candidate["bbox"], existing["bbox"])
                contain_ratio = self._contain_ratio(existing["bbox"], candidate["bbox"])
                lower_confidence = candidate["score"] + 0.08 < existing["score"]
                text_explains_visual = (
                    overlap_ratio >= 0.92
                    or contain_ratio >= 0.92
                    or (
                        candidate["category"] == "formula"
                        and overlap_ratio >= 0.72
                        and candidate["area"] <= existing["area"] * 1.35
                    )
                )
                if lower_confidence and text_explains_visual:
                    return "cross_category_visual_text_duplicate"
            # 跨类别：检查候选文本是否被已有文本完全包含（视觉重复）。
            # 典型场景：低分 title 检测出 "瓦的北红海省博物馆。"，但高分 text
            # 已包含完整段落 "瓦的北红海省博物馆。博物馆二层陈列着..."
            if candidate["text"] and existing["text"]:
                if (
                    existing["category"] in {"figure", "table"}
                    and candidate["category"] in self._TEXTLIKE_DEDUP_CATEGORIES
                ):
                    return None
                short_text, long_text = sorted(
                    (candidate["text"], existing["text"]), key=len,
                )
                overlap_ratio = self._overlap_ratio(candidate["bbox"], existing["bbox"])
                contain_ratio = self._contain_ratio(candidate["bbox"], existing["bbox"])
                candidate_is_visual = candidate["category"] in {"figure", "formula", "table"}
                # 空间判定放宽：允许 bbox 邻近（adjacent），不一定严格重叠。
                # 典型场景：低分 title 的 bbox=[1153,776,1329,797] 与高分 text 的
                # bbox=[1154,800,1507,990] 垂直相接但不重叠，实际是同一段 OCR 内容的
                # 不同检测粒度。
                spatial_match = (
                    overlap_ratio >= 0.85
                    or contain_ratio >= 0.85
                    or (
                        not candidate_is_visual
                        and self._adjacent_ratio(candidate["bbox"], existing["bbox"]) >= 0.78
                    )
                )
                if len(short_text) >= 4 and short_text in long_text and spatial_match:
                    return "cross_category_text_duplicate"
                if (
                    candidate["category"] in self._TEXTLIKE_DEDUP_CATEGORIES
                    and existing["category"] in self._TEXTLIKE_DEDUP_CATEGORIES
                    and len(short_text) >= 8
                    and spatial_match
                    and SequenceMatcher(None, candidate["text"], existing["text"]).ratio() >= 0.70
                ):
                    return "cross_category_similar_text_duplicate"
            return None

        cand_box = candidate["bbox"]
        exist_box = existing["bbox"]
        contain_ratio = self._contain_ratio(cand_box, exist_box)
        overlap_ratio = self._overlap_ratio(cand_box, exist_box)
        cand_text = candidate["text"]
        exist_text = existing["text"]

        if candidate["category"] in {"figure", "table"} and same_category:
            if contain_ratio >= 0.95 or overlap_ratio >= 0.90:
                return "nested_visual_duplicate"
            return None

        if cand_text and exist_text:
            short_text, long_text = sorted((cand_text, exist_text), key=len)
            min_text_len = 1 if same_caption_family else 4
            # 空间判定同样允许邻近（adjacent）
            spatial_match = (
                contain_ratio >= 0.90
                or overlap_ratio >= 0.85
                or self._adjacent_ratio(cand_box, exist_box) >= 0.85
            )
            if (
                len(short_text) >= min_text_len
                and short_text in long_text
                and spatial_match
            ):
                return "nested_text_duplicate"
            if (
                same_category
                and spatial_match
                and min(len(cand_text), len(exist_text)) >= min_text_len
                and SequenceMatcher(None, cand_text, exist_text).ratio() >= 0.70
            ):
                return "similar_text_duplicate"

        return None

    @staticmethod
    def _extract_region_text(region: dict) -> str:
        res = region.get("res")
        if not isinstance(res, list):
            return ""
        parts: List[str] = []
        for item in res:
            if isinstance(item, dict):
                text = item.get("text", "")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                rhs = item[1]
                if isinstance(rhs, (list, tuple)) and rhs:
                    text = rhs[0]
                else:
                    text = rhs
            else:
                text = ""
            if text:
                parts.append(str(text))
        return "\n".join(parts)

    @classmethod
    def _extract_region_line_texts(cls, region: dict) -> List[str]:
        res = region.get("res")
        if not isinstance(res, list):
            return []
        lines: List[str] = []
        for item in res:
            if isinstance(item, dict):
                text = item.get("text", "")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                rhs = item[1]
                if isinstance(rhs, (list, tuple)) and rhs:
                    text = rhs[0]
                else:
                    text = rhs
            else:
                text = ""
            normalized = cls._normalize_text(text)
            if normalized:
                lines.append(str(text))
        return lines

    @classmethod
    def _shared_boundary_line_count(cls, previous: dict, current: dict) -> int:
        prev_type = str(previous.get("type", "")).lower()
        curr_type = str(current.get("type", "")).lower()
        if prev_type not in cls._TEXT_TYPES or curr_type not in cls._TEXT_TYPES:
            return 0

        prev_bbox = cls._safe_bbox(previous.get("bbox"))
        curr_bbox = cls._safe_bbox(current.get("bbox"))
        # 仅在两个文本块几乎位于同一列时，才判断“首行串带”。
        if cls._axis_overlap_ratio(prev_bbox, curr_bbox, axis="x") < 0.88:
            return 0
        # 放宽过头会误伤相邻段落，因此将容忍间距收紧到更保守的范围。
        if cls._vertical_gap(prev_bbox, curr_bbox) > 20.0:
            return 0

        prev_lines = cls._extract_region_line_texts(previous)
        curr_lines = cls._extract_region_line_texts(current)
        if not prev_lines or not curr_lines:
            return 0

        max_shared = min(3, len(prev_lines), len(curr_lines))
        best = 0
        for shared in range(1, max_shared + 1):
            prev_slice = prev_lines[-shared:]
            curr_slice = curr_lines[:shared]
            if all(
                cls._normalize_text(left) == cls._normalize_text(right)
                and cls._normalize_text(left)
                for left, right in zip(prev_slice, curr_slice)
            ):
                best = shared
        # 单行重复在报纸/多栏场景中误伤率较高，这里只接受至少 2 行的
        # 严格边界重复，优先保证段首完整保留。
        return best if best >= 2 else 0

    @staticmethod
    def _clone_region(region: dict) -> dict:
        cloned = dict(region)
        if isinstance(region.get("res"), list):
            cloned["res"] = list(region["res"])
        return cloned

    @classmethod
    def _trim_region_prefix_lines(cls, region: dict, count: int) -> Optional[dict]:
        res = region.get("res")
        if not isinstance(res, list):
            return region
        remaining = list(res[count:])
        if not remaining:
            return None
        region["res"] = remaining
        bbox = cls._bbox_from_region_res(remaining)
        if bbox is not None:
            region["bbox"] = bbox
        return region

    @classmethod
    def _trim_title_leading_formula_number(cls, region: dict) -> dict:
        if str(region.get("type", "")).lower() != "title":
            return region
        res = region.get("res")
        if not isinstance(res, list) or len(res) < 2:
            return region

        first = res[0]
        first_text = cls._normalize_text(cls._extract_item_text(first))
        if re.fullmatch(r"\(?\d{1,3}\)?", first_text or "") is None:
            return region

        first_bbox = cls._bbox_from_single_region_res(first)
        rest_bbox = cls._bbox_from_region_res(res[1:])
        if first_bbox is None or rest_bbox is None:
            return region
        page_like_gap = rest_bbox[0] - first_bbox[2]
        first_height = max(1.0, first_bbox[3] - first_bbox[1])
        if page_like_gap < max(18.0, first_height * 0.8):
            return region

        trimmed = cls._clone_region(region)
        trimmed["res"] = list(res[1:])
        trimmed["bbox"] = rest_bbox
        return trimmed

    @staticmethod
    def _extract_item_text(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("text", ""))
        if isinstance(item, (list, tuple)) and len(item) == 2:
            rhs = item[1]
            if isinstance(rhs, (list, tuple)) and rhs:
                return str(rhs[0])
            return str(rhs)
        return ""

    @classmethod
    def _bbox_from_single_region_res(cls, item: Any) -> Optional[List[float]]:
        return cls._bbox_from_region_res([item])

    @staticmethod
    def _bbox_from_region_res(res: List[Any]) -> Optional[List[float]]:
        polys: List[List[List[float]]] = []
        for item in res:
            if isinstance(item, dict):
                poly = item.get("text_region")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                poly = item[0]
                if hasattr(poly, "tolist"):
                    poly = poly.tolist()
            else:
                poly = None
            if isinstance(poly, list) and poly:
                polys.append(poly)
        if not polys:
            return None

        xs = [float(pt[0]) for poly in polys for pt in poly]
        ys = [float(pt[1]) for poly in polys for pt in poly]
        if not xs or not ys:
            return None
        return [min(xs), min(ys), max(xs), max(ys)]

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = str(text or "")
        text = text.replace("\n", " ")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _axis_overlap_ratio(b1: List[float], b2: List[float], axis: str = "x") -> float:
        if axis == "x":
            left = max(b1[0], b2[0])
            right = min(b1[2], b2[2])
            span1 = max(1.0, b1[2] - b1[0])
            span2 = max(1.0, b2[2] - b2[0])
        else:
            left = max(b1[1], b2[1])
            right = min(b1[3], b2[3])
            span1 = max(1.0, b1[3] - b1[1])
            span2 = max(1.0, b2[3] - b2[1])
        overlap = max(0.0, right - left)
        return overlap / min(span1, span2)

    @staticmethod
    def _vertical_gap(b1: List[float], b2: List[float]) -> float:
        if b1[1] <= b2[3] and b2[1] <= b1[3]:
            return 0.0
        return min(abs(b1[1] - b2[3]), abs(b2[1] - b1[3]))

    @staticmethod
    def _safe_bbox(raw_bbox: Any) -> List[float]:
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            return [float(v) for v in raw_bbox]
        return [0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def _overlap_ratio(b1: List[float], b2: List[float]) -> float:
        inter_w = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        inter_h = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        inter_area = inter_w * inter_h
        area1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        area2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        return inter_area / min(area1, area2)

    @staticmethod
    def _contain_ratio(b1: List[float], b2: List[float]) -> float:
        # b1 相对于 b2 的“被包含度”
        inter_w = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        inter_h = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        inter_area = inter_w * inter_h
        area1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        return inter_area / area1

    @staticmethod
    def _adjacent_ratio(b1: List[float], b2: List[float]) -> float:
        """计算候选框相对于目标框的“邻近覆盖率”。

        与 overlap/contain 不同，此方法允许 bbox 相接但不重叠的情况，
        只要两者在至少一个轴上的投影完全或几乎对齐。
        """
        # X 轴投影重合度
        x_left = max(b1[0], b2[0])
        x_right = min(b1[2], b2[2])
        x_align = max(0.0, x_right - x_left) / max(min(b1[2] - b1[0], b2[2] - b2[0]), 1.0)

        # Y 轴：允许 b1 的底部与 b2 的顶部相接（或相反）
        y_gap = 0.0
        if b1[3] < b2[1]:
            y_gap = b2[1] - b1[3]
        elif b2[3] < b1[1]:
            y_gap = b1[1] - b2[3]
        # 否则垂直重叠

        min_height = min(max(b1[3] - b1[1], 1.0), max(b2[3] - b2[1], 1.0))
        if y_gap > min_height * 0.5:
            return 0.0  # 相距太远

        # 邻近度：gap 越小越接近
        proximity = max(0.0, 1.0 - y_gap / min_height)
        return x_align * proximity

    # ------------------------------------------------------------------
    # 逐区域转换
    # ------------------------------------------------------------------

    def _convert_region(
        self,
        region: dict,
        image_width: int,
        image_height: int,
        index: int = 0,
    ) -> Dict[str, Any]:
        """将单个 PaddleOCR 区域字典转换为 v2.0 区块字典。"""
        type_name = region.get("type", "text").lower()
        category = self._CATEGORY_MAP.get(type_name, type_name)
        bbox = region.get("bbox", [0, 0, image_width, image_height])
        confidence = float(region.get("score", 1.0))

        block: Dict[str, Any] = {
            "id": f"blk_{index}",
            "category": category,
            "bbox": [float(v) for v in bbox],
            "confidence": confidence,
            "order": index,
        }
        attributes: Dict[str, Any] = {}
        if isinstance(region.get("attributes"), dict):
            attributes.update(region["attributes"])
        for source_key, target_key in (
            ("raw_type", "raw_layout_label"),
            ("model_order", "model_order"),
            ("layout_model", "layout_model"),
        ):
            value = region.get(source_key)
            if value is not None:
                attributes[target_key] = value
        if attributes:
            block["attributes"] = attributes

        res = region.get("res")

        # -- 文本类型：将 OCR 结果转换为 text_lines ------------------
        if type_name in self._TEXT_TYPES:
            text_lines = self._convert_text_lines(res)
            block["text_lines"] = text_lines
            block["text"] = "\n".join(
                tl["text"] for tl in text_lines if tl.get("text")
            )
            text_bbox = self._bbox_from_text_lines(text_lines)
            if text_bbox is not None:
                block["bbox"] = self._union_bbox(block["bbox"], text_bbox)

        # -- 表格：提取 HTML --------------------------------------------
        elif type_name == "table":
            if isinstance(res, dict) and "html" in res:
                block["html"] = res["html"]

        # -- 公式：提取 LaTeX ----------------------------------------
        elif category == "formula":
            if isinstance(res, dict) and "latex" in res:
                block["latex"] = res["latex"]

        # -- 将 ROI 图像编码为 base64 PNG ---------------------------------
        roi_img = region.get("img")
        if roi_img is not None and isinstance(roi_img, np.ndarray):
            encoded = self._encode_image(roi_img)
            if encoded is not None:
                block["image_base64"] = encoded

        return block

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_text_lines(res: Any) -> List[Dict[str, Any]]:
        """将 OCR 结果转换为 text_lines 格式。

        支持两种格式：
        - 新格式：字典列表 ``{text, confidence, text_region}``
        - 旧格式：元组列表 ``(text_region, (text, conf))``
        """
        text_lines: List[Dict[str, Any]] = []
        if not isinstance(res, list):
            return text_lines

        for item in res:
            if isinstance(item, dict):
                line: Dict[str, Any] = {
                    "text": item.get("text", ""),
                    "confidence": float(item.get("confidence", 1.0)),
                }
                if "text_region" in item:
                    line["poly"] = item["text_region"]
                text_lines.append(line)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                # 旧版 PaddleOCR 格式: (text_region, (text, conf))
                region, text_conf = item
                text = text_conf[0] if isinstance(text_conf, (list, tuple)) else str(text_conf)
                conf = float(text_conf[1]) if isinstance(text_conf, (list, tuple)) and len(text_conf) > 1 else 1.0
                poly = region if isinstance(region, list) else region.tolist()
                text_lines.append({
                    "text": text,
                    "confidence": conf,
                    "poly": poly,
                })

        return text_lines

    @staticmethod
    def _bbox_from_text_lines(text_lines: List[Dict[str, Any]]) -> Optional[List[float]]:
        polys: List[List[List[float]]] = []
        for line in text_lines:
            poly = line.get("poly")
            if isinstance(poly, list) and poly:
                polys.append(poly)
        if not polys:
            return None

        xs = [float(pt[0]) for poly in polys for pt in poly if isinstance(pt, (list, tuple)) and len(pt) >= 2]
        ys = [float(pt[1]) for poly in polys for pt in poly if isinstance(pt, (list, tuple)) and len(pt) >= 2]
        if not xs or not ys:
            return None
        return [min(xs), min(ys), max(xs), max(ys)]

    @staticmethod
    def _union_bbox(b1: List[float], b2: List[float]) -> List[float]:
        return [
            min(float(b1[0]), float(b2[0])),
            min(float(b1[1]), float(b2[1])),
            max(float(b1[2]), float(b2[2])),
            max(float(b1[3]), float(b2[3])),
        ]

    @staticmethod
    def _encode_image(img: np.ndarray) -> Optional[str]:
        """将 numpy 图像数组编码为 base64 PNG 字符串。"""
        success, buf = cv2.imencode(".png", img)
        if success:
            return base64.b64encode(buf.tobytes()).decode("ascii")
        return None
