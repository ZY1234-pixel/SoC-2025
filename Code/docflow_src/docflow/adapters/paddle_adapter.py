"""PaddleOCR / ppstructure 输出的适配器。

将 PaddleOCR 的版面分析与结构识别结果转换为 DocFlow v2.0 标准 JSON 格式。
"""

from __future__ import annotations

import base64
import re
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
    })
    _DEDUP_CATEGORIES = frozenset({
        "text",
        "title",
        "reference",
        "figure",
        "table",
        "figure_caption",
        "table_caption",
        "table_footnote",
        "formula_caption",
    })
    _CAPTION_FAMILY = frozenset({
        "figure_caption",
        "table_caption",
        "formula_caption",
        "table_footnote",
    })

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
        filtered_results, cleanup_report = self._suppress_nested_duplicates(results)
        filtered_results, trim_report = self._trim_carry_over_text_regions(filtered_results)
        cleanup_report.extend(trim_report)
        page_attributes = None
        if cleanup_report:
            page_attributes = {
                "cleanup_removed_count": len(cleanup_report),
                "cleanup_rule_counts": dict(Counter(item["reason"] for item in cleanup_report)),
            }

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
                    "category": mapped,
                    "bbox": bbox,
                    "text": text,
                    "score": score,
                    "area": area,
                }
            )

        # 同类中优先保留信息量更大、得分更高的块
        ranked = sorted(
            enriched,
            key=lambda item: (
                len(item["text"]),
                item["area"],
                item["score"],
                -item["index"],
            ),
            reverse=True,
        )

        kept: List[dict] = []
        report: List[dict] = []
        for candidate in ranked:
            if candidate["category"] not in self._DEDUP_CATEGORIES:
                kept.append(candidate)
                continue
            duplicate = False
            duplicate_reason = None
            for existing in kept:
                duplicate_reason = self._is_nested_duplicate(candidate, existing)
                if duplicate_reason is not None:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
            else:
                report.append(
                    {
                        "reason": duplicate_reason,
                        "removed_index": candidate["index"],
                        "removed_category": candidate["category"],
                    }
                )

        kept.sort(key=lambda item: item["index"])
        return [item["region"] for item in kept], report

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
        same_category = candidate["category"] == existing["category"]
        same_caption_family = (
            candidate["category"] in self._CAPTION_FAMILY
            and existing["category"] in self._CAPTION_FAMILY
        )
        if not same_category and not same_caption_family:
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
            if (
                len(short_text) >= min_text_len
                and short_text in long_text
                and (contain_ratio >= 0.90 or overlap_ratio >= 0.85)
            ):
                return "nested_text_duplicate"

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
        if cls._axis_overlap_ratio(prev_bbox, curr_bbox, axis="x") < 0.75:
            return 0
        if cls._vertical_gap(prev_bbox, curr_bbox) > 36.0:
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
        if best == 0 and curr_lines:
            first_curr = cls._normalize_text(curr_lines[0])
            prev_tail = [cls._normalize_text(text) for text in prev_lines[-2:]]
            if first_curr and len(first_curr) >= 4:
                for tail in prev_tail:
                    if first_curr in tail or tail.endswith(first_curr):
                        best = 1
                        break
        return best

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

        res = region.get("res")

        # -- 文本类型：将 OCR 结果转换为 text_lines ------------------
        if type_name in self._TEXT_TYPES:
            text_lines = self._convert_text_lines(res)
            block["text_lines"] = text_lines
            block["text"] = "\n".join(
                tl["text"] for tl in text_lines if tl.get("text")
            )

        # -- 表格：提取 HTML --------------------------------------------
        elif type_name == "table":
            if isinstance(res, dict) and "html" in res:
                block["html"] = res["html"]

        # -- 公式：提取 LaTeX ----------------------------------------
        elif type_name == "equation":
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
    def _encode_image(img: np.ndarray) -> Optional[str]:
        """将 numpy 图像数组编码为 base64 PNG 字符串。"""
        success, buf = cv2.imencode(".png", img)
        if success:
            return base64.b64encode(buf.tobytes()).decode("ascii")
        return None
