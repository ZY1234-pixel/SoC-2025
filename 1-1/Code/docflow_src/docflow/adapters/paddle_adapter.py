"""PaddleOCR / ppstructure 输出的适配器。

将 PaddleOCR 的版面分析与结构识别结果转换为 DocFlow v2.0 标准 JSON 格式。
"""

from __future__ import annotations

import base64
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

        for idx, region in enumerate(results):
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
                    "blocks": blocks,
                }
            ],
        }

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
