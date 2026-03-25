"""版面恢复主管线。

:class:`RecoveryPipeline` 串联 JSON 加载、校验、文档模型构建、
版面分析与渲染等步骤，实现端到端的版面恢复。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from docflow.config import RecoveryConfig
from docflow.schema.validator import validate_input, normalize_input
from docflow.model.base import Block, BlockType
from docflow.model.blocks.text_block import TextBlock, Paragraph
from docflow.model.blocks.factory import BlockFactory
from docflow.model.page import Document, Page
from docflow.model.zone import Zone
from docflow.layout.sorter import sort_layout
from docflow.layout.paragraph_detector import split_into_paragraphs
from docflow.layout.style_inferrer import infer_block_styles
from docflow.renderer.base import BaseRenderer
from docflow.renderer.docx_renderer import DocxRenderer
from docflow.renderer.markdown_renderer import MarkdownRenderer
from docflow.renderer.pdf_renderer import PdfRenderer
from docflow.model.blocks.image_block import ImageBlock
from docflow.model.blocks.equation_block import EquationBlock
from docflow.model.blocks.table_block import TableBlock

_ZONE_STRIP_TYPES = {
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
}
_MULTICOL_TAIL_ABSORB_TYPES = {
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.REFERENCE,
    BlockType.TABLE_CAPTION,
    BlockType.FIGURE_CAPTION,
    BlockType.FORMULA_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.FIGURE,
    BlockType.TABLE,
    BlockType.EQUATION,
}
_STRIP_TYPES = {
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
}
_TITLE_LEVEL_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:[\.、])?\s*\S")
_FOOTER_LIKE_RE = re.compile(r"(©|copyright|https?://|www\.|journal\.com|verlag|kgaa)", re.IGNORECASE)


def _infer_title_heading_level(text: str) -> Optional[int]:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return None
    match = _TITLE_LEVEL_RE.match(normalized)
    if not match:
        return None
    return match.group(1).count(".") + 1


class RecoveryPipeline:
    """版面恢复主编排器。

    用法::

        pipeline = RecoveryPipeline()
        pipeline.recover("input.json", "output.docx", format="docx")
    """

    # 默认渲染器注册表
    _DEFAULT_RENDERERS: Dict[str, Type[BaseRenderer]] = {
        "docx": DocxRenderer,
        "markdown": MarkdownRenderer,
        "md": MarkdownRenderer,
        "pdf": PdfRenderer,
    }

    def __init__(self, config: Optional[RecoveryConfig] = None) -> None:
        self.config = config or RecoveryConfig()
        self._renderers: Dict[str, BaseRenderer] = {}

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def recover(
        self,
        json_input: Union[str, dict, Path],
        output_path: str,
        format: str = "docx",
    ) -> str:
        """执行完整的版面恢复管线。

        参数
        ----------
        json_input:
            原始 JSON 字符串、字典或 JSON 文件路径。
        output_path:
            输出文件的目标路径。
        format:
            输出格式 ——  ``"docx"``、``"markdown"`` / ``"md"``、``"pdf"`` 之一。

        返回
        -------
        成功时返回 *output_path*。
        """
        # 1. 加载 JSON
        data = self._load_json(json_input)

        # 2. 校验并规范化
        is_valid, errors = validate_input(data)
        if not is_valid:
            raise ValueError(
                "Invalid input JSON:\n  " + "\n  ".join(errors)
            )
        data = normalize_input(data)

        # 3. 构建文档模型
        document = self._build_document(data)

        # 4. 渲染
        renderer = self._get_renderer(format)
        renderer.render(document, output_path)

        # 5. 返回输出路径
        return output_path

    # ------------------------------------------------------------------
    # 向后兼容 API
    # ------------------------------------------------------------------

    def run(self, json_input: Union[str, dict, Path], output_path: str, fmt: str = "docx") -> str:
        """向后兼容别名，等价于 :meth:`recover`。"""
        return self.recover(json_input, output_path, format=fmt)

    def run_from_dict(self, data: dict, output_path: str, fmt: str = "docx") -> str:
        """向后兼容别名：从 dict 输入恢复输出。"""
        return self.recover(data, output_path, format=fmt)

    def build_document(self, json_input: Union[str, dict, Path]) -> Document:
        """构建并返回内部 :class:`Document` 模型，不执行渲染。"""
        data = self._load_json(json_input)
        is_valid, errors = validate_input(data)
        if not is_valid:
            raise ValueError("Invalid input JSON:\n  " + "\n  ".join(errors))
        data = normalize_input(data)
        return self._build_document(data)

    # ------------------------------------------------------------------
    # JSON 加载
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(json_input: Union[str, dict, Path]) -> dict:
        """从字符串、字典或文件路径加载 JSON。

        - *dict*: 直接返回。
        - *str*: 若为已存在的文件路径则读取解析；否则按原始 JSON 解析。
        - *Path*: 读取并解析。
        """
        if isinstance(json_input, dict):
            return json_input

        if isinstance(json_input, Path):
            with open(json_input, "r", encoding="utf-8") as fh:
                return json.load(fh)

        # 字符串 -- 可能是文件路径或原始 JSON
        if isinstance(json_input, str):
            if os.path.isfile(json_input):
                with open(json_input, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            return json.loads(json_input)

        raise TypeError(
            f"json_input must be str, dict, or Path, got {type(json_input)}"
        )

    # ------------------------------------------------------------------
    # 文档模型构建
    # ------------------------------------------------------------------

    def _build_document(self, data: dict) -> Document:
        """从规范化的 JSON *data* 构建 :class:`Document` 对象。"""
        doc = Document(
            metadata=data.get("metadata") or {},
        )

        for page_data in data.get("pages", []):
            page = self._build_page(page_data)
            doc.pages.append(page)

        return doc

    def _build_page(self, page_data: dict) -> Page:
        """从页面字典构建单个 :class:`Page` 对象。"""
        page = Page(
            index=page_data["page_index"],
            image_width=page_data.get("width",
                                      page_data.get("image_width", 0)),
            image_height=page_data.get("height",
                                       page_data.get("image_height", 0)),
        )

        # 可选的图片来源
        if "image_path" in page_data:
            page.image_path = page_data["image_path"]
        if "image_base64" in page_data:
            page.image_base64 = page_data["image_base64"]
        if "style_defaults" in page_data and isinstance(page_data["style_defaults"], dict):
            page.style_defaults = dict(page_data["style_defaults"])
        if "attributes" in page_data and isinstance(page_data["attributes"], dict):
            page.attributes = dict(page_data["attributes"])
        if "relations" in page_data and isinstance(page_data["relations"], list):
            page.relations = list(page_data["relations"])

        # -- 根据宽高比检测页面尺寸（A4、Letter 等）----------
        page.detect_page_size()

        # -- 通过 BlockFactory 构建区块 ----------------------------------
        raw_blocks: List[dict] = page_data.get("blocks", [])
        blocks: List[Block] = [BlockFactory.create(bd) for bd in raw_blocks]

        # -- 纠正常见的分类错误 ------------------------------------------
        self._fix_block_categories(
            blocks,
            page_width=page.image_width,
            page_height=page.image_height,
        )

        # -- 从页面图片补充缺失的图像数据 -----------------------
        self._fill_missing_images(blocks, getattr(page, 'image_path', None))

        # -- 根据区块边界框估算页边距 ---------------------
        page.estimate_margins(blocks)

        # -- 需要时执行版面分析 ----------------------------------
        if self._needs_layout_analysis(raw_blocks) and len(blocks) > 1:
            blocks = sort_layout(
                blocks,
                page.image_width,
                image_height=page.image_height,
                max_cols=self.config.max_cols,
                cluster_thresh=self.config.column_cluster_thresh,
                column_confidence_min=self.config.column_confidence_min,
                zone_strip_height_ratio=self.config.zone_strip_height_ratio,
            )

        # -- 提升顶部作者署名/导语短行，避免误落入正文分栏 --------------------
        self._promote_top_byline_rows(
            blocks,
            page_width=page.image_width,
            page_height=page.image_height,
        )

        # -- 纠正 OCR/版面分析未识别出的局部并排图文带 ----------------------
        self._promote_side_by_side_hero_bands(
            blocks,
            page_width=page.image_width,
            page_height=page.image_height,
        )

        # -- 检测文本区块中的段落 ------------------------------
        for block in blocks:
            if isinstance(block, TextBlock) and block.lines:
                para_groups = split_into_paragraphs(
                    block.lines,
                    min_indent_px=self.config.paragraph_indent_px,
                    list_marker_enabled=self.config.paragraph_list_marker_enabled,
                )
                block.paragraphs = [
                    Paragraph(lines=group) for group in para_groups
                ]

        # -- 估算字号 --------------------------------------------
        mapper = page.coord_mapper
        for block in blocks:
            if isinstance(block, TextBlock):
                block.estimate_font_size(mapper)

        # -- 将区块分组到 Zone ----------------------------------------
        page.zones = self._blocks_to_zones(
            blocks,
            image_width=page.image_width,
            image_height=page.image_height,
        )

        # -- 样式推断（字号、对齐、行距、缩进、bold/italic 等）-----------
        # 仅填充 JSON 中未明确提供的字段，已有值不覆盖
        infer_block_styles(
            page.zones,
            mapper,
            justify_min_lines=self.config.align_justify_min_lines,
            page_width_px=page.image_width,
        )

        return page

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_block_categories(
        blocks: List[Block],
        page_width: int = 0,
        page_height: int = 0,
    ) -> None:
        """纠正常见的版面分析分类错误。

        例如 PaddleOCR 有时将表格标题（"TABLE I ..."）识别为 header。
        """
        section_title_re = re.compile(
            r"^\s*(\d+(?:\.\d+)*[\.、]|\d+[)）]|\(?\d+\)|[（(]?[一二三四五六七八九十百]+[)）\.、])\s*\S+"
        )
        for block in blocks:
            if not isinstance(block, TextBlock):
                continue
            text = block.full_text().strip()
            if not text:
                continue
            upper = text.upper()
            near_top = (
                page_height > 0
                and float(block.bbox.y1) <= max(float(page_height) * 0.16, 1.0)
            )
            if block.block_type == BlockType.HEADER:
                if re.match(r'TABLE\s', upper):
                    block.block_type = BlockType.TABLE_CAPTION
                elif re.match(r'FIG(URE|\.)\s', upper):
                    block.block_type = BlockType.FIGURE_CAPTION
                else:
                    is_numbered_section = bool(section_title_re.match(text))
                    shortish = len(text) <= 28
                    narrow = (
                        page_width <= 0
                        or float(block.bbox.width) <= max(float(page_width) * 0.42, 1.0)
                    )
                    if is_numbered_section and shortish and narrow and near_top:
                        block.block_type = BlockType.TITLE
            elif block.block_type == BlockType.FOOTER:
                footer_like = bool(_FOOTER_LIKE_RE.search(text))
                title_like = (
                    near_top
                    and not footer_like
                    and len(text) <= 120
                    and len(text.split()) >= 3
                    and float(block.bbox.height) >= max(float(page_height) * 0.015, 36.0)
                    and float(block.bbox.width) >= max(float(page_width) * 0.22, 1.0)
                )
                if title_like:
                    block.block_type = BlockType.TITLE
            elif block.block_type == BlockType.TEXT:
                near_bottom = (
                    page_height > 0
                    and float(block.bbox.y2) >= max(float(page_height) * 0.92, 1.0)
                )
                footer_like = bool(_FOOTER_LIKE_RE.search(text))
                if near_bottom and footer_like and len(text) <= 120:
                    block.block_type = BlockType.FOOTER

            if block.block_type == BlockType.TITLE:
                level = _infer_title_heading_level(text)
                if level is not None:
                    if block.attributes is None:
                        block.attributes = {}
                    block.attributes.setdefault("heading_level", level)

    @staticmethod
    def _promote_side_by_side_hero_bands(
        blocks: List[Block],
        page_width: int = 0,
        page_height: int = 0,
    ) -> None:
        """将被漏判的“左图右文/右图左文”图文带提升为局部双栏结构。

        仅处理当前仍是单栏的块，并要求一侧为较大的图像块，另一侧为与其
        垂直重叠的标题/正文块，避免误伤普通单栏页面。
        """
        if page_width <= 0 or len(blocks) < 2:
            return

        figure_like = (BlockType.FIGURE, BlockType.IMAGE if hasattr(BlockType, "IMAGE") else BlockType.FIGURE)
        candidates = sorted(
            [
                block for block in blocks
                if block.col_count == 1
                and block.block_type in figure_like
                and float(block.bbox.width) <= float(page_width) * 0.52
                and float(block.bbox.height) >= max(float(page_height) * 0.12, 140.0)
            ],
            key=lambda b: (b.bbox.y1, b.bbox.x1),
        )
        for anchor in candidates:
            overlapping: List[Block] = []
            side = None
            for block in blocks:
                if block is anchor or block.col_count != 1:
                    continue
                if block.block_type not in {BlockType.TITLE, BlockType.TEXT, BlockType.REFERENCE, BlockType.ABSTRACT}:
                    continue
                y_overlap = min(float(anchor.bbox.y2), float(block.bbox.y2)) - max(float(anchor.bbox.y1), float(block.bbox.y1))
                if y_overlap < 36.0:
                    continue
                gap_left = float(anchor.bbox.x1) - float(block.bbox.x2)
                gap_right = float(block.bbox.x1) - float(anchor.bbox.x2)
                if gap_right >= 24.0 and float(block.bbox.width) >= float(page_width) * 0.22:
                    candidate_side = "right"
                elif gap_left >= 24.0 and float(block.bbox.width) >= float(page_width) * 0.22:
                    candidate_side = "left"
                else:
                    continue
                if side is None:
                    side = candidate_side
                if candidate_side != side:
                    continue
                overlapping.append(block)

            if not overlapping or side is None:
                continue

            band_blocks = [anchor] + overlapping
            band_width = max(float(block.bbox.x2) for block in band_blocks) - min(float(block.bbox.x1) for block in band_blocks)
            if band_width < float(page_width) * 0.55:
                continue

            left_col_blocks = [anchor] if side == "right" else overlapping
            right_col_blocks = overlapping if side == "right" else [anchor]
            if not left_col_blocks or not right_col_blocks:
                continue

            for block in left_col_blocks:
                block.col_count = 2
                block.col_index = 0
                block.spanned_cols = [0]
            for block in right_col_blocks:
                block.col_count = 2
                block.col_index = 1
                block.spanned_cols = [1]

    @staticmethod
    def _promote_top_byline_rows(
        blocks: List[Block],
        page_width: int = 0,
        page_height: int = 0,
    ) -> None:
        """将顶部居中的署名短行从正文分栏中提升为单栏行。

        典型场景：报纸主标题下方的作者行、地点行、导语短行。
        """
        if page_width <= 0 or page_height <= 0:
            return
        page_center = float(page_width) * 0.5
        top_limit = float(page_height) * 0.22
        min_width = float(page_width) * 0.12
        max_width = float(page_width) * 0.32

        for block in blocks:
            if not isinstance(block, TextBlock):
                continue
            if block.block_type != BlockType.TEXT:
                continue
            if block.count_lines() != 1:
                continue
            text = block.full_text().strip()
            if not text or len(text) > 40:
                continue
            width = float(block.bbox.width)
            height = float(block.bbox.height)
            if width < min_width or width > max_width:
                continue
            if height > max(float(page_height) * 0.03, 34.0):
                continue
            if float(block.bbox.y1) > top_limit:
                continue
            center = (float(block.bbox.x1) + float(block.bbox.x2)) * 0.5
            if abs(center - page_center) > float(page_width) * 0.12:
                continue
            if block.attributes is None:
                block.attributes = {}
            block.attributes["is_byline_row"] = True
            if block.col_count > 1:
                block.col_count = 1
                block.col_index = 0
                block.spanned_cols = [0]

    @staticmethod
    def _fill_missing_images(blocks: List[Block], image_path: Optional[str]) -> None:
        """从页面图片中裁剪区域，为缺失 image_data 的区块补充图像数据。"""
        if not image_path or not os.path.isfile(image_path):
            return

        needs_fill = [
            b for b in blocks
            if isinstance(b, (ImageBlock, EquationBlock, TableBlock))
            and not b.image_data
        ]
        if not needs_fill:
            return

        try:
            from PIL import Image
            import io
            page_img = Image.open(image_path)
        except Exception:
            return

        for block in needs_fill:
            try:
                bbox = block.bbox
                crop = page_img.crop((
                    max(0, int(bbox.x1)),
                    max(0, int(bbox.y1)),
                    min(page_img.width, int(bbox.x2)),
                    min(page_img.height, int(bbox.y2)),
                ))
                buf = io.BytesIO()
                crop.save(buf, format='PNG')
                block.image_data = buf.getvalue()
            except Exception:
                pass

    @staticmethod
    def _needs_layout_analysis(raw_blocks: List[dict]) -> bool:
        """若 *raw_blocks* 中有区块缺少显式的列信息（即输入 JSON 中未提供
        ``col_count``），则返回 *True*。"""
        for bd in raw_blocks:
            if "col_count" not in bd:
                return True
        return False

    @staticmethod
    def _blocks_to_zones(
        blocks: List[Block],
        image_width: int = 0,
        image_height: int = 0,
    ) -> List[Zone]:
        """将连续的同 ``col_count`` 区块分组为 :class:`Zone` 对象。

        初始分组后，夹在多栏区域之间的微小单栏区域（≤ 1 个区块）
        会被吸收到前方的多栏区域中，以减少不必要的分节符。
        """
        if not blocks:
            return []

        def _is_top_strip_block(block: Block) -> bool:
            if block.block_type not in _STRIP_TYPES:
                return False
            if image_height <= 0:
                return True
            return float(block.bbox.y1) <= max(float(image_height) * 0.18, 1.0)

        def _is_bottom_strip_block(block: Block) -> bool:
            if block.block_type not in _STRIP_TYPES:
                return False
            if image_height <= 0:
                return True
            return float(block.bbox.y2) >= max(float(image_height) * 0.82, 1.0)

        core_blocks = list(blocks)
        prefix_strip_blocks: List[Block] = []
        while core_blocks and _is_top_strip_block(core_blocks[0]):
            prefix_strip_blocks.append(core_blocks.pop(0))

        suffix_strip_blocks: List[Block] = []
        while core_blocks and _is_bottom_strip_block(core_blocks[-1]):
            suffix_strip_blocks.insert(0, core_blocks.pop())

        blocks = core_blocks
        if not blocks:
            strip_blocks = prefix_strip_blocks + suffix_strip_blocks
            if not strip_blocks:
                return []
            return [Zone(col_count=1, blocks=sorted(strip_blocks, key=lambda b: b.bbox.x1), has_spanned=False)]

        zones: List[Zone] = []
        current_blocks: List[Block] = [blocks[0]]
        current_col_count: int = blocks[0].col_count

        for block in blocks[1:]:
            if block.col_count == current_col_count:
                current_blocks.append(block)
            else:
                has_spanned = any(
                    len(b.spanned_cols) > 1 for b in current_blocks
                )
                zones.append(Zone(
                    col_count=current_col_count,
                    blocks=current_blocks,
                    has_spanned=has_spanned,
                ))
                current_blocks = [block]
                current_col_count = block.col_count

        # 输出最后一组
        has_spanned = any(len(b.spanned_cols) > 1 for b in current_blocks)
        zones.append(Zone(
            col_count=current_col_count,
            blocks=current_blocks,
            has_spanned=has_spanned,
        ))

        # 后处理：将微小单栏区域吸收到相邻的多栏区域
        if image_width > 0 and len(zones) >= 2:
            merged: List[Zone] = []
            idx = 0
            while idx < len(zones):
                zone = zones[idx]
                if zone.col_count == 1 and merged and merged[-1].col_count > 1:
                    target = merged[-1]
                    movable: List[Block] = []
                    remain: List[Block] = []
                    target_bottom = max((b.bbox.y2 for b in target.blocks), default=0.0)
                    col_w = image_width / max(target.col_count, 1)

                    for b in zone.blocks:
                        near_prev_zone = (float(b.bbox.y1) - float(target_bottom)) <= 48.0
                        narrow_enough = float(b.bbox.width) <= col_w * 0.92
                        absorbable = (
                            len(zone.blocks) <= 2
                            and b.block_type in _MULTICOL_TAIL_ABSORB_TYPES
                            and b.block_type not in _ZONE_STRIP_TYPES
                            and not bool(getattr(b, "attributes", {}) and b.attributes.get("is_byline_row"))
                            and near_prev_zone
                            and narrow_enough
                        )
                        if absorbable:
                            movable.append(b)
                        else:
                            remain.append(b)

                    if movable:
                        for b in movable:
                            b.col_count = target.col_count
                            cx = (b.bbox.x1 + b.bbox.x2) / 2
                            b.col_index = min(int(cx / col_w), target.col_count - 1)
                            b.spanned_cols = [b.col_index]
                        target.blocks.extend(movable)
                        target.has_spanned = target.has_spanned or any(
                            len(getattr(b, "spanned_cols", [])) > 1 for b in movable
                        )
                        target.blocks.sort(key=lambda b: (b.col_index, b.bbox.y1))

                    if not movable and idx + 1 < len(zones) and zones[idx + 1].col_count > 1:
                        next_target = zones[idx + 1]
                        next_movable: List[Block] = []
                        next_remain: List[Block] = []
                        target_top = min((b.bbox.y1 for b in next_target.blocks), default=float("inf"))
                        col_w = image_width / max(next_target.col_count, 1)

                        for b in remain:
                            near_next_zone = (float(target_top) - float(b.bbox.y2)) <= 72.0
                            narrow_enough = float(b.bbox.width) <= col_w * 0.92
                            absorbable = (
                                len(zone.blocks) <= 4
                                and b.block_type in _MULTICOL_TAIL_ABSORB_TYPES
                                and b.block_type not in _ZONE_STRIP_TYPES
                                and not bool(getattr(b, "attributes", {}) and b.attributes.get("is_byline_row"))
                                and near_next_zone
                                and narrow_enough
                            )
                            if absorbable:
                                next_movable.append(b)
                            else:
                                next_remain.append(b)

                        if next_movable:
                            for b in next_movable:
                                b.col_count = next_target.col_count
                                cx = (b.bbox.x1 + b.bbox.x2) / 2
                                b.col_index = min(int(cx / col_w), next_target.col_count - 1)
                                b.spanned_cols = [b.col_index]
                            next_target.blocks.extend(next_movable)
                            next_target.blocks.sort(key=lambda b: (b.col_index, b.bbox.y1))
                            remain = next_remain

                    if remain:
                        has_spanned = any(len(b.spanned_cols) > 1 for b in remain)
                        merged.append(Zone(
                            col_count=1,
                            blocks=remain,
                            has_spanned=has_spanned,
                        ))
                    idx += 1
                    continue

                if zone.col_count == 1 and idx + 1 < len(zones) and zones[idx + 1].col_count > 1:
                    target = zones[idx + 1]
                    movable = []
                    remain = []
                    target_top = min((b.bbox.y1 for b in target.blocks), default=float("inf"))
                    col_w = image_width / max(target.col_count, 1)

                    for b in zone.blocks:
                        near_next_zone = (float(target_top) - float(b.bbox.y2)) <= 72.0
                        narrow_enough = float(b.bbox.width) <= col_w * 0.92
                        absorbable = (
                            len(zone.blocks) <= 4
                            and b.block_type in _MULTICOL_TAIL_ABSORB_TYPES
                            and b.block_type not in _ZONE_STRIP_TYPES
                            and not bool(getattr(b, "attributes", {}) and b.attributes.get("is_byline_row"))
                            and near_next_zone
                            and narrow_enough
                        )
                        if absorbable:
                            movable.append(b)
                        else:
                            remain.append(b)

                    if movable:
                        for b in movable:
                            b.col_count = target.col_count
                            cx = (b.bbox.x1 + b.bbox.x2) / 2
                            b.col_index = min(int(cx / col_w), target.col_count - 1)
                            b.spanned_cols = [b.col_index]
                        target.blocks.extend(movable)
                        target.blocks.sort(key=lambda b: (b.col_index, b.bbox.y1))
                    if remain:
                        merged.append(Zone(
                            col_count=1,
                            blocks=remain,
                            has_spanned=any(len(b.spanned_cols) > 1 for b in remain),
                        ))
                    idx += 1
                    continue

                merged.append(zone)
                idx += 1
            zones = merged

        # 合并连续同 col_count 的区域（吸收操作可能产生相邻同列数区域）
        if len(zones) >= 2:
            consolidated: List[Zone] = [zones[0]]
            for zone in zones[1:]:
                prev = consolidated[-1]
                if (
                    zone.col_count == prev.col_count
                    and zone.rendering_strategy != "strip_row"
                    and prev.rendering_strategy != "strip_row"
                ):
                    prev.blocks.extend(zone.blocks)
                    prev.has_spanned = prev.has_spanned or zone.has_spanned
                    # 重排序，保证 col-0 在 col-1 之前
                    prev.blocks.sort(key=lambda b: (b.col_index, b.bbox.y1))
                else:
                    consolidated.append(zone)
            zones = consolidated

        if prefix_strip_blocks:
            zones.insert(0, Zone(
                col_count=1,
                blocks=sorted(prefix_strip_blocks, key=lambda b: b.bbox.x1),
                has_spanned=False,
            ))
        if suffix_strip_blocks:
            zones.append(Zone(
                col_count=1,
                blocks=sorted(suffix_strip_blocks, key=lambda b: b.bbox.x1),
                has_spanned=False,
            ))

        return zones

    def _get_renderer(self, format: str) -> BaseRenderer:
        """返回指定 *format* 的缓存渲染器实例。"""
        fmt = format.lower()
        if fmt not in self._renderers:
            cls = self._DEFAULT_RENDERERS.get(fmt)
            if cls is None:
                raise ValueError(
                    f"Unknown output format: {format!r}.  "
                    f"Supported: {', '.join(self._DEFAULT_RENDERERS)}"
                )
            self._renderers[fmt] = cls(config=self.config)
        return self._renderers[fmt]
