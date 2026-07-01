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
from docflow.layout.column_detector import detect_columns, detect_spanned_blocks
from docflow.layout.paragraph_detector import split_into_paragraphs
from docflow.layout.style_inferrer import infer_block_styles
from docflow.layout.color_inferrer import infer_text_colors
from docflow.layout.font_classifier import FontClassifier
from docflow.renderer.base import BaseRenderer
from docflow.renderer.docx_renderer import DocxRenderer
from docflow.renderer.markdown_renderer import MarkdownRenderer
from docflow.renderer.pdf_renderer import PdfRenderer
from docflow.model.blocks.image_block import ImageBlock
from docflow.model.blocks.equation_block import EquationBlock
from docflow.model.blocks.table_block import TableBlock
from docflow.utils.constants import FIGURE_TYPES

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
_TITLE_LEVEL_RE = re.compile(
    r"^\s*(?:(\d+(?:\.\d+)*)(?:[\.、])?|[（(]?([一二三四五六七八九十百]+)[)）\.、])\s*\S"
)
_FOOTER_LIKE_RE = re.compile(r"(©|copyright|https?://|www\.|journal\.com|verlag|kgaa)", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(r"^\s*(?:图|fig(?:ure)?\.?)\s*[\d一二三四五六七八九十]+(?:[-－—]\d+)?\s*\S*", re.IGNORECASE)
_TABLE_CAPTION_RE = re.compile(r"^\s*(?:表|table)\s*[\d一二三四五六七八九十]+(?:[-－—]\d+)?\s*\S*", re.IGNORECASE)
_PAGE_NUMBER_RE = re.compile(r"^\s*[-—–]?\s*\d{1,4}\s*[-—–]?\s*$")
_ACADEMIC_FOOTER_NOTE_RE = re.compile(
    r"\b(?:correspondence\s+to|received\s+\d{1,2}\s+\w+\s+\d{4}|accepted\s+\d{1,2}\s+\w+\s+\d{4})\b",
    re.IGNORECASE,
)
_FORMULA_NUMBER_TEXT_RE = re.compile(r"^\s*\(?\s*\d{1,3}[a-zA-Z]?\s*\)?\s*$")


def _infer_title_heading_level(text: str) -> Optional[int]:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return None
    match = _TITLE_LEVEL_RE.match(normalized)
    if not match:
        return None
    numeric = match.group(1)
    if numeric:
        return numeric.count(".") + 1
    return 1


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
        self._font_classifier: Optional[FontClassifier] = None

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
        category_fix_count = self._fix_block_categories(
            blocks,
            page_width=page.image_width,
            page_height=page.image_height,
        )
        pre_spurious_visual_suppressed, blocks = self._suppress_spurious_visual_blocks_over_text(
            blocks,
            page_width=page.image_width,
            page_height=page.image_height,
        )

        # -- 从页面图片补充缺失的图像数据 -----------------------
        self._fill_missing_images(blocks, getattr(page, 'image_path', None))

        # -- 根据区块边界框估算页边距 ---------------------
        page.estimate_margins(blocks)

        # -- 需要时执行版面分析 ----------------------------------
        used_model_order = self._should_use_model_order(raw_blocks)
        model_order_repaired = 0
        if used_model_order:
            blocks = self._apply_model_order_metadata(blocks)
            self._assign_model_order_columns(
                blocks,
                page_width=page.image_width,
            )
            if bool(getattr(self.config, "model_order_geometric_repair_enabled", False)):
                model_order_repaired, blocks = self._repair_anomalous_model_order(
                    blocks,
                    page_width=page.image_width,
                    page_height=page.image_height,
                )
        elif self._needs_layout_analysis(raw_blocks) and len(blocks) > 1:
            blocks = sort_layout(
                blocks,
                page.image_width,
                image_height=page.image_height,
                max_cols=self.config.max_cols,
                cluster_thresh=self.config.column_cluster_thresh,
                column_confidence_min=self.config.column_confidence_min,
                zone_strip_height_ratio=self.config.zone_strip_height_ratio,
                strategy=self.config.reading_order_strategy,
                xycutpp_beta=self.config.xycutpp_beta,
                xycutpp_density_threshold=self.config.xycutpp_density_threshold,
                xycutpp_min_gap_ratio=self.config.xycutpp_min_gap_ratio,
                xycutpp_title_width_ratio=self.config.xycutpp_title_width_ratio,
            )

        # -- 提升顶部作者署名/导语短行，避免误落入正文分栏 --------------------
        byline_promotions = self._promote_top_byline_rows(
            blocks,
            page_width=page.image_width,
            page_height=page.image_height,
        )

        # -- 纠正 OCR/版面分析未识别出的局部并排图文带 ----------------------
        hero_band_promotions = self._promote_side_by_side_hero_bands(
            blocks,
            page_width=page.image_width,
            page_height=page.image_height,
        )

        decorative_icon_suppressed, blocks = self._suppress_decorative_title_icons(
            blocks,
            page_width=page.image_width,
            page_height=page.image_height,
        )
        spurious_visual_suppressed, blocks = self._suppress_spurious_visual_blocks_over_text(
            blocks,
            page_width=page.image_width,
            page_height=page.image_height,
        )
        spurious_visual_suppressed += pre_spurious_visual_suppressed
        figure_text_dedup_suppressed, blocks = self._suppress_overlapped_figure_text_duplicates(
            blocks,
            page_width=page.image_width,
            page_height=page.image_height,
        )
        formula_number_merges, blocks = self._merge_formula_numbers_into_equations(
            blocks,
            page_width=page.image_width,
            page_height=page.image_height,
        )

        blocks = self._merge_short_continuation_fragments(blocks)
        blocks = self._trim_repeated_prefix_within_flows(blocks)

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
        font_mapper = page.full_coord_mapper
        for block in blocks:
            if isinstance(block, TextBlock):
                block.estimate_font_size(font_mapper)

        font_classification_stats = self._classify_block_fonts(page, blocks)

        # -- 将区块分组到 Zone ----------------------------------------
        page.zones = self._blocks_to_zones(
            blocks,
            image_width=page.image_width,
            image_height=page.image_height,
        )
        weak_multicolumn_evidence = self._has_weak_multicolumn_evidence(page, blocks)
        if weak_multicolumn_evidence:
            self._collapse_to_single_column(page, blocks)

        self._annotate_page_profile(page, blocks)
        render_mode = str((page.attributes or {}).get("render_mode", ""))

        # -- 样式推断（字号、对齐、行距、缩进、bold/italic 等）-----------
        # 仅填充 JSON 中未明确提供的字段，已有值不覆盖
        infer_block_styles(
            page.zones,
            mapper,
            justify_min_lines=self.config.align_justify_min_lines,
            page_width_px=page.image_width,
            font_mapper=font_mapper,
            reflow_title_page_width_px=page.image_width if render_mode == "reflow" else 0.0,
        )
        color_inference_stats = infer_text_colors(page, blocks)

        if page.attributes is None:
            page.attributes = {}
        strategy_name = str(self.config.reading_order_strategy or "").strip().lower()
        if strategy_name in {"legacy", "auto", "xycutpp", "xycutpp_paper", "xycutpp_hybrid", "newspaper_hybrid"}:
            page.attributes["xycutpp_debug"] = self._collect_xycutpp_proto_debug(blocks)
        page.attributes["rule_stats"] = {
            "category_fix_count": category_fix_count,
            "byline_promotions": byline_promotions,
            "hero_band_promotions": hero_band_promotions,
            "decorative_icon_suppressed": decorative_icon_suppressed,
            "spurious_visual_suppressed": spurious_visual_suppressed,
            "figure_text_dedup_suppressed": figure_text_dedup_suppressed,
            "formula_number_merges": formula_number_merges,
            "model_order_geometric_repair": model_order_repaired,
            "zone_count": len(page.zones),
            "weak_multicolumn_collapsed": int(weak_multicolumn_evidence),
        }
        if font_classification_stats:
            page.attributes["font_classification"] = font_classification_stats
        if color_inference_stats:
            page.attributes["color_inference"] = color_inference_stats
        page.attributes["quality_metrics"] = self._page_quality_metrics(page, blocks)

        return page

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _should_use_model_order(self, raw_blocks: List[dict]) -> bool:
        strategy_name = str(self.config.reading_order_strategy or "").strip().lower()
        if strategy_name != "model_order":
            return False
        return bool(raw_blocks) and all(
            isinstance(block.get("attributes"), dict)
            and "model_order" in block["attributes"]
            for block in raw_blocks
        )

    @staticmethod
    def _apply_model_order_metadata(blocks: List[Block]) -> List[Block]:
        def _model_order(block: Block, fallback: int) -> tuple[float, int]:
            attrs = getattr(block, "attributes", None) or {}
            try:
                return float(attrs.get("model_order")), fallback
            except (TypeError, ValueError):
                return float(fallback), fallback

        ordered_blocks = [
            block for _, block in sorted(
                enumerate(blocks),
                key=lambda item: _model_order(item[1], item[0]),
            )
        ]
        for index, block in enumerate(ordered_blocks):
            block.col_count = int(getattr(block, "col_count", 1) or 1)
            block.col_index = int(getattr(block, "col_index", 0) or 0)
            block.spanned_cols = list(getattr(block, "spanned_cols", None) or [block.col_index])
            if block.attributes is None:
                block.attributes = {}
            block.attributes["reading_order_strategy"] = "model_order"
            block.attributes.setdefault("model_order", index)
        return ordered_blocks

    def _assign_model_order_columns(self, blocks: List[Block], page_width: int) -> None:
        if page_width <= 0 or len(blocks) < 2:
            return

        skeleton = [
            block for block in blocks
            if isinstance(block, TextBlock)
            and block.block_type in {
                BlockType.TEXT,
                BlockType.ABSTRACT,
                BlockType.REFERENCE,
                BlockType.FIGURE_CAPTION,
                BlockType.TABLE_CAPTION,
                BlockType.FOOTNOTE,
            }
            and block.block_type not in _ZONE_STRIP_TYPES
            and float(block.bbox.width) <= float(page_width) * 0.42
            and (
                block.count_lines() >= 2
                or len((block.full_text() or "").strip()) >= 18
            )
        ]
        if len(skeleton) < 2:
            for block in blocks:
                block.col_count = 1
                block.col_index = 0
                block.spanned_cols = [0]
            return

        col_bounds = self._side_note_three_column_bounds(blocks, page_width)
        if not col_bounds:
            _columns, col_bounds = detect_columns(
                skeleton,
                page_width,
                max_cols=self.config.max_cols,
                cluster_thresh=self.config.column_cluster_thresh,
            )
        col_count = len(col_bounds)
        if col_count <= 1:
            for block in blocks:
                block.col_count = 1
                block.col_index = 0
                block.spanned_cols = [0]
            return

        detect_spanned_blocks(blocks, col_bounds)
        for block in blocks:
            block.col_count = col_count
            self._collapse_weak_title_span_to_anchor_column(block, col_bounds)
            if block.block_type in _ZONE_STRIP_TYPES:
                block.spanned_cols = list(range(col_count))
                block.col_index = 0
            if block.attributes is None:
                block.attributes = {}
            block.attributes["column_source"] = "model_order_geometry"

    @staticmethod
    def _collapse_weak_title_span_to_anchor_column(block: Block, col_bounds: List[tuple[float, float]]) -> bool:
        if (
            not isinstance(block, TextBlock)
            or block.block_type != BlockType.TITLE
            or len(col_bounds) < 2
            or len(getattr(block, "spanned_cols", []) or []) <= 1
        ):
            return False

        width = max(float(block.bbox.width), 1.0)
        overlaps: List[tuple[int, float]] = []
        for col_idx, (cx1, cx2) in enumerate(col_bounds):
            overlap = max(0.0, min(float(block.bbox.x2), float(cx2)) - max(float(block.bbox.x1), float(cx1)))
            if overlap > 0:
                overlaps.append((col_idx, overlap))
        if len(overlaps) <= 1:
            return False

        overlaps.sort(key=lambda item: item[1], reverse=True)
        best_col, best_overlap = overlaps[0]
        second_overlap = overlaps[1][1]
        if best_overlap / width < 0.55:
            return False
        if second_overlap > max(32.0, width * 0.08):
            return False

        block.col_index = best_col
        block.spanned_cols = [best_col]
        return True

    @classmethod
    def _repair_anomalous_model_order(
        cls,
        blocks: List[Block],
        *,
        page_width: int,
        page_height: int,
    ) -> tuple[int, List[Block]]:
        if any(
            int(getattr(block, "col_count", 1) or 1) > 2
            for block in blocks
        ):
            return 0, blocks
        if not cls._model_order_needs_geometry_repair(blocks, page_width=page_width, page_height=page_height):
            return 0, blocks

        cjk_ratio = cls._text_cjk_ratio(blocks)
        figure_count = sum(1 for b in blocks if b.block_type in FIGURE_TYPES or b.block_type == BlockType.FIGURE)
        band_major = cjk_ratio >= 0.35 and figure_count >= 1
        repaired = sorted(
            blocks,
            key=lambda block: cls._model_order_geometry_key(
                block,
                page_width=page_width,
                page_height=page_height,
                band_major=band_major,
            ),
        )
        cls._reassign_repaired_model_order_columns(
            repaired,
            page_width=page_width,
            band_major=band_major,
        )
        repaired = sorted(
            repaired,
            key=lambda block: cls._model_order_geometry_key(
                block,
                page_width=page_width,
                page_height=page_height,
                band_major=band_major,
            ),
        )
        for index, block in enumerate(repaired):
            if block.attributes is None:
                block.attributes = {}
            block.attributes["reading_order_strategy"] = "model_order_geometric_repair"
            block.attributes["model_order_repair_rank"] = index
        return 1, repaired

    @staticmethod
    def _reassign_repaired_model_order_columns(
        blocks: List[Block],
        *,
        page_width: int,
        band_major: bool,
    ) -> None:
        if page_width <= 0 or len(blocks) < 2:
            return

        page_w = max(float(page_width), 1.0)
        text_skeleton = [
            block for block in blocks
            if isinstance(block, TextBlock)
            and block.block_type in {
                BlockType.TEXT,
                BlockType.TITLE,
                BlockType.ABSTRACT,
                BlockType.REFERENCE,
                BlockType.FIGURE_CAPTION,
                BlockType.TABLE_CAPTION,
                BlockType.FOOTNOTE,
            }
            and block.block_type not in _ZONE_STRIP_TYPES
            and 0.12 <= float(block.bbox.width) / page_w <= 0.50
            and (block.count_lines() >= 2 or len((block.full_text() or "").strip()) >= 24)
        ]
        column_source = text_skeleton
        if len(column_source) < 3:
            column_source = [
                block for block in blocks
                if isinstance(block, TextBlock)
                and block.block_type in {
                    BlockType.TEXT,
                    BlockType.ABSTRACT,
                    BlockType.REFERENCE,
                    BlockType.FOOTNOTE,
                }
                and block.block_type not in _ZONE_STRIP_TYPES
                and 0.08 <= float(block.bbox.width) / page_w <= 0.58
            ]
        if len(column_source) < 3:
            return

        col_bounds = RecoveryPipeline._side_note_three_column_bounds(blocks, page_width)
        if col_bounds:
            columns = []
        else:
            columns, col_bounds = detect_columns(
                column_source,
                page_width,
                max_cols=2,
                cluster_thresh=0.10,
            )
        if len(col_bounds) < 2:
            if band_major:
                RecoveryPipeline._collapse_repaired_text_flow_columns(blocks)
                return
            col_bounds = RecoveryPipeline._fallback_two_column_bounds(column_source, page_width)
            if len(col_bounds) < 2:
                return

        col_centers = [(float(x1) + float(x2)) * 0.5 for x1, x2 in col_bounds]

        def _nearest_col(block: Block) -> int:
            cx = (float(block.bbox.x1) + float(block.bbox.x2)) * 0.5
            return min(range(len(col_centers)), key=lambda idx: abs(cx - col_centers[idx]))

        for block in blocks:
            if block.block_type in _ZONE_STRIP_TYPES or block.block_type in {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER}:
                block.col_count = 1
                block.col_index = 0
                block.spanned_cols = [0]
                continue

            width_ratio = float(block.bbox.width) / page_w
            if band_major:
                if block.block_type in {BlockType.FIGURE_CAPTION, BlockType.TABLE_CAPTION}:
                    block.col_count = 1
                    block.col_index = 0
                    block.spanned_cols = [0]
                    continue
                if block.block_type == BlockType.TITLE:
                    anchored_col = RecoveryPipeline._anchored_title_column(block, blocks, col_centers, page_width)
                    if anchored_col is None:
                        block.col_count = 1
                        block.col_index = 0
                        block.spanned_cols = [0]
                        continue
                    block.col_count = len(col_bounds)
                    block.col_index = anchored_col
                    block.spanned_cols = [anchored_col]
                    continue
                col = _nearest_col(block)
                block.col_count = len(col_bounds)
                block.col_index = col
                block.spanned_cols = [col]
                continue

            cross_page = width_ratio >= 0.58
            if cross_page and block.block_type in {BlockType.TITLE, BlockType.FOOTNOTE, BlockType.FIGURE_CAPTION, BlockType.TABLE_CAPTION}:
                block.col_count = 1
                block.col_index = 0
                block.spanned_cols = [0]
                continue

            col = _nearest_col(block)
            block.col_count = len(col_bounds)
            block.col_index = col
            block.spanned_cols = [col]

    @staticmethod
    def _collapse_repaired_text_flow_columns(blocks: List[Block]) -> None:
        for block in blocks:
            if block.block_type in {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER}:
                block.col_count = 1
                block.col_index = 0
                block.spanned_cols = [0]
                continue
            if isinstance(block, TextBlock):
                block.col_count = 1
                block.col_index = 0
                block.spanned_cols = [0]
                continue
            if block.block_type not in {BlockType.FIGURE, BlockType.TABLE, BlockType.FORMULA, BlockType.EQUATION}:
                block.col_count = 1
                block.col_index = 0
                block.spanned_cols = [0]

    @staticmethod
    def _anchored_title_column(
        title: Block,
        blocks: List[Block],
        col_centers: List[float],
        page_width: int,
    ) -> Optional[int]:
        if not isinstance(title, TextBlock) or not col_centers or page_width <= 0:
            return None
        page_w = max(float(page_width), 1.0)
        if float(title.bbox.width) > page_w * 0.22:
            return None
        title_center = (float(title.bbox.x1) + float(title.bbox.x2)) * 0.5
        body_candidates = [
            block for block in blocks
            if isinstance(block, TextBlock)
            and block is not title
            and block.block_type in {BlockType.TEXT, BlockType.ABSTRACT, BlockType.REFERENCE}
            and float(block.bbox.y1) >= float(title.bbox.y2) - max(8.0, float(title.bbox.height) * 0.5)
            and float(block.bbox.y1) <= float(title.bbox.y2) + max(160.0, float(title.bbox.height) * 5.0)
        ]
        if not body_candidates:
            return None
        nearest_body = min(
            body_candidates,
            key=lambda block: (
                max(0.0, float(block.bbox.y1) - float(title.bbox.y2)),
                abs(((float(block.bbox.x1) + float(block.bbox.x2)) * 0.5) - title_center),
            ),
        )
        body_center = (float(nearest_body.bbox.x1) + float(nearest_body.bbox.x2)) * 0.5
        return min(range(len(col_centers)), key=lambda idx: abs(body_center - col_centers[idx]))

    @staticmethod
    def _fallback_two_column_bounds(blocks: List[Block], page_width: int) -> List[tuple[float, float]]:
        if len(blocks) < 4 or page_width <= 0:
            return []
        page_w = max(float(page_width), 1.0)
        centers = sorted((float(block.bbox.x1) + float(block.bbox.x2)) * 0.5 for block in blocks)
        gaps = [(centers[idx + 1] - centers[idx], idx) for idx in range(len(centers) - 1)]
        if not gaps:
            return []
        max_gap, split_idx = max(gaps, key=lambda item: item[0])
        if max_gap < page_w * 0.18:
            return []
        divider = (centers[split_idx] + centers[split_idx + 1]) * 0.5
        left = [block for block in blocks if (float(block.bbox.x1) + float(block.bbox.x2)) * 0.5 < divider]
        right = [block for block in blocks if block not in left]
        if len(left) < 2 or len(right) < 2:
            return []
        return [
            (min(float(block.bbox.x1) for block in left), max(float(block.bbox.x2) for block in left)),
            (min(float(block.bbox.x1) for block in right), max(float(block.bbox.x2) for block in right)),
        ]

    @staticmethod
    def _side_note_three_column_bounds(blocks: List[Block], page_width: int) -> List[tuple[float, float]]:
        """Detect textbook pages with a left side-note rail plus two body rails.

        This is a geometric page pattern, not a sample-specific exception: a
        narrow annotation track on the left coexists with two regular body
        columns to its right.  Collapsing it to two columns makes the side note
        and first body column compete for the same Word column.
        """
        if page_width <= 0 or len(blocks) < 5:
            return []
        page_w = max(float(page_width), 1.0)

        side_notes = [
            block for block in blocks
            if isinstance(block, TextBlock)
            and block.block_type in {BlockType.FOOTNOTE, BlockType.TEXT}
            and block.block_type not in _ZONE_STRIP_TYPES
            and float(block.bbox.x1) <= page_w * 0.28
            and float(block.bbox.x2) <= page_w * 0.42
            and page_w * 0.10 <= float(block.bbox.width) <= page_w * 0.34
            and (
                block.count_lines() >= 3
                or len((block.full_text() or "").strip()) >= 45
            )
        ]
        if not side_notes:
            return []

        side_right = max(float(block.bbox.x2) for block in side_notes)
        body_source = [
            block for block in blocks
            if isinstance(block, TextBlock)
            and block.block_type in {BlockType.TEXT, BlockType.ABSTRACT, BlockType.REFERENCE, BlockType.TITLE}
            and block.block_type not in _ZONE_STRIP_TYPES
            and float(block.bbox.x1) >= side_right + page_w * 0.035
            and page_w * 0.16 <= float(block.bbox.width) <= page_w * 0.42
            and (
                block.count_lines() >= 2
                or len((block.full_text() or "").strip()) >= 24
            )
        ]
        if len(body_source) < 4:
            return []

        _body_columns, body_bounds = detect_columns(
            body_source,
            page_width,
            max_cols=2,
            cluster_thresh=0.08,
        )
        if len(body_bounds) != 2:
            return []
        first_body_left = float(body_bounds[0][0])
        if first_body_left - side_right < page_w * 0.045:
            return []
        if float(body_bounds[1][0]) - float(body_bounds[0][1]) < page_w * 0.012:
            return []

        side_top = min(float(block.bbox.y1) for block in side_notes)
        side_bottom = max(float(block.bbox.y2) for block in side_notes)
        body_overlap = [
            block for block in body_source
            if min(side_bottom, float(block.bbox.y2)) - max(side_top, float(block.bbox.y1)) > 0
        ]
        if len(body_overlap) < 2:
            return []

        side_bounds = (
            min(float(block.bbox.x1) for block in side_notes),
            max(float(block.bbox.x2) for block in side_notes),
        )
        return [side_bounds, *[(float(x1), float(x2)) for x1, x2 in body_bounds]]

    @classmethod
    def _model_order_needs_geometry_repair(
        cls,
        blocks: List[Block],
        *,
        page_width: int,
        page_height: int,
    ) -> bool:
        if page_width <= 0 or page_height <= 0 or len(blocks) < 4:
            return False
        max_cols = max((int(getattr(block, "col_count", 1) or 1) for block in blocks), default=1)
        if max_cols > 2:
            return False

        core = [
            block for block in blocks
            if block.block_type not in _ZONE_STRIP_TYPES
            and block.block_type not in {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER}
        ]
        if len(core) < 4:
            return False

        cjk_ratio = cls._text_cjk_ratio(core)
        figure_count = sum(1 for b in core if b.block_type in FIGURE_TYPES or b.block_type == BlockType.FIGURE)
        band_major = cjk_ratio >= 0.35 and figure_count >= 1
        geometric = sorted(
            core,
            key=lambda block: cls._model_order_geometry_key(
                block,
                page_width=page_width,
                page_height=page_height,
                band_major=band_major,
            ),
        )
        rank = {id(block): idx for idx, block in enumerate(geometric)}
        severe_inversions = 0
        y_backtracks = 0
        rank_drop_limit = max(2, int(len(core) * 0.18))
        y_back_limit = max(float(page_height) * 0.08, 90.0)
        for prev, curr in zip(core, core[1:]):
            if rank[id(prev)] - rank[id(curr)] >= rank_drop_limit:
                severe_inversions += 1
            same_track = (
                band_major
                or int(getattr(prev, "col_index", 0) or 0) == int(getattr(curr, "col_index", 0) or 0)
                or max(0.0, min(float(prev.bbox.x2), float(curr.bbox.x2)) - max(float(prev.bbox.x1), float(curr.bbox.x1)))
                >= min(float(prev.bbox.width), float(curr.bbox.width)) * 0.35
            )
            if same_track and float(curr.bbox.y1) + y_back_limit < float(prev.bbox.y1):
                y_backtracks += 1

        top_limit = float(page_height) * 0.22
        late_top_structural = any(
            idx >= max(3, int(len(core) * 0.45))
            and float(block.bbox.y1) <= top_limit
            and block.block_type in {BlockType.TITLE, BlockType.FIGURE_CAPTION, BlockType.TEXT, BlockType.FOOTNOTE}
            for idx, block in enumerate(core)
        )

        return severe_inversions >= 2 or y_backtracks >= 2 or (late_top_structural and severe_inversions >= 1)

    @staticmethod
    def _model_order_geometry_key(
        block: Block,
        *,
        page_width: int,
        page_height: int,
        band_major: bool,
    ) -> tuple:
        del page_height
        strip_group = 1
        if block.block_type in {BlockType.HEADER, BlockType.PAGE_NUMBER}:
            strip_group = 0
        elif block.block_type == BlockType.FOOTER:
            strip_group = 3

        y1 = float(block.bbox.y1)
        x1 = float(block.bbox.x1)
        if strip_group != 1:
            return (strip_group, y1, x1)

        if band_major:
            return (strip_group, y1, x1)

        page_w = max(float(page_width), 1.0)
        spanned = (
            len(getattr(block, "spanned_cols", []) or []) > 1
            or float(block.bbox.width) >= page_w * 0.52
        )
        col = -1 if spanned else int(getattr(block, "col_index", 0) or 0)
        return (strip_group, col, y1, x1)

    @staticmethod
    def _text_cjk_ratio(blocks: List[Block]) -> float:
        cjk_chars = 0
        total_chars = 0
        for block in blocks:
            if not isinstance(block, TextBlock):
                continue
            text = block.full_text()
            total_chars += len(text)
            cjk_chars += sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        return (cjk_chars / total_chars) if total_chars else 0.0

    def _get_font_classifier(self) -> Optional[FontClassifier]:
        if not bool(getattr(self.config, "font_classification_enabled", True)):
            return None
        if self._font_classifier is not None:
            return self._font_classifier
        try:
            self._font_classifier = FontClassifier.from_config(self.config)
        except Exception:
            self._font_classifier = None
        return self._font_classifier

    def _classify_block_fonts(self, page: Page, blocks: List[Block]) -> Optional[dict]:
        if not bool(getattr(self.config, "font_classification_enabled", True)):
            return None
        classifier = self._get_font_classifier()
        if classifier is None:
            return {"enabled": True, "available": False, "reason": "classifier_init_failed", "applied": 0}
        try:
            return classifier.classify_page(page, blocks)
        except Exception as exc:
            return {"enabled": True, "available": False, "reason": str(exc), "applied": 0}

    @staticmethod
    def _fix_block_categories(
        blocks: List[Block],
        page_width: int = 0,
        page_height: int = 0,
    ) -> int:
        """纠正常见的版面分析分类错误。

        例如 PaddleOCR 有时将表格标题（"TABLE I ..."）识别为 header。
        """
        section_title_re = re.compile(
            r"^\s*(\d+(?:\.\d+)*[\.、]|\d+[)）]|\(?\d+\)|[（(]?[一二三四五六七八九十百]+[)）\.、])\s*\S+"
        )
        changes = 0
        for block in blocks:
            if not isinstance(block, TextBlock):
                continue
            text = block.full_text().strip()
            if not text:
                continue
            compact_text = re.sub(r"\s+", "", text)
            upper = text.upper()
            near_top = (
                page_height > 0
                and float(block.bbox.y1) <= max(float(page_height) * 0.16, 1.0)
            )
            near_bottom = (
                page_height > 0
                and float(block.bbox.y2) >= max(float(page_height) * 0.92, 1.0)
            )
            near_footer_band = (
                page_height > 0
                and float(block.bbox.y1) >= max(float(page_height) * 0.80, 1.0)
            )
            if (
                block.block_type in {BlockType.TEXT, BlockType.FIGURE_CAPTION}
                and near_bottom
                and _PAGE_NUMBER_RE.match(compact_text)
                and float(block.bbox.width) <= max(float(page_width) * 0.12, 120.0)
            ):
                block.block_type = BlockType.PAGE_NUMBER
                changes += 1
                continue
            if block.block_type in {BlockType.TEXT, BlockType.TITLE}:
                if _FIGURE_CAPTION_RE.match(compact_text):
                    block.block_type = BlockType.FIGURE_CAPTION
                    changes += 1
                elif _TABLE_CAPTION_RE.match(compact_text):
                    block.block_type = BlockType.TABLE_CAPTION
                    changes += 1
            if block.block_type == BlockType.FIGURE_CAPTION:
                # References such as "Fig. 3 visualizes ..." are ordinary body
                # text, not captions. Captions are usually short labels placed
                # directly next to the visual object.
                if (
                    re.match(r"^\s*fig(?:ure)?\.?\s*\d+\s+\w+", text, re.IGNORECASE)
                    and (block.count_lines() >= 4 or len(text) >= 120)
                    and float(block.bbox.width) >= max(float(page_width) * 0.32, 1.0)
                ):
                    block.block_type = BlockType.TEXT
                    changes += 1
            if block.block_type == BlockType.HEADER:
                if re.match(r'TABLE\s', upper):
                    block.block_type = BlockType.TABLE_CAPTION
                    changes += 1
                elif re.match(r'FIG(URE|\.)\s', upper):
                    block.block_type = BlockType.FIGURE_CAPTION
                    changes += 1
                else:
                    is_numbered_section = bool(section_title_re.match(text))
                    shortish = len(text) <= 28
                    narrow = (
                        page_width <= 0
                        or float(block.bbox.width) <= max(float(page_width) * 0.42, 1.0)
                    )
                    if is_numbered_section and shortish and narrow and near_top:
                        block.block_type = BlockType.TITLE
                        changes += 1
            elif block.block_type == BlockType.FOOTER:
                attrs = getattr(block, "attributes", None) or {}
                raw_label = str(attrs.get("raw_layout_label", "") or "")
                if (
                    raw_label == "vision_footnote"
                    and not near_bottom
                    and len(text) >= 24
                    and float(block.bbox.width) >= max(float(page_width) * 0.58, 1.0)
                ):
                    block.block_type = BlockType.TEXT
                    changes += 1
                    continue
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
                    changes += 1
            elif block.block_type == BlockType.FOOTNOTE:
                attrs = getattr(block, "attributes", None) or {}
                raw_label = str(attrs.get("raw_layout_label", "") or "")
                if (
                    raw_label == "vision_footnote"
                    and not near_bottom
                    and len(text) >= 24
                    and float(block.bbox.width) >= max(float(page_width) * 0.58, 1.0)
                ):
                    block.block_type = BlockType.TEXT
                    changes += 1
            elif block.block_type == BlockType.TEXT:
                footer_like = bool(_FOOTER_LIKE_RE.search(text))
                academic_footer_note = bool(_ACADEMIC_FOOTER_NOTE_RE.search(text))
                if (
                    ((near_bottom or near_footer_band) and footer_like and len(text) <= 120)
                    or (near_footer_band and academic_footer_note)
                ):
                    block.block_type = BlockType.FOOTER
                    changes += 1
                elif (
                    _infer_title_heading_level(text) is not None
                    and len(text) <= 36
                    and block.count_lines() <= 2
                    and float(block.bbox.width) <= max(float(page_width) * 0.45, 1.0)
                ):
                    block.block_type = BlockType.TITLE
                    changes += 1

            if block.block_type == BlockType.TITLE:
                level = _infer_title_heading_level(text)
                if level is not None:
                    if block.attributes is None:
                        block.attributes = {}
                    block.attributes.setdefault("heading_level", level)
        return changes

    @staticmethod
    def _promote_side_by_side_hero_bands(
        blocks: List[Block],
        page_width: int = 0,
        page_height: int = 0,
    ) -> int:
        """将被漏判的“左图右文/右图左文”图文带提升为局部双栏结构。

        仅处理当前仍是单栏的块，并要求一侧为较大的图像块，另一侧为与其
        垂直重叠的标题/正文块，避免误伤普通单栏页面。
        """
        if page_width <= 0 or len(blocks) < 2:
            return 0
        promoted = 0

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
                promoted += 1
            for block in right_col_blocks:
                block.col_count = 2
                block.col_index = 1
                block.spanned_cols = [1]
                promoted += 1
        return promoted

    @staticmethod
    def _promote_top_byline_rows(
        blocks: List[Block],
        page_width: int = 0,
        page_height: int = 0,
    ) -> int:
        """将顶部居中的署名短行从正文分栏中提升为单栏行。

        典型场景：报纸主标题下方的作者行、地点行、导语短行。
        """
        if page_width <= 0 or page_height <= 0:
            return 0
        page_center = float(page_width) * 0.5
        top_limit = float(page_height) * 0.22
        min_width = float(page_width) * 0.12
        max_width = float(page_width) * 0.32
        promoted = 0

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
            promoted += 1
            if block.col_count > 1:
                block.col_count = 1
                block.col_index = 0
                block.spanned_cols = [0]
        return promoted

    @staticmethod
    def _suppress_decorative_title_icons(
        blocks: List[Block],
        page_width: int = 0,
        page_height: int = 0,
    ) -> tuple[int, List[Block]]:
        """吸附标题旁的小装饰图标，避免其单独参与阅读顺序。"""
        if page_width <= 0 or page_height <= 0 or len(blocks) < 2:
            return 0, blocks

        figure_like = (BlockType.FIGURE, BlockType.IMAGE if hasattr(BlockType, "IMAGE") else BlockType.FIGURE)
        titles = [blk for blk in blocks if isinstance(blk, TextBlock) and blk.block_type == BlockType.TITLE]
        if not titles:
            return 0, blocks

        kept: List[Block] = []
        suppressed = 0
        for block in blocks:
            if not isinstance(block, ImageBlock) or block.block_type not in figure_like:
                kept.append(block)
                continue

            width = float(block.bbox.width)
            height = float(block.bbox.height)
            if width > float(page_width) * 0.10 or height > max(float(page_height) * 0.08, 120.0):
                kept.append(block)
                continue

            best_title: TextBlock | None = None
            best_score = float("inf")
            for title in titles:
                if title is block:
                    continue
                title_text = title.full_text().strip()
                if not title_text or len(title_text) > 80:
                    continue
                vertical_overlap = min(float(block.bbox.y2), float(title.bbox.y2)) - max(float(block.bbox.y1), float(title.bbox.y1))
                center_gap_y = abs(float(block.bbox.center[1]) - float(title.bbox.center[1]))
                horizontal_gap = min(
                    abs(float(block.bbox.x2) - float(title.bbox.x1)),
                    abs(float(title.bbox.x2) - float(block.bbox.x1)),
                )
                horizontal_overlap = max(0.0, min(float(block.bbox.x2), float(title.bbox.x2)) - max(float(block.bbox.x1), float(title.bbox.x1)))
                near_same_band = vertical_overlap >= -4.0 or center_gap_y <= max(float(height), float(title.bbox.height)) * 1.2
                near_horizontally = horizontal_overlap >= 8.0 or horizontal_gap <= max(float(page_width) * 0.05, 64.0)
                if not (near_same_band and near_horizontally):
                    continue
                score = max(0.0, horizontal_gap) + center_gap_y * 0.35 - horizontal_overlap * 0.2
                if score < best_score:
                    best_score = score
                    best_title = title

            if best_title is None:
                kept.append(block)
                continue

            if best_title.attributes is None:
                best_title.attributes = {}
            best_title.attributes["has_decorative_icon"] = True
            best_title.attributes["decorative_icon_bbox"] = [
                float(block.bbox.x1), float(block.bbox.y1), float(block.bbox.x2), float(block.bbox.y2)
            ]
            suppressed += 1

        return suppressed, kept

    @staticmethod
    def _suppress_spurious_visual_blocks_over_text(
        blocks: List[Block],
        page_width: int = 0,
        page_height: int = 0,
    ) -> tuple[int, List[Block]]:
        """Drop figure/formula boxes that are actually text-region duplicates."""
        if len(blocks) < 2:
            return 0, blocks

        text_blocks = [
            blk for blk in blocks
            if isinstance(blk, TextBlock)
            and blk.block_type in {BlockType.TEXT, BlockType.TITLE, BlockType.REFERENCE, BlockType.ABSTRACT}
        ]
        if not text_blocks:
            return 0, blocks

        def _intersection_area(a: Block, b: Block) -> float:
            overlap_w = max(0.0, min(float(a.bbox.x2), float(b.bbox.x2)) - max(float(a.bbox.x1), float(b.bbox.x1)))
            overlap_h = max(0.0, min(float(a.bbox.y2), float(b.bbox.y2)) - max(float(a.bbox.y1), float(b.bbox.y1)))
            return overlap_w * overlap_h

        drop_ids: set[int] = set()
        for block in blocks:
            if not isinstance(block, (ImageBlock, EquationBlock)):
                continue
            if block.block_type not in {BlockType.FIGURE, BlockType.FORMULA, BlockType.EQUATION}:
                continue

            attrs = getattr(block, "attributes", None) or {}
            nested_children = attrs.get("nested_children") if isinstance(attrs, dict) else None
            if (
                block.block_type == BlockType.FIGURE
                and page_height > 0
                and float(block.bbox.y1) >= float(page_height) * 0.88
                and float(block.bbox.height) <= max(float(page_height) * 0.08, 120.0)
                and isinstance(nested_children, list)
                and nested_children
                and all(
                    isinstance(child, dict)
                    and str(child.get("type") or child.get("category") or "") in {"page_number", "footer"}
                    for child in nested_children
                )
            ):
                drop_ids.add(id(block))
                continue

            block_area = max(float(block.bbox.area), 1.0)
            covered_text_area = 0.0
            long_text_hits = 0
            title_hits = 0
            for text_block in text_blocks:
                overlap_area = _intersection_area(block, text_block)
                if overlap_area <= 0:
                    continue
                text_area = max(float(text_block.bbox.area), 1.0)
                contains_text = overlap_area / text_area >= 0.82
                mostly_text = overlap_area / block_area >= 0.82
                long_text_like = (
                    text_block.block_type == BlockType.TEXT
                    and (
                        text_block.count_lines() >= 4
                        or len(text_block.full_text().strip()) >= 120
                    )
                )
                title_duplicate = (
                    text_block.block_type == BlockType.TITLE
                    and contains_text
                    and float(block.bbox.height) <= max(float(page_height) * 0.04, 72.0)
                )
                long_text_duplicate = (
                    long_text_like
                    and contains_text
                    and mostly_text
                )
                if title_duplicate or long_text_duplicate:
                    drop_ids.add(id(block))
                    break

                if contains_text:
                    covered_text_area += min(overlap_area, text_area)
                    if long_text_like:
                        long_text_hits += 1
                    elif text_block.block_type == BlockType.TITLE:
                        title_hits += 1

            if id(block) in drop_ids:
                continue
            aggregate_text_duplicate = (
                block.block_type == BlockType.FIGURE
                and long_text_hits >= 2
                and covered_text_area / block_area >= 0.55
            )
            narrow_section_band_duplicate = (
                block.block_type in {BlockType.FORMULA, BlockType.EQUATION}
                and title_hits >= 1
                and covered_text_area / block_area >= 0.65
                and float(block.bbox.height) <= max(float(page_height) * 0.05, 96.0)
            )
            if aggregate_text_duplicate or narrow_section_band_duplicate:
                drop_ids.add(id(block))

        if not drop_ids:
            return 0, blocks
        return len(drop_ids), [blk for blk in blocks if id(blk) not in drop_ids]

    @staticmethod
    def _suppress_overlapped_figure_text_duplicates(
        blocks: List[Block],
        page_width: int = 0,
        page_height: int = 0,
    ) -> tuple[int, List[Block]]:
        """移除与 figure 高重叠的短文本 OCR 泄漏。"""
        if len(blocks) < 2:
            return 0, blocks

        figures = [
            blk for blk in blocks
            if isinstance(blk, ImageBlock) and blk.block_type == BlockType.FIGURE
        ]
        if not figures:
            return 0, blocks

        drop_ids: set[int] = set()
        for fig in figures:
            fig_area = max(float(fig.bbox.area), 1.0)
            for blk in blocks:
                if blk is fig or not isinstance(blk, TextBlock):
                    continue
                if blk.block_type not in {BlockType.TEXT, BlockType.TITLE}:
                    continue
                text = blk.full_text().strip()
                if not text:
                    continue
                compact = re.sub(r"\s+", "", text)
                if len(compact) > 12 or blk.count_lines() > 3:
                    continue
                overlap_w = max(0.0, min(float(fig.bbox.x2), float(blk.bbox.x2)) - max(float(fig.bbox.x1), float(blk.bbox.x1)))
                overlap_h = max(0.0, min(float(fig.bbox.y2), float(blk.bbox.y2)) - max(float(fig.bbox.y1), float(blk.bbox.y1)))
                overlap_area = overlap_w * overlap_h
                if overlap_area <= 0:
                    continue
                blk_area = max(float(blk.bbox.area), 1.0)
                overlap_ratio = overlap_area / min(fig_area, blk_area)
                area_ratio = blk_area / fig_area
                if overlap_ratio < 0.92 or not (0.65 <= area_ratio <= 1.35):
                    continue
                drop_ids.add(id(blk))

        if not drop_ids:
            return 0, blocks
        kept = [blk for blk in blocks if id(blk) not in drop_ids]
        return len(drop_ids), kept

    @staticmethod
    def _merge_formula_numbers_into_equations(
        blocks: List[Block],
        page_width: int = 0,
        page_height: int = 0,
    ) -> tuple[int, List[Block]]:
        """Attach PP-DocLayoutV3 formula_number boxes to the matching formula.

        Formula numbers are semantic text, but the detector also gives us a crop.
        If left as an independent EquationBlock, the DOCX renderer treats that
        crop like a formula image and scales "(17)" into a huge object.  Merging
        preserves the model's reading-order item while making the renderer emit
        the number as right-aligned text in the formula paragraph.
        """
        if len(blocks) < 2:
            return 0, blocks

        def _attrs(block: Block) -> dict:
            return getattr(block, "attributes", None) or {}

        def _formula_number_text(block: Block) -> str:
            value = str(_attrs(block).get("formula_number_text", "") or "").strip()
            if value and _FORMULA_NUMBER_TEXT_RE.match(value):
                return value
            return ""

        def _is_number_block(block: Block) -> bool:
            attrs = _attrs(block)
            raw_label = str(attrs.get("raw_layout_label", "") or "")
            if raw_label != "formula_number":
                return False
            if not isinstance(block, EquationBlock):
                return False
            text = _formula_number_text(block)
            if not text:
                return False
            if page_width > 0 and float(block.bbox.width) > max(float(page_width) * 0.12, 120.0):
                return False
            if page_height > 0 and float(block.bbox.height) > max(float(page_height) * 0.08, 120.0):
                return False
            return True

        def _vertical_score(body: Block, num: Block) -> Optional[float]:
            overlap = min(float(body.bbox.y2), float(num.bbox.y2)) - max(float(body.bbox.y1), float(num.bbox.y1))
            body_h = max(float(body.bbox.height), 1.0)
            num_h = max(float(num.bbox.height), 1.0)
            center_delta = abs((float(body.bbox.y1) + float(body.bbox.y2)) * 0.5 - (float(num.bbox.y1) + float(num.bbox.y2)) * 0.5)
            if overlap >= min(body_h, num_h) * 0.18:
                return center_delta
            if center_delta <= max(body_h, num_h) * 0.75:
                return center_delta + max(0.0, -overlap)
            return None

        drop_ids: set[int] = set()
        merges = 0
        formula_blocks = [
            block for block in blocks
            if isinstance(block, EquationBlock)
            and block.block_type in {BlockType.FORMULA, BlockType.EQUATION}
            and not _is_number_block(block)
        ]
        for number_block in blocks:
            if not _is_number_block(number_block):
                continue
            number_text = _formula_number_text(number_block)
            candidates: List[tuple[float, Block]] = []
            for formula in formula_blocks:
                score_y = _vertical_score(formula, number_block)
                if score_y is None:
                    continue
                if float(formula.bbox.x1) >= float(number_block.bbox.x2):
                    continue
                horizontal_gap = max(0.0, float(number_block.bbox.x1) - float(formula.bbox.x2))
                if page_width > 0 and horizontal_gap > max(float(page_width) * 0.28, 240.0):
                    continue
                same_col_bonus = 0.0 if int(getattr(formula, "col_index", -1) or 0) == int(getattr(number_block, "col_index", -2) or 0) else 20.0
                candidates.append((score_y + horizontal_gap * 0.08 + same_col_bonus, formula))
            if not candidates:
                continue
            _, target = min(candidates, key=lambda item: item[0])
            if target.attributes is None:
                target.attributes = {}
            target.attributes["formula_number_text"] = number_text
            target.attributes["formula_number_bbox"] = [
                float(number_block.bbox.x1),
                float(number_block.bbox.y1),
                float(number_block.bbox.x2),
                float(number_block.bbox.y2),
            ]
            drop_ids.add(id(number_block))
            merges += 1

        if not drop_ids:
            return 0, blocks
        return merges, [block for block in blocks if id(block) not in drop_ids]

    @staticmethod
    def _merge_short_continuation_fragments(blocks: List[Block]) -> List[Block]:
        """吸收被错误切成独立块的短续接文本。"""
        if len(blocks) < 2:
            return blocks

        merged: List[Block] = []

        def _norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def _ends_sentence(text: str) -> bool:
            return bool(re.search(r"[。！？!?\.][”\"']?\s*$", text or ""))

        for block in blocks:
            if (
                merged
                and isinstance(block, TextBlock)
                and isinstance(merged[-1], TextBlock)
                and block.block_type in {BlockType.TEXT, BlockType.TITLE}
                and merged[-1].block_type in {BlockType.TEXT, BlockType.TITLE, BlockType.REFERENCE}
                and block.lines
                and block.count_lines() == 1
            ):
                prev = merged[-1]
                prev_text = _norm(prev.full_text())
                curr_text = _norm(block.full_text())
                horizontal_gap = float(block.bbox.x1) - float(prev.bbox.x2)
                same_row = abs(float(block.bbox.y1) - float(prev.bbox.y1)) <= 28.0
                next_column_top_fragment = (
                    block.col_index == prev.col_index + 1
                    and float(block.bbox.y1) < float(prev.bbox.y1)
                    and float(prev.bbox.y1) - float(block.bbox.y1) <= 120.0
                    and 0.0 <= horizontal_gap <= 96.0
                )
                continuation_like = (
                    prev_text
                    and curr_text
                    and len(curr_text) <= 24
                    and not _ends_sentence(prev_text)
                    and (
                        (0.0 <= horizontal_gap <= 96.0 and same_row)
                        or next_column_top_fragment
                    )
                )
                if continuation_like:
                    prev.lines.extend(block.lines)
                    prev.lines.sort(
                        key=lambda line: (
                            min((float(pt[1]) for pt in line.text_region), default=float(prev.bbox.y1)),
                            min((float(pt[0]) for pt in line.text_region), default=float(prev.bbox.x1)),
                        )
                    )
                    prev.bbox = prev.bbox.union(block.bbox)
                    continue

            merged.append(block)

        return merged

    @staticmethod
    def _trim_repeated_prefix_within_flows(blocks: List[Block]) -> List[Block]:
        """仅在同一 flow 内裁剪明显重复的前缀行。"""
        if len(blocks) < 2:
            return blocks

        def _flow_id(block: Block) -> str:
            attrs = getattr(block, "attributes", None) or {}
            return str(attrs.get("flow_id", ""))

        def _norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def _x_overlap_ratio(a: Block, b: Block) -> float:
            overlap = max(0.0, min(float(a.bbox.x2), float(b.bbox.x2)) - max(float(a.bbox.x1), float(b.bbox.x1)))
            span = min(max(float(a.bbox.width), 1.0), max(float(b.bbox.width), 1.0))
            return overlap / span

        def _vertical_gap(a: Block, b: Block) -> float:
            if float(a.bbox.y1) <= float(b.bbox.y2) and float(b.bbox.y1) <= float(a.bbox.y2):
                return 0.0
            return min(abs(float(a.bbox.y1) - float(b.bbox.y2)), abs(float(b.bbox.y1) - float(a.bbox.y2)))

        def _trim_lines(block: TextBlock, count: int) -> bool:
            remaining = list(block.lines[count:])
            if not remaining:
                return False
            block.lines = remaining
            xs: List[float] = []
            ys: List[float] = []
            for line in remaining:
                if line.text_region:
                    xs.extend(float(pt[0]) for pt in line.text_region)
                    ys.extend(float(pt[1]) for pt in line.text_region)
            if xs and ys:
                block.bbox = block.bbox.__class__(
                    x1=min(xs),
                    y1=min(ys),
                    x2=max(xs),
                    y2=max(ys),
                )
            return True

        trimmed: List[Block] = [blocks[0]]
        for block in blocks[1:]:
            prev_flow_id = _flow_id(trimmed[-1]) if trimmed else ""
            same_flow = bool(prev_flow_id) and prev_flow_id == _flow_id(block)
            same_column_track = (
                getattr(block, "col_index", 0) == getattr(trimmed[-1], "col_index", 0)
                and getattr(block, "col_count", 1) == getattr(trimmed[-1], "col_count", 1)
            ) if trimmed else False
            if (
                trimmed
                and isinstance(block, TextBlock)
                and isinstance(trimmed[-1], TextBlock)
                and (same_flow or same_column_track)
                and _x_overlap_ratio(trimmed[-1], block) >= 0.75
                and _vertical_gap(trimmed[-1], block) <= 48.0
            ):
                prev = trimmed[-1]
                prev_lines = [_norm(line.text) for line in prev.lines if _norm(line.text)]
                curr_lines = [_norm(line.text) for line in block.lines if _norm(line.text)]
                max_shared = min(2, len(prev_lines), len(curr_lines))
                shared = 0
                for size in range(max_shared, 0, -1):
                    if prev_lines[-size:] == curr_lines[:size]:
                        shared = size
                        break
                if shared == 0 and prev_lines and curr_lines:
                    prev_last = prev_lines[-1]
                    curr_first = curr_lines[0]
                    if (
                        len(prev.lines) <= 2
                        and len(prev_last) >= 12
                        and (
                            curr_first.startswith(prev_last)
                            or prev_last.startswith(curr_first)
                        )
                    ):
                        shared = 1
                if shared > 0:
                    keep = _trim_lines(block, shared)
                    if not keep:
                        continue
            trimmed.append(block)

        trimmed = RecoveryPipeline._absorb_short_tail_fragments_within_flows(trimmed)
        return RecoveryPipeline._straighten_same_flow_boundaries(trimmed)

    @staticmethod
    def _absorb_short_tail_fragments_within_flows(blocks: List[Block]) -> List[Block]:
        """把同一 flow / 同列中紧贴正文底部的短尾块吸回前一正文块。"""
        if len(blocks) < 2:
            return blocks

        def _flow_id(block: Block) -> str:
            attrs = getattr(block, "attributes", None) or {}
            return str(attrs.get("flow_id", ""))

        merged: List[Block] = []
        for block in blocks:
            if (
                merged
                and isinstance(block, TextBlock)
                and isinstance(merged[-1], TextBlock)
                and block.block_type == BlockType.TEXT
                and merged[-1].block_type == BlockType.TEXT
                and _flow_id(block)
                and _flow_id(block) == _flow_id(merged[-1])
                and getattr(block, "col_index", 0) == getattr(merged[-1], "col_index", 0)
            ):
                prev = merged[-1]
                vertical_gap = float(block.bbox.y1) - float(prev.bbox.y2)
                overlap = max(0.0, min(float(block.bbox.x2), float(prev.bbox.x2)) - max(float(block.bbox.x1), float(prev.bbox.x1)))
                overlap_ratio = overlap / max(1.0, min(float(block.bbox.width), float(prev.bbox.width)))
                curr_text = block.full_text()
                absorbable = (
                    vertical_gap <= 36.0
                    and overlap_ratio >= 0.80
                    and prev.count_lines() >= 4
                    and 1 <= block.count_lines() <= 2
                    and len((curr_text or "").strip()) <= 120
                )
                if absorbable:
                    prev.lines.extend(block.lines)
                    prev.bbox = prev.bbox.union(block.bbox)
                    continue

            merged.append(block)

        return merged

    @staticmethod
    def _straighten_same_flow_boundaries(blocks: List[Block]) -> List[Block]:
        if len(blocks) < 2:
            return blocks

        def _flow_id(block: Block) -> str:
            attrs = getattr(block, "attributes", None) or {}
            return str(attrs.get("flow_id", ""))

        for prev, curr in zip(blocks, blocks[1:]):
            prev_flow_id = _flow_id(prev)
            curr_flow_id = _flow_id(curr)
            # 仅对显式 article-flow 内的相邻块做边界拉直。
            # 几何兜底路径下 flow_id 可能为空，若继续按空串相等处理，
            # 会把普通相邻块误判为同一 flow，导致 bbox 被大幅裁坏。
            if not prev_flow_id or not curr_flow_id or prev_flow_id != curr_flow_id:
                continue
            if getattr(prev, "col_index", 0) != getattr(curr, "col_index", 0):
                continue
            overlap = min(float(prev.bbox.y2), float(curr.bbox.y2)) - max(float(prev.bbox.y1), float(curr.bbox.y1))
            prev_short = isinstance(prev, TextBlock) and len(prev.lines) <= 2
            curr_short = isinstance(curr, TextBlock) and len(curr.lines) <= 2
            if overlap <= 0 or (overlap > 36.0 and not prev_short and not curr_short):
                continue
            mid = (max(float(prev.bbox.y1), float(curr.bbox.y1)) + min(float(prev.bbox.y2), float(curr.bbox.y2))) * 0.5
            prev.bbox.y2 = max(float(prev.bbox.y1), mid - 1.0)
            curr.bbox.y1 = min(float(curr.bbox.y2), mid + 1.0)

        return blocks

    @staticmethod
    def _annotate_page_profile(page: Page, blocks: List[Block]) -> None:
        if page.attributes is None:
            page.attributes = {}
        profile = RecoveryPipeline._infer_layout_profile(page, blocks)
        page.attributes["layout_profile"] = profile
        page.attributes["render_mode"] = RecoveryPipeline._render_mode_for_profile(profile)

    @staticmethod
    def _render_mode_for_profile(profile: str) -> str:
        if profile in {"single_column", "table_heavy", "textbook_mixed"}:
            return "reflow"
        if profile == "academic_two_col":
            return "native_columns"
        return "grid"

    @staticmethod
    def _collect_xycutpp_proto_debug(blocks: List[Block]) -> dict:
        phase_counts: dict[str, int] = {}
        strategy_counts: dict[str, int] = {}
        cross_candidates: list[str] = []
        restore_pairs: list[dict] = []

        for block in blocks:
            attrs = getattr(block, "attributes", None) or {}
            debug = attrs.get("xycutpp_proto")
            if not isinstance(debug, dict):
                continue

            phase = str(debug.get("phase", "") or "")
            strategy = str(debug.get("strategy", "") or "")
            if phase:
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            if strategy:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            if bool(debug.get("cross_candidate")):
                block_id = getattr(block, "block_id", "")
                if block_id:
                    cross_candidates.append(block_id)
            if "restore_rank" in debug:
                restore_pairs.append(
                    {
                        "id": getattr(block, "block_id", ""),
                        "phase": phase,
                        "restore_anchor_id": str(debug.get("restore_anchor_id", "") or ""),
                        "restore_rank": float(debug.get("restore_rank", -1.0)),
                    }
                )

        restore_pairs.sort(key=lambda item: (item["restore_rank"], item["id"]))
        return {
            "phase_counts": phase_counts,
            "strategy_counts": strategy_counts,
            "cross_candidates": cross_candidates,
            "restore_pairs": restore_pairs,
        }

    @staticmethod
    def _infer_layout_profile(page: Page, blocks: List[Block]) -> str:
        max_cols = max((zone.col_count for zone in page.zones), default=1)
        table_count = sum(1 for b in blocks if b.block_type == BlockType.TABLE)
        figure_count = sum(1 for b in blocks if b.block_type in FIGURE_TYPES or b.block_type == BlockType.FIGURE)
        title_count = sum(1 for b in blocks if b.block_type == BlockType.TITLE)
        cjk_chars = 0
        total_chars = 0
        for b in blocks:
            if isinstance(b, TextBlock):
                text = b.full_text()
                total_chars += len(text)
                cjk_chars += sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        cjk_ratio = (cjk_chars / total_chars) if total_chars else 0.0

        if table_count >= 2 and len(blocks) <= 12:
            return "table_heavy"
        if max_cols <= 1:
            return "single_column"
        if RecoveryPipeline._has_weak_multicolumn_evidence(page, blocks):
            return "single_column"
        if max_cols == 2 and cjk_ratio < 0.25 and title_count <= 4:
            return "academic_two_col"
        if max_cols >= 3 and figure_count >= 1 and title_count >= 2:
            return "newspaper_mixed"
        if cjk_ratio >= 0.35 and (table_count >= 1 or figure_count >= 1):
            return "textbook_mixed"
        return "generic_complex"

    @staticmethod
    def _collapse_to_single_column(page: Page, blocks: List[Block]) -> None:
        """将已判定为弱分栏的页面元数据折叠为单栏。

        该步骤不改变阅读顺序，只清理早期分栏猜测留下的列索引，
        防止后续对齐/渲染继续把短文本块当作独立列处理。
        """
        for block in blocks:
            block.col_count = 1
            block.col_index = 0
            block.spanned_cols = [0]
        for zone in page.zones:
            zone.col_count = 1
            zone.has_spanned = False
        page.zones = RecoveryPipeline._merge_adjacent_single_column_zones(page.zones, blocks)

    @staticmethod
    def _merge_adjacent_single_column_zones(zones: List[Zone], blocks: List[Block]) -> List[Zone]:
        if not zones:
            return []
        order_index = {id(block): idx for idx, block in enumerate(blocks)}

        def _sort_zone(zone: Zone) -> None:
            zone.blocks.sort(key=lambda block: order_index.get(id(block), 10**9))

        merged: List[Zone] = []
        for zone in zones:
            _sort_zone(zone)
            if (
                merged
                and zone.col_count == 1
                and merged[-1].col_count == 1
                and zone.rendering_strategy != "strip_row"
                and merged[-1].rendering_strategy != "strip_row"
            ):
                merged[-1].blocks.extend(zone.blocks)
                _sort_zone(merged[-1])
                merged[-1].has_spanned = False
                continue
            merged.append(zone)
        return merged

    @staticmethod
    def _has_weak_multicolumn_evidence(page: Page, blocks: List[Block]) -> bool:
        """判断当前多栏判定是否缺少稳定的并行正文证据。

        真正的分栏通常会在至少两个列中各自形成多个窄正文块；如果页面
        主要由跨栏/宽文本块组成，少量短块不应把整页推成多栏渲染。
        """
        max_cols = max((zone.col_count for zone in page.zones), default=1)
        if max_cols <= 1:
            return False

        text_blocks = [
            b for b in blocks
            if isinstance(b, TextBlock)
            and b.block_type in {BlockType.TEXT, BlockType.TITLE, BlockType.ABSTRACT, BlockType.REFERENCE}
        ]
        if len(text_blocks) < 3:
            return False

        page_width = max(float(page.image_width), 1.0)
        total_area = sum(max(float(b.bbox.area), 0.0) for b in text_blocks)
        if total_area <= 0:
            return False

        wide_or_spanned_area = 0.0
        confined_by_col: dict[int, list[TextBlock]] = {}
        for block in text_blocks:
            width_ratio = float(block.bbox.width) / page_width
            spanned = len(getattr(block, "spanned_cols", []) or []) > 1 or int(getattr(block, "col_count", 1) or 1) <= 1
            if spanned or width_ratio >= 0.52:
                wide_or_spanned_area += max(float(block.bbox.area), 0.0)
                continue

            text = (block.full_text() or "").strip()
            if len(text) >= 12 or block.count_lines() >= 2:
                confined_by_col.setdefault(int(getattr(block, "col_index", 0) or 0), []).append(block)

        stable_columns = 0
        for col_blocks in confined_by_col.values():
            col_blocks = sorted(col_blocks, key=lambda b: (float(b.bbox.y1), float(b.bbox.x1)))
            if len(col_blocks) < 2:
                continue
            y_span = max(float(b.bbox.y2) for b in col_blocks) - min(float(b.bbox.y1) for b in col_blocks)
            text_len = sum(len((b.full_text() or "").strip()) for b in col_blocks)
            if y_span >= float(page.image_height) * 0.12 and text_len >= 40:
                stable_columns += 1

        wide_ratio = wide_or_spanned_area / total_area
        return wide_ratio >= 0.60 and stable_columns < 2

    @staticmethod
    def _page_quality_metrics(page: Page, blocks: List[Block]) -> dict:
        max_cols = max((zone.col_count for zone in page.zones), default=1)
        table_count = sum(1 for b in blocks if b.block_type == BlockType.TABLE)
        figure_count = sum(1 for b in blocks if b.block_type in FIGURE_TYPES or b.block_type == BlockType.FIGURE)
        title_count = sum(1 for b in blocks if b.block_type == BlockType.TITLE)
        block_area = sum(max(float(b.bbox.area), 0.0) for b in blocks)
        page_area = max(float(page.image_width * page.image_height), 1.0)
        return {
            "zone_count": len(page.zones),
            "max_cols": max_cols,
            "table_count": table_count,
            "figure_count": figure_count,
            "title_count": title_count,
            "content_density": round(block_area / page_area, 4),
            "weak_multicolumn_evidence": RecoveryPipeline._has_weak_multicolumn_evidence(page, blocks),
        }

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

        def _flow_id(block: Block) -> str:
            attrs = getattr(block, "attributes", None) or {}
            return str(attrs.get("flow_id", ""))

        def _flow_kind(block: Block) -> str:
            attrs = getattr(block, "attributes", None) or {}
            return str(attrs.get("flow_kind", ""))

        def _region_id(block: Block) -> str:
            attrs = getattr(block, "attributes", None) or {}
            debug = attrs.get("xycutpp_proto", {}) if isinstance(attrs, dict) else {}
            if isinstance(debug, dict):
                return str(debug.get("region_id", "") or "")
            return ""

        def _region_kind(block: Block) -> str:
            attrs = getattr(block, "attributes", None) or {}
            debug = attrs.get("xycutpp_proto", {}) if isinstance(attrs, dict) else {}
            if isinstance(debug, dict):
                return str(debug.get("region_kind", "") or "")
            return ""

        def _is_structural_region_kind(kind: str) -> bool:
            return kind in {
                "local_parallel_text_band",
                "wraparound_section",
                "spanning_article_band",
            }

        def _same_zone_region(
            left_region_id: str,
            left_region_kind: str,
            right_region_id: str,
            right_region_kind: str,
        ) -> bool:
            if left_region_id == right_region_id:
                return True
            return not _is_structural_region_kind(left_region_kind) and not _is_structural_region_kind(right_region_kind)

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

        source_order_index = {id(block): idx for idx, block in enumerate(blocks)}
        core_blocks = list(blocks)
        prefix_strip_blocks: List[Block] = []
        while core_blocks and _is_top_strip_block(core_blocks[0]):
            prefix_strip_blocks.append(core_blocks.pop(0))

        suffix_strip_blocks: List[Block] = [
            block for block in core_blocks if _is_bottom_strip_block(block)
        ]
        if suffix_strip_blocks:
            suffix_ids = {id(block) for block in suffix_strip_blocks}
            core_blocks = [block for block in core_blocks if id(block) not in suffix_ids]
            suffix_strip_blocks.sort(key=lambda b: source_order_index.get(id(b), 10**9))

        blocks = core_blocks
        if not blocks:
            strip_blocks = prefix_strip_blocks + suffix_strip_blocks
            if not strip_blocks:
                return []
            return [Zone(col_count=1, blocks=sorted(strip_blocks, key=lambda b: b.bbox.x1), has_spanned=False, flow_id="", flow_kind="")]

        order_index = {id(block): idx for idx, block in enumerate(blocks)}

        def _preserve_order(items: List[Block]) -> None:
            items.sort(key=lambda b: order_index.get(id(b), 10**9))

        def _nearest_zone_col(block: Block, target: Zone) -> int:
            col_count = max(int(target.col_count or 1), 1)
            if image_width <= 0 or col_count <= 1:
                return 0
            fallback_w = float(image_width) / float(col_count)
            centers: List[float] = []
            for ci in range(col_count):
                members = [
                    b for b in target.blocks
                    if int(getattr(b, "col_index", 0) or 0) == ci
                    and len(getattr(b, "spanned_cols", []) or []) <= 1
                ]
                if members:
                    centers.append(sum((float(b.bbox.x1) + float(b.bbox.x2)) * 0.5 for b in members) / len(members))
                else:
                    centers.append((ci + 0.5) * fallback_w)
            cx = (float(block.bbox.x1) + float(block.bbox.x2)) * 0.5
            return min(range(col_count), key=lambda ci: abs(cx - centers[ci]))

        def _continuation_column_for_zone(zone: Zone, target: Zone) -> Optional[int]:
            if image_width <= 0 or target.col_count <= 1 or not zone.blocks:
                return None
            col_count = max(int(target.col_count or 1), 1)
            col_w = float(image_width) / float(col_count)
            assigned_cols: List[int] = []
            for b in zone.blocks:
                if (
                    b.block_type not in _MULTICOL_TAIL_ABSORB_TYPES
                    or b.block_type in _ZONE_STRIP_TYPES
                    or bool(getattr(b, "attributes", {}) and b.attributes.get("is_byline_row"))
                    or float(b.bbox.width) > col_w * 0.95
                ):
                    return None
                assigned_cols.append(_nearest_zone_col(b, target))
            if not assigned_cols or len(set(assigned_cols)) != 1:
                return None

            col = assigned_cols[0]
            prior_col_blocks = [
                b for b in target.blocks
                if int(getattr(b, "col_index", 0) or 0) == col
                and len(getattr(b, "spanned_cols", []) or []) <= 1
            ]
            if not prior_col_blocks:
                return None

            first_top = min(float(b.bbox.y1) for b in zone.blocks)
            target_bottom = max((float(b.bbox.y2) for b in target.blocks), default=0.0)
            same_col_bottom = max(float(b.bbox.y2) for b in prior_col_blocks)
            same_col_gap_limit = max(96.0, float(image_height) * 0.025 if image_height > 0 else 0.0)
            target_gap_limit = max(64.0, float(image_height) * 0.015 if image_height > 0 else 0.0)
            if first_top - same_col_bottom > same_col_gap_limit:
                return None
            if first_top - target_bottom > target_gap_limit:
                return None
            return col

        zones: List[Zone] = []
        current_blocks: List[Block] = [blocks[0]]
        current_col_count: int = blocks[0].col_count
        current_flow_id: str = _flow_id(blocks[0])
        current_flow_kind: str = _flow_kind(blocks[0])
        current_region_id: str = _region_id(blocks[0])
        current_region_kind: str = _region_kind(blocks[0])

        for block in blocks[1:]:
            block_flow_id = _flow_id(block)
            block_flow_kind = _flow_kind(block)
            block_region_id = _region_id(block)
            block_region_kind = _region_kind(block)
            if (
                block.col_count == current_col_count
                and block_flow_id == current_flow_id
                and _same_zone_region(
                    current_region_id,
                    current_region_kind,
                    block_region_id,
                    block_region_kind,
                )
            ):
                current_blocks.append(block)
                if current_region_id != block_region_id:
                    current_region_id = ""
                    current_region_kind = ""
            else:
                has_spanned = any(
                    len(b.spanned_cols) > 1 for b in current_blocks
                )
                zones.append(Zone(
                    col_count=current_col_count,
                    blocks=current_blocks,
                    has_spanned=has_spanned,
                    flow_id=current_flow_id,
                    flow_kind=current_flow_kind,
                    region_id=current_region_id,
                    region_kind=current_region_kind,
                ))
                current_blocks = [block]
                current_col_count = block.col_count
                current_flow_id = block_flow_id
                current_flow_kind = block_flow_kind
                current_region_id = block_region_id
                current_region_kind = block_region_kind

        # 输出最后一组
        has_spanned = any(len(b.spanned_cols) > 1 for b in current_blocks)
        zones.append(Zone(
            col_count=current_col_count,
            blocks=current_blocks,
            has_spanned=has_spanned,
            flow_id=current_flow_id,
            flow_kind=current_flow_kind,
            region_id=current_region_id,
            region_kind=current_region_kind,
        ))

        # 后处理：将微小单栏区域吸收到相邻的多栏区域
        if image_width > 0 and len(zones) >= 2:
            merged: List[Zone] = []
            idx = 0
            while idx < len(zones):
                zone = zones[idx]
                if zone.col_count == 1 and merged and merged[-1].col_count > 1 and zone.flow_id == merged[-1].flow_id:
                    target = merged[-1]
                    continuation_col = _continuation_column_for_zone(zone, target)
                    if continuation_col is not None:
                        for b in zone.blocks:
                            b.col_count = target.col_count
                            b.col_index = continuation_col
                            b.spanned_cols = [continuation_col]
                        target.blocks.extend(zone.blocks)
                        target.has_spanned = target.has_spanned or any(
                            len(getattr(b, "spanned_cols", [])) > 1 for b in zone.blocks
                        )
                        _preserve_order(target.blocks)
                        idx += 1
                        continue

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
                        _preserve_order(target.blocks)

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
                            _preserve_order(next_target.blocks)
                            remain = next_remain

                    if remain:
                        has_spanned = any(len(b.spanned_cols) > 1 for b in remain)
                        merged.append(Zone(
                            col_count=1,
                            blocks=remain,
                            has_spanned=has_spanned,
                            flow_id=_flow_id(remain[0]) if remain else "",
                            flow_kind=_flow_kind(remain[0]) if remain else "",
                            region_id=_region_id(remain[0]) if remain else "",
                            region_kind=_region_kind(remain[0]) if remain else "",
                        ))
                    idx += 1
                    continue

                if zone.col_count == 1 and idx + 1 < len(zones) and zones[idx + 1].col_count > 1 and zone.flow_id == zones[idx + 1].flow_id:
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
                        _preserve_order(target.blocks)
                    if remain:
                        merged.append(Zone(
                            col_count=1,
                            blocks=remain,
                            has_spanned=any(len(b.spanned_cols) > 1 for b in remain),
                            flow_id=_flow_id(remain[0]) if remain else "",
                            flow_kind=_flow_kind(remain[0]) if remain else "",
                            region_id=_region_id(remain[0]) if remain else "",
                            region_kind=_region_kind(remain[0]) if remain else "",
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
                    and zone.flow_id == prev.flow_id
                    and _same_zone_region(
                        prev.region_id,
                        prev.region_kind,
                        zone.region_id,
                        zone.region_kind,
                    )
                    and zone.rendering_strategy != "strip_row"
                    and prev.rendering_strategy != "strip_row"
                ):
                    prev.blocks.extend(zone.blocks)
                    prev.has_spanned = prev.has_spanned or zone.has_spanned
                    if prev.region_id != zone.region_id:
                        prev.region_id = ""
                        prev.region_kind = ""
                    _preserve_order(prev.blocks)
                else:
                    consolidated.append(zone)
            zones = consolidated

        if prefix_strip_blocks:
            zones.insert(0, Zone(
                col_count=1,
                blocks=sorted(prefix_strip_blocks, key=lambda b: source_order_index.get(id(b), 10**9)),
                has_spanned=False,
                flow_id="",
                flow_kind="",
                region_id="",
                region_kind="",
            ))
        if suffix_strip_blocks:
            zones.append(Zone(
                col_count=1,
                blocks=sorted(suffix_strip_blocks, key=lambda b: source_order_index.get(id(b), 10**9)),
                has_spanned=False,
                flow_id="",
                flow_kind="",
                region_id="",
                region_kind="",
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
