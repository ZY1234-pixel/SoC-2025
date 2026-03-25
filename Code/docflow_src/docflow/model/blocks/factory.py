"""BlockFactory —— 从原始字典创建类型化的 Block 实例。"""

from __future__ import annotations

import base64
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from docflow.model.base import BBox, Block, BlockType
from docflow.model.blocks.text_block import TextBlock, TextLine
from docflow.model.blocks.table_block import TableBlock
from docflow.model.blocks.image_block import ImageBlock
from docflow.model.blocks.equation_block import EquationBlock
from docflow.schema.models import BlockStyle, TextLineStyle


# 注册表项的类型别名：(区块类, 额外关键字参数)
_RegistryEntry = Tuple[Type[Block], Dict[str, Any]]


class BlockFactory:
    """从版面分析管线输出的原始字典实例化具体的 :class:`Block` 子类
    

    工厂维护一个注册表，将区块类型名字符串映射到
    ``(block_class, extra_kwargs)`` pairs.  Use :meth:`register` to extend
    注册表。
    """

    # -- 类级注册表 ------------------------------------------------

    _registry: Dict[str, _RegistryEntry] = {
        # 文本类区块
        "text": (TextBlock, {"block_type": BlockType.TEXT}),
        "title": (TextBlock, {"block_type": BlockType.TITLE}),
        "header": (TextBlock, {"block_type": BlockType.HEADER}),
        "footer": (TextBlock, {"block_type": BlockType.FOOTER}),
        "page_number": (TextBlock, {"block_type": BlockType.PAGE_NUMBER}),
        "reference": (TextBlock, {"block_type": BlockType.REFERENCE}),
        "abstract": (TextBlock, {"block_type": BlockType.ABSTRACT}),
        "footnote": (TextBlock, {"block_type": BlockType.FOOTNOTE}),
        "figure_caption": (TextBlock, {"block_type": BlockType.FIGURE_CAPTION}),
        "table_caption": (TextBlock, {"block_type": BlockType.TABLE_CAPTION}),
        "table_footnote": (TextBlock, {"block_type": BlockType.TABLE_FOOTNOTE}),
        "formula_caption": (TextBlock, {"block_type": BlockType.FORMULA_CAPTION}),
        "code": (TextBlock, {"block_type": BlockType.CODE}),
        "list": (TextBlock, {"block_type": BlockType.LIST}),
        # 非文本类区块
        "table": (TableBlock, {}),
        "figure": (ImageBlock, {}),
        "formula": (EquationBlock, {}),
        "equation": (EquationBlock, {}),  # 别名
    }

    # -- 公开 API ----------------------------------------------------------

    @classmethod
    def create(cls, block_dict: dict) -> Block:
        """从原始 *block_dict* 创建 :class:`Block`。

        *block_dict* 中的预期键：

        * ``category`` -- string label
        * ``bbox`` -- ``[x1, y1, x2, y2]``
        * ``confidence`` -- float (optional, default 1.0)

        类型特有键：

        * **text types**: ``text_lines`` -- list of dicts with ``text``,
          ``confidence``, and optional ``poly``.
        * **table**: ``html`` -- HTML string.
        * **figure**: ``image_base64`` *or* ``image_path``.
        * **formula**: ``latex``, ``image_base64`` *or* ``image_path``.
        """
        type_name: str = block_dict.get("category", "text").lower()
        entry: Optional[_RegistryEntry] = cls._registry.get(type_name)
        if entry is None:
            # 兜底为通用 TextBlock
            entry = (TextBlock, {"block_type": BlockType.TEXT})

        block_class, extra_kwargs = entry

        # -- 公共字段 ---------------------------------------------------
        raw_bbox = block_dict.get("bbox", [0, 0, 0, 0])
        bbox = BBox(
            x1=float(raw_bbox[0]),
            y1=float(raw_bbox[1]),
            x2=float(raw_bbox[2]),
            y2=float(raw_bbox[3]),
        )
        confidence = float(block_dict.get("confidence", 1.0))
        col_count = int(block_dict.get("col_count", 1))
        col_index = int(block_dict.get("col_index", 0))
        spanned_cols = block_dict.get("spanned_cols", [0])

        common_kwargs: Dict[str, Any] = {
            "bbox": bbox,
            "block_id": str(block_dict.get("id", "")),
            "confidence": confidence,
            "col_count": col_count,
            "col_index": col_index,
            "spanned_cols": list(spanned_cols),
            "attributes": block_dict.get("attributes"),
            "spans": list(block_dict.get("spans", []) or []),
        }
        common_kwargs.update(extra_kwargs)

        # -- 类型特有字段 -------------------------------------------
        if issubclass(block_class, TextBlock):
            lines = cls._build_text_lines(block_dict.get("text_lines", []))
            common_kwargs["lines"] = lines
            block_style = cls._parse_block_style(block_dict.get("style"))
            if block_style is not None:
                common_kwargs["style"] = block_style

        elif issubclass(block_class, TableBlock):
            common_kwargs["html"] = block_dict.get("html",
                                                    block_dict.get("table_html"))
            common_kwargs["image_data"] = cls._load_image_bytes(block_dict)

        elif issubclass(block_class, ImageBlock):
            common_kwargs["image_data"] = cls._load_image_bytes(block_dict)

        elif issubclass(block_class, EquationBlock):
            common_kwargs["latex"] = block_dict.get("latex")
            common_kwargs["image_data"] = cls._load_image_bytes(block_dict)

        return block_class(**common_kwargs)

    @classmethod
    def register(cls, type_name: str, block_class: Type[Block], **kwargs: Any) -> None:
        """注册（或覆盖）从 *type_name* 到 *block_class* 的映射。

        额外的 *kwargs* 将被存储并转发给构造函数
        whenever a block of this type is created.
        """
        cls._registry[type_name.lower()] = (block_class, kwargs)

    # -- 内部辅助方法 ----------------------------------------------------

    @staticmethod
    def _build_text_lines(raw_lines: List[dict]) -> List[TextLine]:
        """将原始字典列表转换为 :class:`TextLine` 对象。"""
        result: List[TextLine] = []
        for rl in raw_lines:
            text = rl.get("text", "")
            conf = float(rl.get("confidence", 1.0))
            region = rl.get("poly")
            line_style = BlockFactory._parse_text_line_style(rl.get("style"))
            result.append(TextLine(text=text, confidence=conf,
                                   text_region=region, style=line_style))
        return result

    @staticmethod
    def _parse_block_style(raw: Optional[dict]) -> Optional[BlockStyle]:
        """将 block.style 字典解析为 :class:`BlockStyle`，缺失则返回 None。"""
        if not raw or not isinstance(raw, dict):
            return None
        return BlockStyle(
            font_size_pt=raw.get("font_size_pt"),
            font_family=raw.get("font_family"),
            font_family_western=raw.get("font_family_western"),
            bold=raw.get("bold"),
            italic=raw.get("italic"),
            color=raw.get("color"),
            alignment=raw.get("alignment"),
            line_spacing=raw.get("line_spacing"),
            first_line_indent_pt=raw.get("first_line_indent_pt"),
            space_before_pt=raw.get("space_before_pt"),
            space_after_pt=raw.get("space_after_pt"),
        )

    @staticmethod
    def _parse_text_line_style(raw: Optional[dict]) -> Optional[TextLineStyle]:
        """将 text_line.style 字典解析为 :class:`TextLineStyle`，缺失则返回 None。"""
        if not raw or not isinstance(raw, dict):
            return None
        return TextLineStyle(
            font_size_pt=raw.get("font_size_pt"),
            font_family=raw.get("font_family"),
            font_family_western=raw.get("font_family_western"),
            bold=raw.get("bold"),
            italic=raw.get("italic"),
            underline=raw.get("underline"),
            strikethrough=raw.get("strikethrough"),
            color=raw.get("color"),
            background_color=raw.get("background_color"),
            superscript=raw.get("superscript"),
            subscript=raw.get("subscript"),
        )

    @staticmethod
    def _load_image_bytes(block_dict: dict) -> Optional[bytes]:
        """尝试解码 ``image_base64`` 或读取 ``image_path``。"""
        import re
        b64 = block_dict.get("image_base64")
        if b64:
            # 跳过占位符字符串，如 "<768636 chars>"
            if re.fullmatch(r'<\d+ chars>', b64):
                b64 = None
        if b64:
            # 修复缺失的 base64 填充
            missing = len(b64) % 4
            if missing:
                b64 += '=' * (4 - missing)
            try:
                data = base64.b64decode(b64)
                # 合理性检查：有效图片至少 ~100 字节
                if len(data) >= 100:
                    return data
            except Exception:
                pass
            return None

        img_path = block_dict.get("image_path")
        if img_path and os.path.isfile(img_path):
            with open(img_path, "rb") as fh:
                return fh.read()

        return None
