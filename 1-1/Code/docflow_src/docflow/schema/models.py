"""标准版面分析输出格式 — v2.0 schema models.

Defines the data model for a layout-analysis-engine-agnostic intermediate
representation.  Covers block-level categories (text, title, table, figure,
formula …), span-level inline elements, reading order, and inter-block
relations.

Runtime validation: :mod:`docflow.schema.validator`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# 区块类别枚举
# ═══════════════════════════════════════════════════════════════════════

class BlockCategory(enum.Enum):
    """区块级版面类别（对齐 OmniDocBench + 扩展）。"""

    TEXT = "text"
    TITLE = "title"
    TABLE = "table"
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    TABLE_CAPTION = "table_caption"
    TABLE_FOOTNOTE = "table_footnote"
    FORMULA = "formula"
    FORMULA_CAPTION = "formula_caption"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    REFERENCE = "reference"
    ABSTRACT = "abstract"
    CODE = "code"
    LIST = "list"
    FOOTNOTE = "footnote"
    WATERMARK = "watermark"
    ABANDON = "abandon"


class SpanCategory(enum.Enum):
    """行内（Span 级）元素类别。"""

    TEXT_LINE = "text_line"
    INLINE_FORMULA = "inline_formula"
    SUPERSCRIPT = "superscript"
    SUBSCRIPT = "subscript"


class RelationType(enum.Enum):
    """区块间的语义关系类型。"""

    TRUNCATED = "truncated"
    CAPTION_OF = "caption_of"
    FOOTNOTE_OF = "footnote_of"
    FORMULA_LABEL_OF = "formula_label_of"


class DocType(enum.Enum):
    """文档类型页面属性。"""

    ACADEMIC = "academic"
    NEWSPAPER = "newspaper"
    BOOK = "book"
    MAGAZINE = "magazine"
    REPORT = "report"
    SLIDES = "slides"
    NOTES = "notes"
    EXAM = "exam"
    OTHER = "other"


class LayoutType(enum.Enum):
    """页面版面类型属性。"""

    SINGLE_COLUMN = "single_column"
    DOUBLE_COLUMN = "double_column"
    MULTI_COLUMN = "multi_column"
    COMPLEX = "complex"


# 用于快速成员检测的辅助集合
BLOCK_CATEGORIES: frozenset[str] = frozenset(c.value for c in BlockCategory)
SPAN_CATEGORIES: frozenset[str] = frozenset(c.value for c in SpanCategory)
RELATION_TYPES: frozenset[str] = frozenset(r.value for r in RelationType)

# 携带文本内容的类别
TEXT_CATEGORIES: frozenset[str] = frozenset({
    "text", "title", "header", "footer", "page_number",
    "reference", "abstract", "code", "list", "footnote",
    "figure_caption", "table_caption", "table_footnote",
    "formula_caption",
})

# 在下游恢复中以图像方式渲染的类别
FIGURE_CATEGORIES: frozenset[str] = frozenset({
    "figure", "formula",
})


# ═══════════════════════════════════════════════════════════════════════════
# 样式模型 (v2.0)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TextLineStyle:
    """单行文本的字符/Run 级别样式。

    所有字段可选 —— 省略的值从父级 block.style 继承，
    再从 page.style_defaults，最后从渲染器配置。
    """

    font_size_pt: Optional[float] = None
    font_family: Optional[str] = None          # CJK 字体名
    font_family_western: Optional[str] = None  # 西文字体名
    bold: Optional[bool] = None
    italic: Optional[bool] = None
    underline: Optional[bool] = None
    strikethrough: Optional[bool] = None
    color: Optional[str] = None                # CSS 十六进制, 如 "#FF0000"
    background_color: Optional[str] = None
    superscript: Optional[bool] = None
    subscript: Optional[bool] = None


@dataclass
class BlockStyle:
    """版面区块的段落级样式。

    所有字段可选 —— 省略的值从
    page.style_defaults 继承，再从渲染器配置。
    """

    font_size_pt: Optional[float] = None
    font_family: Optional[str] = None
    font_family_western: Optional[str] = None
    bold: Optional[bool] = None
    italic: Optional[bool] = None
    color: Optional[str] = None
    alignment: Optional[str] = None            # left / center / right / justify
    line_spacing: Optional[float] = None       # 倍数
    first_line_indent_pt: Optional[float] = None
    space_before_pt: Optional[float] = None
    space_after_pt: Optional[float] = None


@dataclass
class PageStyleDefaults:
    """页面级基线样式 —— 页面上“最常见”的格式。

    对于电子 PDF，通过统计各字符的众数来派生。
    对于扫描文档，留空以回退到渲染器配置。
    """

    font_size_pt: Optional[float] = None
    font_family: Optional[str] = None
    font_family_western: Optional[str] = None
    line_spacing: Optional[float] = None
    paragraph_spacing_before_pt: Optional[float] = None
    paragraph_spacing_after_pt: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════
# 数据模型 (v2.0)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TextLine:
    """单行 OCR 识别文本。"""

    text: str
    confidence: float = 1.0
    poly: Optional[List[List[float]]] = None
    style: Optional[TextLineStyle] = None


@dataclass
class Span:
    """区块内的行内元素（公式、上标等）。"""

    id: str
    category: str  # one of SpanCategory values
    bbox: List[float]  # [x1, y1, x2, y2]
    poly: Optional[List[List[float]]] = None
    text: Optional[str] = None
    latex: Optional[str] = None
    confidence: float = 1.0


@dataclass
class Block:
    """页面上的单个版面区块。"""

    id: str
    category: str  # one of BlockCategory values
    bbox: List[float]  # [x1, y1, x2, y2]

    poly: Optional[List[List[float]]] = None
    confidence: float = 1.0
    order: Optional[int] = None

    # 内容字段（根据类别填充）
    text: Optional[str] = None
    text_lines: Optional[List[TextLine]] = None
    html: Optional[str] = None
    latex: Optional[str] = None

    # 图像内容
    image_path: Optional[str] = None
    image_base64: Optional[str] = None

    # 类别特定属性和行内 Span
    attributes: Optional[Dict[str, Any]] = None
    spans: Optional[List[Span]] = None

    # 样式
    style: Optional[BlockStyle] = None


@dataclass
class Relation:
    """两个区块之间的有向语义关系。"""

    type: str  # one of RelationType values
    source_id: str
    target_id: str


@dataclass
class PageAttributes:
    """可选的页面级描述属性。"""

    doc_type: Optional[str] = None
    layout_type: Optional[str] = None
    language: Optional[str] = None
    is_scanned: Optional[bool] = None


@dataclass
class Page:
    """文档中的单个页面。"""

    page_index: int
    width: int
    height: int
    blocks: List[Block] = field(default_factory=list)

    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    attributes: Optional[PageAttributes] = None
    relations: Optional[List[Relation]] = None
    style_defaults: Optional[PageStyleDefaults] = None


@dataclass
class Metadata:
    """文档级元数据。"""

    engine: Optional[str] = None
    engine_version: Optional[str] = None
    created_at: Optional[str] = None
    source_file: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class Document:
    """顶层文档 —— v2.0 Schema 的根对象。"""

    version: str = "2.0"
    pages: List[Page] = field(default_factory=list)
    metadata: Optional[Metadata] = None



