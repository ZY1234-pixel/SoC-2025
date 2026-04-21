"""Markdown 文档渲染器。

Migrated from recovery_to_markdown.py.  Converts a :class:`Document` into
Markdown text, writing one file per document.
"""

from __future__ import annotations

import base64
import os
import re
from hashlib import sha1
from typing import TYPE_CHECKING

from docflow.model.base import Block, BlockType
from docflow.model.blocks.text_block import TextBlock, join_text_segments
from docflow.model.blocks.table_block import TableBlock
from docflow.model.blocks.image_block import ImageBlock
from docflow.model.blocks.equation_block import EquationBlock
from docflow.renderer.base import BaseRenderer

if TYPE_CHECKING:
    from docflow.model.page import Document

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency fallback
    BeautifulSoup = None


# 需要在 Markdown 正文中转义的字符
_MD_ESCAPE_CHARS = ("*", "`", "~", "$")


class MarkdownRenderer(BaseRenderer):
    """将 :class:`Document` 渲染为 Markdown 格式。"""

    def __init__(self, config=None) -> None:
        super().__init__(config=config)
        self._image_mode = "data_uri"
        self._asset_dir: str | None = None
        self._asset_prefix: str = ""
        self._image_seq = 0
        self._image_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def render(self, document: "Document", output_path: str, **options) -> None:
        """将 *document* 渲染为 Markdown 文件并保存到 *output_path*。"""
        image_mode = str(options.get("image_mode", "files")).lower()
        asset_dir = options.get("asset_dir")
        if image_mode == "files":
            if not asset_dir:
                root, _ = os.path.splitext(output_path)
                asset_dir = root + "_assets"
            os.makedirs(asset_dir, exist_ok=True)
            asset_prefix = str(options.get("asset_prefix") or os.path.basename(asset_dir))
        else:
            asset_dir = None
            asset_prefix = ""

        text = self._render_document_md(
            document=document,
            image_mode=image_mode,
            asset_dir=asset_dir,
            asset_prefix=asset_prefix,
        )
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def render_bytes(self, document: "Document", **options) -> bytes:
        """将 *document* 渲染为 UTF-8 编码的 Markdown 字节。"""
        image_mode = str(options.get("image_mode", "data_uri")).lower()
        text = self._render_document_md(
            document=document,
            image_mode=image_mode,
            asset_dir=None,
            asset_prefix="",
        )
        return text.encode("utf-8")

    def _render_document_md(
        self,
        document: "Document",
        image_mode: str,
        asset_dir: str | None,
        asset_prefix: str,
    ) -> str:
        self._image_mode = image_mode
        self._asset_dir = asset_dir
        self._asset_prefix = asset_prefix
        self._image_seq = 0
        self._image_cache = {}
        parts: list[str] = []

        for page in document.pages:
            for zone in page.zones:
                for block in zone.blocks:
                    md = self._render_block_md(block)
                    if md:
                        parts.append(md)

        text = "\n\n".join(parts)
        # 将三个或更多连续换行符压缩为两个
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    # ------------------------------------------------------------------
    # 逐区块渲染
    # ------------------------------------------------------------------

    def _render_block_md(self, block: Block) -> str:
        """根据区块类型分派到对应的渲染辅助方法。"""
        bt = block.block_type

        if bt == BlockType.TITLE:
            return self._render_title(block)

        if bt in (BlockType.TEXT, BlockType.REFERENCE):
            return self._render_text_block(block)

        if bt == BlockType.TABLE:
            return self._render_table(block)

        if bt == BlockType.FIGURE:
            return self._render_figure(block)

        if bt == BlockType.EQUATION:
            return self._render_equation(block)

        if bt in (BlockType.HEADER, BlockType.FOOTER):
            return ""

        if bt in (BlockType.TABLE_CAPTION, BlockType.FIGURE_CAPTION,
                  BlockType.TABLE_FOOTNOTE, BlockType.FORMULA_CAPTION):
            return self._render_caption(block)

        # 图片说明、表格说明或未知类型
        return self._render_generic(block)

    # ------------------------------------------------------------------
    # 类型特定渲染方法
    # ------------------------------------------------------------------

    def _render_title(self, block: TextBlock) -> str:
        text = self._collect_text(block)
        return "# " + text

    def _render_text_block(self, block: TextBlock) -> str:
        """渲染文本或参考文献区块。

        使用段落合并逻辑：如果区块已预计算段落，每个段落成为独立的
        Markdown 段落（以空行分隔）。否则直接拼接原始行。

        各段落内应用以下启发式：
        - **首行缩进法**: 若首行相对后续行向右缩进，则开始新段落。
        - **尾行法**: 若某行未填满区块宽度，视其末尾为段落边界。
        """
        if block.paragraphs:
            para_texts: list[str] = []
            for para in block.paragraphs:
                raw = para.text
                escaped = self._escape_md(raw)
                para_texts.append(escaped)
            return "\n\n".join(para_texts)

        # 回退：拼接所有行文本
        text = join_text_segments([line.text for line in block.lines]) if block.lines else ""
        return self._escape_md(text)

    @staticmethod
    def _clean_cell_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text.replace("|", "\\|")

    def _render_table(self, block: TableBlock) -> str:
        if block.html:
            md = self._html_table_to_markdown(block.html)
            if md:
                return md
        return ""

    def _render_figure(self, block: ImageBlock) -> str:
        if block.image_data:
            return self._render_image_markdown(block.image_data, "figure")
        return ""

    def _render_equation(self, block: EquationBlock) -> str:
        if block.latex:
            return f"$${block.latex}$$"
        if block.image_data:
            return self._render_image_markdown(block.image_data, "equation")
        return ""

    @staticmethod
    def _render_caption(block: Block) -> str:
        """渲染表格/图片/公式说明，各行间用换行分隔。"""
        if hasattr(block, "lines") and block.lines:
            return "  \n".join(line.text.strip() for line in block.lines if line.text.strip())
        return ""

    @staticmethod
    def _render_generic(block: Block) -> str:
        """渲染上述未处理的任意区块类型。"""
        if hasattr(block, "lines") and block.lines:
            return " ".join(line.text for line in block.lines)
        return ""

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_text(block: TextBlock) -> str:
        """拼接 *block* 中所有行的文本。"""
        if block.lines:
            return join_text_segments([line.text for line in block.lines])
        return ""

    @staticmethod
    def _escape_md(text: str) -> str:
        """转义 *text* 中的 Markdown 特殊字符。"""
        for ch in _MD_ESCAPE_CHARS:
            text = text.replace(ch, "\\" + ch)
        return text

    def _html_table_to_markdown(self, html: str) -> str:
        if not html:
            return ""
        if BeautifulSoup is None:
            plain = re.sub(r"<[^>]+>", " ", html)
            plain = re.sub(r"\s+", " ", plain).strip()
            return plain

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table is None:
            return ""

        pending: dict[int, int] = {}
        rows: list[list[str]] = []
        max_cols = 0

        for tr in table.find_all("tr"):
            row: list[str] = []
            col_idx = 0

            while pending.get(col_idx, 0) > 0:
                row.append("")
                pending[col_idx] -= 1
                if pending[col_idx] <= 0:
                    pending.pop(col_idx, None)
                col_idx += 1

            for cell in tr.find_all(["th", "td"], recursive=False):
                while pending.get(col_idx, 0) > 0:
                    row.append("")
                    pending[col_idx] -= 1
                    if pending[col_idx] <= 0:
                        pending.pop(col_idx, None)
                    col_idx += 1

                text = self._clean_cell_text(cell.get_text(" ", strip=True))
                colspan = max(1, int(cell.get("colspan", 1) or 1))
                rowspan = max(1, int(cell.get("rowspan", 1) or 1))

                row.append(text)
                for _ in range(colspan - 1):
                    row.append("")

                if rowspan > 1:
                    for off in range(colspan):
                        pending[col_idx + off] = max(pending.get(col_idx + off, 0), rowspan - 1)
                col_idx += colspan

            max_cols = max(max_cols, len(row))
            rows.append(row)

        if not rows:
            return ""

        max_cols = max(max_cols, max((len(r) for r in rows), default=0))
        rows = [r + [""] * (max_cols - len(r)) for r in rows]
        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * max_cols) + " |",
        ]
        for r in body:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)

    @staticmethod
    def _detect_image_format(image_data: bytes) -> tuple[str, str]:
        if image_data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png", "image/png"
        if image_data.startswith(b"\xff\xd8"):
            return "jpg", "image/jpeg"
        if image_data.startswith(b"GIF87a") or image_data.startswith(b"GIF89a"):
            return "gif", "image/gif"
        if image_data.startswith(b"RIFF") and image_data[8:12] == b"WEBP":
            return "webp", "image/webp"
        return "png", "image/png"

    def _render_image_markdown(self, image_data: bytes, alt: str) -> str:
        if not image_data:
            return ""

        digest = sha1(image_data).hexdigest()
        if digest in self._image_cache:
            return f"![{alt}]({self._image_cache[digest]})"

        ext, mime = self._detect_image_format(image_data)

        if self._image_mode == "files" and self._asset_dir:
            self._image_seq += 1
            filename = f"image_{self._image_seq:04d}.{ext}"
            abs_path = os.path.join(self._asset_dir, filename)
            with open(abs_path, "wb") as fh:
                fh.write(image_data)
            rel = f"{self._asset_prefix}/{filename}" if self._asset_prefix else filename
            self._image_cache[digest] = rel
            return f"![{alt}]({rel})"

        b64 = base64.b64encode(image_data).decode("ascii")
        uri = f"data:{mime};base64,{b64}"
        self._image_cache[digest] = uri
        return f"![{alt}]({uri})"
