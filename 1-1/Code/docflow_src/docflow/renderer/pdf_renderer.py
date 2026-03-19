"""PDF 渲染器 —— 通过 DOCX + LibreOffice 转换。

生成临时 DOCX 然后通过 LibreOffice headless CLI 转换为 PDF。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING

from docflow.renderer.base import BaseRenderer
from docflow.renderer.docx_renderer import DocxRenderer

if TYPE_CHECKING:
    from docflow.model.page import Document


class PdfRenderer(BaseRenderer):
    """通过 DOCX + LibreOffice 将 :class:`Document` 渲染为 PDF。"""

    def __init__(self, config=None) -> None:
        super().__init__(config=config)
        self._docx_renderer = DocxRenderer(config=self.config)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def render(self, document: "Document", output_path: str, **options) -> None:
        """将 *document* 渲染为 PDF 文件并保存到 *output_path*。

        若未安装 LibreOffice 则抛出 :class:`RuntimeError`。
        """
        lo_path = self._find_libreoffice()
        if lo_path is None:
            raise RuntimeError(
                "LibreOffice is required for PDF rendering but was not found. "
                "Install it with: sudo apt install libreoffice  (Linux) or "
                "download from https://www.libreoffice.org/"
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_docx = os.path.join(tmp_dir, "output.docx")
            self._docx_renderer.render(document, tmp_docx, **options)

            subprocess.run(
                [
                    lo_path,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    tmp_dir,
                    tmp_docx,
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )

            tmp_pdf = os.path.join(tmp_dir, "output.pdf")
            if not os.path.isfile(tmp_pdf):
                raise RuntimeError(
                    "LibreOffice conversion completed but PDF was not produced."
                )

            shutil.copy2(tmp_pdf, output_path)

    def render_bytes(self, document: "Document", **options) -> bytes:
        """将 *document* 渲染为内存中的 PDF 字节。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_pdf = os.path.join(tmp_dir, "output.pdf")
            self.render(document, tmp_pdf, **options)
            with open(tmp_pdf, "rb") as fh:
                return fh.read()

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _find_libreoffice() -> str | None:
        """在系统上查找 LibreOffice 可执行文件。"""
        for name in ("libreoffice", "soffice"):
            path = shutil.which(name)
            if path is not None:
                return path
        return None
