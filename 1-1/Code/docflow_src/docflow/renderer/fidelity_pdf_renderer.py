"""高保真 PDF：保留原始页面视觉，叠加不可见文本层。"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import fitz
from bs4 import BeautifulSoup

from docflow.model.stages import INLINE_FORMULA_MARKER_PATTERN, Rect


class FidelityPdfRenderer:
    """生成视觉上接近原页的可搜索 PDF。"""

    def render(self, analysis, page_images, output_path: str) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        document = fitz.open()
        try:
            for page_index, page in enumerate(analysis.pages):
                source = page_images[page_index]
                if isinstance(source, tuple):
                    source = source[1]
                height_px, width_px = source.shape[:2]
                scale = 841.89 / max(width_px, height_px)
                pdf_page = document.new_page(width=width_px * scale, height=height_px * scale)
                self._insert_source_image(pdf_page, source)
                self._insert_searchable_text(pdf_page, page, scale)
            document.save(output, deflate=True, garbage=4)
        finally:
            document.close()

    @staticmethod
    def _insert_source_image(page, image) -> None:
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        if not ok:
            raise RuntimeError("无法编码高保真 PDF 页面图像")
        page.insert_image(page.rect, stream=encoded.tobytes(), keep_proportion=False)

    def _insert_searchable_text(self, page, analysis_page, scale: float) -> None:
        for element in analysis_page.elements:
            for row in element.text_rows:
                text = INLINE_FORMULA_MARKER_PATTERN.sub("", row.text).strip()
                if text:
                    self._insert_text(page, row.bbox, text, scale)
            caption = str(element.payload.get("caption") or "").strip()
            if caption:
                self._insert_text(page, self._caption_bbox(element), caption, scale)
            latex = str(element.payload.get("latex") or "").strip()
            if latex:
                self._insert_text(page, element.bbox, latex, scale)
            for formula in element.payload.get("inline_formulas") or ():
                bbox = formula.get("bbox")
                latex = str(formula.get("latex") or "").strip()
                if bbox and latex:
                    self._insert_text(page, Rect.from_sequence(bbox), latex, scale)
            if element.kind == "table_group" and element.payload.get("html"):
                self._insert_table_text(page, str(element.payload["html"]), scale)

    @staticmethod
    def _caption_bbox(element) -> Rect:
        primary_value = element.payload.get("primary_bbox")
        primary = Rect.from_sequence(primary_value) if primary_value else element.bbox
        bbox = element.bbox
        if element.payload.get("caption_position") == "before":
            return Rect(bbox.x1, bbox.y1, bbox.x2, primary.y1)
        return Rect(bbox.x1, primary.y2, bbox.x2, bbox.y2)

    def _insert_table_text(self, page, html: str, scale: float) -> None:
        soup = BeautifulSoup(html, "html.parser")
        for cell in soup.find_all(["td", "th"]):
            bbox = cell.get("data-page-bbox")
            text = cell.get_text(" ", strip=True)
            if bbox and text:
                self._insert_bbox_text(page, bbox, text, scale)
            for formula in cell.find_all("span", class_="docflow-formula"):
                self._insert_bbox_text(
                    page,
                    formula.get("data-page-bbox"),
                    str(formula.get("data-latex") or "").strip(),
                    scale,
                )

    def _insert_bbox_text(self, page, bbox: str | None, text: str, scale: float) -> None:
        if not bbox or not text:
            return
        values = str(bbox).split(",")
        if len(values) != 4:
            return
        try:
            rect = Rect(*(float(value) for value in values))
        except ValueError:
            return
        self._insert_text(page, rect, text, scale)

    @staticmethod
    def _insert_text(page, rect: Rect, text: str, scale: float) -> None:
        target = fitz.Rect(rect.x1 * scale, rect.y1 * scale, rect.x2 * scale, rect.y2 * scale)
        if target.width <= 0 or target.height <= 0:
            return
        text = " ".join(text.split())
        cjk = any("\u3400" <= char <= "\u9fff" for char in text)
        font = fitz.Font("china-s" if cjk else "tiro")
        font_size = max(min(target.height * 0.72, 48.0), 3.0)
        natural_width = font.text_length(text, font_size)
        if natural_width <= 0:
            return
        x_scale = target.width / natural_width
        baseline = target.y1 - 0.15 * font_size
        page.insert_text(
            (target.x0, baseline),
            text,
            fontname="china-s" if cjk else "tiro",
            fontsize=font_size,
            render_mode=3,
            morph=(fitz.Point(target.x0, baseline), (x_scale, 0, 0, 1, 0, 0)),
        )
