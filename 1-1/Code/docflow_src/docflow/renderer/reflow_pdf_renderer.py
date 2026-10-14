"""Direct source-coordinate PDF rendering without a DOCX intermediate."""

from __future__ import annotations

import base64
import html
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import median

import fitz
from bs4 import BeautifulSoup

from docflow.model.stages import INLINE_FORMULA_MARKER_PATTERN, Rect, TextRow
from docflow.renderer.docx_utils.html_table import (
    get_table_cell_content_nodes,
    get_table_cell_placements,
    get_table_cell_text_lines,
    get_table_dimensions,
    has_table_cell_paragraphs,
)
from docflow.renderer.formula_mathjax import MathJaxFormulaRenderer


_TEXT_KINDS = {"heading", "paragraph_group", "caption"}


class ReflowPdfRenderer:
    """Place analyzed elements back at their source page coordinates."""

    def render(self, analysis, page_images, output_path: str) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        document = fitz.open()
        self._drawn_inline_formulas = set()
        try:
            for page_index, page in enumerate(analysis.pages):
                source = page_images[page_index]
                if isinstance(source, tuple):
                    source = source[1]
                height_px, width_px = source.shape[:2]
                scale = 841.89 / max(width_px, height_px)
                pdf_page = document.new_page(width=width_px * scale, height=height_px * scale)
                roles = {role.role_id: role for role in analysis.roles}
                font_sizes = self._text_font_sizes(page, scale)
                for element in sorted(page.elements, key=lambda item: item.model_order):
                    self._draw_element(
                        pdf_page,
                        element,
                        roles.get(element.role_id),
                        scale,
                        font_sizes.get(element.element_id),
                    )
                for element in page.elements:
                    if element.kind in _TEXT_KINDS:
                        self._draw_inline_formulas(pdf_page, element, scale)
            document.save(output, deflate=True, garbage=4)
        finally:
            document.close()

    def _draw_element(
        self,
        page,
        element,
        role,
        scale: float,
        font_size: float | None = None,
    ) -> None:
        background = element.payload.get("background_color")
        if background:
            page.draw_rect(
                self._pdf_rect(element.bbox, scale),
                color=None,
                fill=self._pdf_color(background),
            )
        if element.kind in _TEXT_KINDS:
            self._draw_text(page, element, role, scale, font_size)
            return
        if element.kind == "table_group" and element.payload.get("html") and self._draw_table(
            page,
            element,
            role,
            scale,
        ):
            self._draw_caption(page, element, role, scale)
            return
        if element.kind == "equation_group" and element.payload.get("latex"):
            if self._draw_formula(page, element.payload["latex"], element.bbox, scale):
                self._draw_caption(page, element, role, scale)
                return
        bbox_value = element.payload.get("primary_bbox") or element.content_bbox or element.bbox
        bbox = bbox_value if isinstance(bbox_value, Rect) else Rect.from_sequence(bbox_value)
        data = self._decode_image(element.payload.get("image_base64"))
        if data:
            page.insert_image(
                self._pdf_rect(bbox, scale),
                stream=data,
                keep_proportion=False,
            )
        self._draw_caption(page, element, role, scale)

    def _draw_text(
        self,
        page,
        element,
        role,
        scale: float,
        font_size: float | None = None,
    ) -> None:
        rows = element.text_rows or (
            TextRow(
                element.text,
                element.content_bbox or element.bbox,
            ),
        )
        alignments = getattr(element, "row_alignments", ()) or (
            str(element.payload.get("alignment") or "left"),
        ) * len(rows)
        last_text_index = max((index for index, row in enumerate(rows) if row.text.strip()), default=-1)
        right_edge = max((row.bbox.x2 * scale for row in rows if row.text.strip()), default=element.bbox.x2 * scale)
        for index, row in enumerate(rows):
            alignment = alignments[index] if index < len(alignments) else "left"
            has_inline_formula = bool(INLINE_FORMULA_MARKER_PATTERN.search(row.text))
            justify = element.kind == "paragraph_group" and index != last_text_index and not has_inline_formula
            rect = (
                fitz.Rect(row.bbox.x1 * scale, row.bbox.y1 * scale, right_edge, row.bbox.y2 * scale)
                if justify
                else self._pdf_rect(row.bbox, scale)
            )
            rendered_size = font_size or max(min(rect.height * 0.78, 48.0), 3.0)
            color = role.color if role else "#000000"
            css = (
                "div{"
                f"font-family:{self._font_family(role)};"
                f"font-size:{rendered_size}pt;line-height:{rendered_size * 1.05}pt;"
                f"color:{color};white-space:pre-wrap;overflow-wrap:anywhere;"
                f"text-align:{alignment};text-align-last:{'justify' if justify else alignment};"
                "}"
            )
            if has_inline_formula:
                items = self._line_items(row, element, scale)
                if index != last_text_index:
                    self._draw_justified_line(page, items, rect, role, rendered_size, element, scale)
                else:
                    self._draw_inline_text_row(page, row, element, role, scale, css, rendered_size)
                continue
            if justify:
                self._draw_justified_line(
                    page,
                    [{"text": token} for token in row.text.split()],
                    rect,
                    role,
                    rendered_size,
                    element,
                    scale,
                )
            else:
                page.insert_htmlbox(
                    rect,
                    f"<div>{html.escape(row.text)}</div>",
                    css=css,
                    scale_low=0.10,
                )

    @staticmethod
    def _text_font_sizes(page, scale: float) -> dict[str, float]:
        raw = {}
        by_role = defaultdict(list)
        paragraph_sizes = []
        for element in page.elements:
            if element.kind not in _TEXT_KINDS:
                continue
            heights = [row.bbox.height * scale * 0.78 for row in element.text_rows if row.text.strip()]
            if not heights:
                bbox = element.content_bbox or element.bbox
                heights = [bbox.height * scale * 0.78]
            size = median(heights)
            raw[element.element_id] = size
            by_role[element.role_id].append(size)
            if element.kind == "paragraph_group":
                paragraph_sizes.append(size)
        normalized = {role_id: median(values) for role_id, values in by_role.items() if values}
        paragraph_size = median(paragraph_sizes) if paragraph_sizes else None
        return {
            element.element_id: (
                paragraph_size
                if element.kind == "paragraph_group" and paragraph_size is not None
                else normalized.get(element.role_id, raw[element.element_id])
            )
            for element in page.elements
            if element.element_id in raw
        }

    def _draw_inline_formulas(self, page, element, scale: float) -> None:
        for index, formula in enumerate(element.payload.get("inline_formulas") or ()):
            if (element.element_id, index) in self._drawn_inline_formulas:
                continue
            bbox = formula.get("bbox")
            if bbox and formula.get("latex") and self._draw_formula(
                page,
                formula["latex"],
                Rect.from_sequence(bbox),
                scale,
            ):
                continue
            data = self._decode_image(formula.get("image_base64"))
            if bbox and data:
                page.insert_image(self._pdf_rect(Rect.from_sequence(bbox), scale), stream=data, keep_proportion=False)

    def _draw_inline_text_row(
        self,
        page,
        row,
        element,
        role,
        scale: float,
        css: str,
        font_size: float,
    ) -> None:
        formulas = element.payload.get("inline_formulas") or ()
        top = row.bbox.y1 * scale
        bottom = row.bbox.y2 * scale
        right = row.bbox.x2 * scale
        cursor = row.bbox.x1 * scale
        gap = max(1.4, font_size * 0.18)

        position = 0
        after_formula = False
        for match in INLINE_FORMULA_MARKER_PATTERN.finditer(row.text):
            text = row.text[position : match.start()]
            if after_formula:
                text = text.lstrip()
            text = text.rstrip()
            if text.strip():
                text_rect = fitz.Rect(cursor, top, right, bottom)
                page.insert_htmlbox(
                    text_rect,
                    f"<div>{html.escape(text)}</div>",
                    css=css,
                    scale_low=0.10,
                )
                cursor = self._text_end_in_rect(page, text_rect, cursor) + gap
            index = int(match.group(1))
            formula = formulas[index] if index < len(formulas) else {}
            bbox = formula.get("bbox")
            if bbox:
                width = (float(bbox[2]) - float(bbox[0])) * scale
                formula_rect = fitz.Rect(
                    cursor,
                    float(bbox[1]) * scale,
                    cursor + width,
                    float(bbox[3]) * scale,
                )
                key = (element.element_id, index)
                self._drawn_inline_formulas.add(key)
                drawn_rect = None
                if formula.get("latex"):
                    drawn_rect = self._draw_formula_rect(page, formula["latex"], formula_rect, align="left")
                if drawn_rect is None:
                    data = self._decode_image(formula.get("image_base64"))
                    if data:
                        page.insert_image(formula_rect, stream=data, keep_proportion=False)
                cursor = (drawn_rect or formula_rect).x1 + gap
                after_formula = True
            position = match.end()
        text = row.text[position:]
        if after_formula:
            text = text.lstrip()
        if text.strip():
            tail_rect = fitz.Rect(cursor, top, right, bottom)
            page.insert_htmlbox(
                tail_rect,
                f"<div>{html.escape(text)}</div>",
                css=css,
                scale_low=0.10,
            )

    @classmethod
    def _line_items(cls, row, element, scale: float) -> list[dict]:
        items = []
        position = 0
        after_formula = False
        for match in INLINE_FORMULA_MARKER_PATTERN.finditer(row.text):
            text = row.text[position : match.start()]
            if after_formula:
                text = text.lstrip()
            items.extend({"text": token} for token in text.rstrip().split())
            index = int(match.group(1))
            formula = (element.payload.get("inline_formulas") or ())[index]
            items.append(
                {
                    "formula": formula,
                    "formula_index": index,
                    "bbox": Rect.from_sequence(formula["bbox"]),
                    "scale": scale,
                }
            )
            after_formula = True
            position = match.end()
        text = row.text[position:]
        if after_formula:
            text = text.lstrip()
        items.extend({"text": token} for token in text.split())
        return items

    def _draw_justified_line(self, page, items, rect, role, font_size: float, element, scale: float) -> None:
        if not items:
            return
        font = self._justify_font(" ".join(str(item.get("text") or "") for item in items), role)
        widths = []
        formula_rects = []
        for item in items:
            if "text" in item:
                widths.append(font.text_length(str(item["text"]), font_size))
                formula_rects.append(None)
            else:
                formula_rect = self._formula_target_rect(item["formula"], item["bbox"], item["scale"], rect.x0)
                widths.append(formula_rect.width)
                formula_rects.append(formula_rect)
        gap = max(0.0, (rect.width - sum(widths)) / max(len(items) - 1, 1))
        writer = fitz.TextWriter(page.rect)
        baseline = rect.y1 + font.descender * font_size
        x = rect.x0
        for item, width, formula_rect in zip(items, widths, formula_rects):
            if "text" in item:
                writer.append((x, baseline), str(item["text"]), font=font, fontsize=font_size)
            else:
                target = fitz.Rect(x, formula_rect.y0, x + width, formula_rect.y1)
                formula = item["formula"]
                self._drawn_inline_formulas.add((element.element_id, item["formula_index"]))
                drawn = None
                if formula.get("latex"):
                    drawn = self._draw_formula_rect(page, formula["latex"], target, align="left")
                if drawn is None:
                    data = self._decode_image(formula.get("image_base64"))
                    if data:
                        page.insert_image(target, stream=data, keep_proportion=False)
            x += width + gap
        writer.write_text(page, color=self._pdf_color(getattr(role, "color", "#000000")))

    @classmethod
    def _formula_target_rect(cls, formula, bbox: Rect, scale: float, x: float) -> fitz.Rect:
        target = cls._pdf_rect(bbox, scale)
        size = cls._formula_content_size(str(formula.get("latex") or ""))
        if not size or size[0] <= 0 or size[1] <= 0:
            return fitz.Rect(x, target.y0, x + target.width, target.y1)
        fit = min(target.width / size[0], target.height / size[1])
        width, height = size[0] * fit, size[1] * fit
        y0 = target.y0 + (target.height - height) / 2.0
        return fitz.Rect(x, y0, x + width, y0 + height)

    @classmethod
    @lru_cache(maxsize=512)
    def _formula_content_size(cls, latex: str) -> tuple[float, float] | None:
        pdf = MathJaxFormulaRenderer.render_pdf(latex)
        if not pdf:
            return None
        with fitz.open(stream=pdf, filetype="pdf") as formula:
            drawings = [drawing["rect"] for drawing in formula[0].get_drawings()]
        if not drawings:
            return None
        content = drawings[0]
        for drawing_rect in drawings[1:]:
            content |= drawing_rect
        return float(content.width), float(content.height)

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return any("\u3400" <= char <= "\u9fff" for char in text)

    @classmethod
    def _justify_font(cls, text: str, role):
        if cls._contains_cjk(text):
            return fitz.Font("china-s")
        bold = bool(getattr(role, "bold", False))
        italic = bool(getattr(role, "italic", False))
        if str(getattr(role, "font_family", "")) == "黑体":
            return fitz.Font(("hebi" if italic else "hebo") if bold else ("heit" if italic else "helv"))
        return fitz.Font(("tibi" if italic else "tibo") if bold else ("tiit" if italic else "tiro"))

    @staticmethod
    def _text_end_in_rect(page, rect: fitz.Rect, start_x: float) -> float:
        spans = [
            span
            for block in page.get_text("dict", clip=rect).get("blocks", ())
            for line in block.get("lines", ())
            for span in line.get("spans", ())
            if span.get("text") and span["bbox"][2] >= start_x - 1.0
        ]
        return max((float(span["bbox"][2]) for span in spans), default=start_x)

    def _draw_table(self, page, element, role, scale: float) -> bool:
        soup = BeautifulSoup(str(element.payload["html"]), "html.parser")
        table = soup.find("table")
        if table is None:
            return False
        rows, columns = get_table_dimensions(table)
        if not rows or not columns:
            return False
        formula_nodes = [
            node
            for cell in table.find_all(["td", "th"])
            for node in cell.find_all("span", class_="docflow-formula")
        ]
        formulas = [str(node.get("data-latex") or "") for node in formula_nodes]
        if any(
            not latex
            or not node.get("data-page-bbox")
            or MathJaxFormulaRenderer.render_pdf(latex) is None
            for node, latex in zip(formula_nodes, formulas)
        ):
            return False
        primary_value = element.payload.get("primary_bbox")
        table_bbox = Rect.from_sequence(primary_value) if primary_value else element.bbox
        placements = get_table_cell_placements(table)
        y_edges = self._table_axis_edges(table_bbox, placements, rows, "y")
        x_edges = self._table_axis_edges(table_bbox, placements, columns, "x")
        row_styles = {
            int(row): (fill, color)
            for row, fill, color in element.payload.get("table_row_styles", ())
        }
        cell_styles = {
            (int(row), int(column)): (str(alignment), bool(bold))
            for row, column, alignment, bold in element.payload.get("table_cell_styles", ())
        }
        rule_style = str(element.payload.get("table_rule_style") or "grid")
        for row, column, row_span, column_span, source in placements:
            rect = Rect(x_edges[column], y_edges[row], x_edges[column + column_span], y_edges[row + row_span])
            pdf_rect = self._pdf_rect(rect, scale)
            fill = row_styles.get(row, (None, None))[0]
            if fill:
                page.draw_rect(pdf_rect, color=None, fill=self._pdf_color(fill))
            if rule_style == "grid":
                page.draw_rect(pdf_rect, color=(0.65, 0.68, 0.72), width=0.5)
            elif rule_style == "horizontal" and row == 0:
                page.draw_line((pdf_rect.x0, pdf_rect.y0), (pdf_rect.x1, pdf_rect.y0), color=(0.65, 0.68, 0.72), width=0.5)
            if not self._draw_table_cell(
                page,
                source,
                pdf_rect,
                role,
                cell_styles.get((row, column), ("left", False)),
                row_styles.get(row, (None, None))[1],
                scale,
            ):
                return False
        return True

    def _draw_table_cell(
        self,
        page,
        source,
        rect,
        role,
        style,
        text_color=None,
        scale: float = 1.0,
    ) -> bool:
        if has_table_cell_paragraphs(source):
            return self._draw_annotated_table_cell(page, source, rect, role, style, text_color, scale)
        alignment, bold = style
        text = "\n".join(get_table_cell_text_lines(source))
        image_nodes = source.find_all("img", src=True)[:4]
        formula_nodes = source.find_all("span", class_="docflow-formula")
        media_height = rect.height * (0.62 if text else 0.90)
        cursor = rect.y0
        for node in image_nodes:
            data = self._decode_image(node.get("src"))
            if data:
                image_rect = fitz.Rect(rect.x0, cursor, rect.x1, min(cursor + media_height, rect.y1))
                page.insert_image(image_rect, stream=data, keep_proportion=True)
                cursor = image_rect.y1
        for node in formula_nodes:
            latex = str(node.get("data-latex") or "")
            page_bbox = str(node.get("data-page-bbox") or "")
            try:
                bbox = [float(value) for value in page_bbox.split(",")]
            except ValueError:
                return False
            if len(bbox) != 4:
                return False
            formula_rect = self._pdf_rect(Rect.from_sequence(bbox), scale)
            if not self._draw_formula_rect(page, latex, formula_rect):
                return False
            cursor = formula_rect.y1
        if text:
            text_rect = fitz.Rect(rect.x0 + 1, cursor, rect.x1 - 1, rect.y1)
            font_size = max(min(text_rect.height / max(text.count("\n") + 1, 1) * 0.72, 18.0), 3.0)
            page.insert_htmlbox(
                text_rect,
                f"<div><{'b' if bold else 'span'}>{html.escape(text)}</{'b' if bold else 'span'}></div>",
                css=self._text_css(role, font_size, alignment, text_color),
                scale_low=0.10,
            )
        return True

    def _draw_annotated_table_cell(self, page, source, rect, role, style, text_color=None, scale: float = 1.0) -> bool:
        _alignment, bold = style
        nodes = get_table_cell_content_nodes(source)
        text_lines = sum(
            max(1, len(node.get_text("\n", strip=True).splitlines()))
            for node in nodes
            if node.name == "p"
        )
        font_size = max(min(rect.height / max(text_lines, 1) * 0.72, 18.0), 3.0)
        cursor = rect.y0
        for node in nodes:
            if node.name == "p":
                text = node.get_text("\n", strip=True)
                if not text:
                    continue
                try:
                    space_before = float(node.get("data-space-before") or 0.0)
                except (TypeError, ValueError):
                    space_before = 0.0
                lines = max(1, len(text.splitlines()))
                block_height = (lines + max(space_before, 0.0)) * font_size * 1.2
                text_rect = fitz.Rect(rect.x0 + 1, cursor, rect.x1 - 1, min(cursor + block_height, rect.y1))
                if text_rect.height <= 1:
                    break
                page.insert_htmlbox(
                    text_rect,
                    f"<div><{'b' if bold else 'span'}>{html.escape(text).replace(chr(10), '<br/>')}</{'b' if bold else 'span'}></div>",
                    css=self._text_css(role, font_size, str(node.get("data-align") or "left"), text_color),
                    scale_low=0.10,
                )
                cursor = text_rect.y1
                continue
            if node.name == "img":
                data = self._decode_image(node.get("src"))
                if data:
                    image_rect = fitz.Rect(rect.x0, cursor, rect.x1, min(cursor + rect.height * 0.42, rect.y1))
                    page.insert_image(image_rect, stream=data, keep_proportion=True)
                    cursor = image_rect.y1
                continue
            latex = str(node.get("data-latex") or "")
            page_bbox = str(node.get("data-page-bbox") or "")
            try:
                bbox = [float(value) for value in page_bbox.split(",")]
            except ValueError:
                return False
            if len(bbox) != 4:
                return False
            formula_rect = self._pdf_rect(Rect.from_sequence(bbox), scale)
            if not self._draw_formula_rect(page, latex, formula_rect):
                return False
            cursor = max(cursor, formula_rect.y1)
        return True

    def _draw_formula(self, page, latex: str, source_rect: Rect, scale: float) -> bool:
        return self._draw_formula_rect(page, str(latex), self._pdf_rect(source_rect, scale))

    @staticmethod
    def _draw_formula_rect(page, latex: str, rect: fitz.Rect, align: str = "center") -> fitz.Rect | None:
        pdf = MathJaxFormulaRenderer.render_pdf(latex)
        if not pdf:
            return None
        try:
            with fitz.open(stream=pdf, filetype="pdf") as formula:
                source_page = formula[0]
                source = source_page.rect
                drawings = [drawing["rect"] for drawing in source_page.get_drawings()]
                content = drawings[0]
                for drawing_rect in drawings[1:]:
                    content |= drawing_rect
                scale = min(rect.width / max(content.width, 0.01), rect.height / max(content.height, 0.01))
                width, height = content.width * scale, content.height * scale
                x0 = rect.x0 if align == "left" else rect.x0 + (rect.width - width) / 2.0
                y0 = rect.y0 + (rect.height - height) / 2.0
                fitted = fitz.Rect(
                    x0 - content.x0 * scale,
                    y0 - content.y0 * scale,
                    x0 - content.x0 * scale + source.width * scale,
                    y0 - content.y0 * scale + source.height * scale,
                )
                page.show_pdf_page(fitted, formula, 0, keep_proportion=False)
            return fitz.Rect(x0, y0, x0 + width, y0 + height)
        except Exception:
            return None


    def _draw_caption(self, page, element, role, scale: float) -> None:
        caption = str(element.payload.get("caption") or "").strip()
        if not caption:
            return
        primary_value = element.payload.get("primary_bbox")
        primary = Rect.from_sequence(primary_value) if primary_value else element.bbox
        bbox = element.bbox
        rect = (
            Rect(bbox.x1, bbox.y1, bbox.x2, primary.y1)
            if element.payload.get("caption_position") == "before"
            else Rect(bbox.x1, primary.y2, bbox.x2, bbox.y2)
        )
        if rect.height <= 1:
            return
        font_size = max(min(rect.height * scale * 0.72, 18.0), 3.0)
        page.insert_textbox(
            self._pdf_rect(rect, scale),
            caption,
            fontname="china-s",
            fontsize=font_size,
            color=self._pdf_color(role.color if role else "#000000"),
            align=self._alignment(element.payload.get("caption_alignment")),
        )

    @staticmethod
    def _pdf_rect(rect: Rect, scale: float) -> fitz.Rect:
        return fitz.Rect(rect.x1 * scale, rect.y1 * scale, rect.x2 * scale, rect.y2 * scale)

    @staticmethod
    def _decode_image(value) -> bytes | None:
        try:
            if isinstance(value, str) and value.startswith("data:"):
                value = value.split(",", 1)[1]
            return base64.b64decode(value) if value else None
        except (IndexError, ValueError, TypeError):
            return None

    @staticmethod
    def _pdf_color(value):
        text = str(value or "#000000").lstrip("#")
        try:
            return tuple(int(text[index : index + 2], 16) / 255.0 for index in (0, 2, 4))
        except (TypeError, ValueError):
            return (0.0, 0.0, 0.0)

    @classmethod
    def _table_axis_edges(cls, table_bbox: Rect, placements, count: int, axis: str) -> list[float]:
        """Build grid lines from original cell boxes instead of text-size ratios."""
        is_y = axis == "y"
        start = table_bbox.y1 if is_y else table_bbox.x1
        end = table_bbox.y2 if is_y else table_bbox.x2
        cells = []
        for row, column, row_span, column_span, source in placements:
            boxes = [
                cls._page_bbox(source),
                *(cls._page_bbox(node) for node in source.find_all("span", class_="docflow-formula")),
            ]
            boxes = [box for box in boxes if box is not None]
            if not boxes:
                continue
            if is_y:
                bounds = (min(box[1] for box in boxes), max(box[3] for box in boxes))
                position, span = row, row_span
            else:
                bounds = (min(box[0] for box in boxes), max(box[2] for box in boxes))
                position, span = column, column_span
            cells.append((position, span, bounds[0], bounds[1]))

        edges = [start]
        for boundary in range(1, count):
            before = [cell_end for position, span, _cell_start, cell_end in cells if position + span == boundary]
            after = [cell_start for position, _span, cell_start, _cell_end in cells if position == boundary]
            if before and after:
                edge = (max(before) + min(after)) / 2.0
            elif before:
                edge = max(before)
            elif after:
                edge = min(after)
            else:
                edge = start + (end - start) * boundary / count
            edges.append(edge)
        edges.append(end)

        for index in range(1, len(edges)):
            if edges[index] <= edges[index - 1]:
                edges[index] = start + (end - start) * index / count
        return edges

    @staticmethod
    def _page_bbox(node) -> tuple[float, float, float, float] | None:
        value = node.get("data-page-bbox")
        if not value:
            return None
        try:
            bbox = tuple(float(part) for part in str(value).split(","))
        except ValueError:
            return None
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None
        return bbox

    @staticmethod
    def _font_family(role) -> str:
        family = str(getattr(role, "font_family", "") or "")
        if family == "黑体":
            return "'Noto Sans CJK SC','Noto Sans','Liberation Sans',sans-serif"
        return "'Noto Serif CJK SC','Noto Serif','Liberation Serif',serif"

    @classmethod
    def _text_css(cls, role, font_size: float, alignment: str, text_color=None) -> str:
        color = text_color or getattr(role, "color", "#000000")
        return (
            "div{"
            f"font-family:{cls._font_family(role)};"
            f"font-size:{font_size}pt;line-height:{font_size * 1.05}pt;"
            f"color:{color};white-space:pre-wrap;overflow-wrap:anywhere;"
            f"text-align:{alignment};"
            "}"
        )

    @staticmethod
    def _alignment(value) -> int:
        return {
            "center": fitz.TEXT_ALIGN_CENTER,
            "right": fitz.TEXT_ALIGN_RIGHT,
            "justify": fitz.TEXT_ALIGN_JUSTIFY,
        }.get(str(value or "left"), fitz.TEXT_ALIGN_LEFT)
