"""Mechanical DOCX emission from a complete ReflowLayoutPlan."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

from docflow.model.stages import FlowKind, ReflowLayoutPlan
from docflow.renderer.docx_utils.html_table import get_table_columns, get_table_dimensions, get_table_rows
from docflow.renderer.docx_utils.table_fmt import clear_table_borders, set_cell_margins, set_table_col_widths


_ALIGNMENT = {
    "left": WD_ALIGN_PARAGRAPH.LEFT,
    "center": WD_ALIGN_PARAGRAPH.CENTER,
    "right": WD_ALIGN_PARAGRAPH.RIGHT,
    "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
}


class ReflowDocxRenderer:
    def render(self, plan: ReflowLayoutPlan, output_path: str) -> None:
        document = self.build(plan)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        document.save(temporary)
        temporary.replace(output)

    def build(self, plan: ReflowLayoutPlan):
        document = Document()
        roles = {role.role_id: role for role in plan.roles}
        for page_number, page in enumerate(plan.pages):
            section = document.sections[0] if page_number == 0 else document.add_section(WD_SECTION.NEW_PAGE)
            self._set_page_geometry(section, page.geometry)
            elements = {element.element_id: element for element in page.elements}
            self._render_furniture(section.header, page.header_element_ids, elements, roles, page.fit_scale)
            self._render_furniture(section.footer, page.footer_element_ids, elements, roles, page.fit_scale)
            for flow in page.sections:
                if flow.kind == FlowKind.SINGLE:
                    for identifier in flow.element_ids:
                        self._render_element(document, elements[identifier], roles, page.fit_scale, self._usable_width(page.geometry))
                elif flow.kind == FlowKind.SEQUENTIAL_COLUMNS:
                    self._render_columns(document, flow, elements, roles, page.fit_scale)
                else:
                    self._render_grid(document, flow, elements, roles, page.fit_scale)
        return document

    @staticmethod
    def _set_page_geometry(section, geometry) -> None:
        section.page_width = Pt(geometry.width_pt)
        section.page_height = Pt(geometry.height_pt)
        section.top_margin = Pt(geometry.margin_top_pt)
        section.right_margin = Pt(geometry.margin_right_pt)
        section.bottom_margin = Pt(geometry.margin_bottom_pt)
        section.left_margin = Pt(geometry.margin_left_pt)
        section.header.is_linked_to_previous = False
        section.footer.is_linked_to_previous = False

    def _render_furniture(self, container, identifiers, elements, roles, fit_scale) -> None:
        self._clear_container(container)
        for identifier in identifiers:
            element = elements[identifier]
            paragraph = container.paragraphs[0] if container.paragraphs else container.add_paragraph()
            self._write_text(paragraph, element, roles, fit_scale)

    def _render_columns(self, container, flow, elements, roles, fit_scale) -> None:
        table = container.add_table(rows=1, cols=len(flow.column_widths_pt))
        self._format_layout_table(table, flow.column_widths_pt)
        for column, cell in enumerate(table.rows[0].cells):
            self._clear_container(cell)
            for identifier in flow.element_ids:
                element = elements[identifier]
                if int(element.payload.get("column", 0)) == column:
                    self._render_element(cell, element, roles, fit_scale, flow.column_widths_pt[column])

    def _render_grid(self, container, flow, elements, roles, fit_scale) -> None:
        row_count = max(cell.row for cell in flow.grid_cells) + 1
        column_count = len(flow.column_widths_pt)
        table = container.add_table(rows=row_count, cols=column_count)
        self._format_layout_table(table, flow.column_widths_pt)
        for row in table.rows:
            for cell in row.cells:
                self._clear_container(cell)
        for grid_cell in flow.grid_cells:
            cell = table.cell(grid_cell.row, grid_cell.column)
            for identifier in grid_cell.element_ids:
                self._render_element(cell, elements[identifier], roles, fit_scale, flow.column_widths_pt[grid_cell.column])

    def _render_element(self, container, element, roles, fit_scale: float, container_width: float) -> None:
        if element.kind in {"heading", "paragraph_group", "caption"}:
            paragraph = container.add_paragraph()
            self._write_text(paragraph, element, roles, fit_scale)
            return
        if element.kind == "figure_group":
            if element.payload.get("caption_position") == "before":
                self._write_caption(container, element.payload.get("caption"), roles, fit_scale)
            self._write_image(container, element, fit_scale, container_width)
            if element.payload.get("caption_position") != "before":
                self._write_caption(container, element.payload.get("caption"), roles, fit_scale)
            return
        if element.kind == "equation_group":
            self._write_equation(container, element, roles, fit_scale, container_width)
            return
        if element.kind == "table_group":
            if element.payload.get("caption_position") == "before":
                self._write_caption(container, element.payload.get("caption"), roles, fit_scale)
            self._write_native_table(container, element, roles, fit_scale, container_width)
            if element.payload.get("caption_position") != "before":
                self._write_caption(container, element.payload.get("caption"), roles, fit_scale)

    def _write_text(self, paragraph, element, roles, fit_scale: float) -> None:
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.alignment = _ALIGNMENT.get(element.payload.get("alignment"), WD_ALIGN_PARAGRAPH.LEFT)
        role = roles.get(element.role_id) or self._body_role(roles)
        if role:
            paragraph.paragraph_format.line_spacing = role.line_spacing
        run = paragraph.add_run(element.text)
        self._style_run(run, role, fit_scale)
        if element.kind == "heading":
            paragraph.paragraph_format.keep_with_next = True

    def _write_caption(self, container, text, roles, fit_scale: float) -> None:
        if not text:
            return
        role = next((role for role_id, role in roles.items() if role_id.startswith("caption_")), None) or self._body_role(roles)
        paragraph = container.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(str(text))
        self._style_run(run, role, fit_scale)

    def _write_image(self, container, element, fit_scale: float, container_width: float) -> None:
        data = self._decode_image(element.payload.get("image_base64"))
        if not data:
            return
        width = container_width * float(element.payload.get("width_fraction", 1.0)) * fit_scale
        paragraph = container.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.add_run().add_picture(io.BytesIO(data), width=Pt(max(width, 0.5)))

    def _write_equation(self, container, element, roles, fit_scale: float, container_width: float) -> None:
        number = str(element.payload.get("number") or "")
        if not number:
            self._write_image(container, element, fit_scale, container_width)
            if not element.payload.get("image_base64") and element.payload.get("latex"):
                paragraph = container.add_paragraph(str(element.payload["latex"]))
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            return
        table = container.add_table(rows=1, cols=2)
        widths = (container_width * 0.88, container_width * 0.12)
        self._format_layout_table(table, widths)
        body = table.cell(0, 0)
        data = self._decode_image(element.payload.get("image_base64"))
        if data:
            paragraph = body.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            width = widths[0] * float(element.payload.get("width_fraction", 1.0)) * fit_scale
            paragraph.add_run().add_picture(io.BytesIO(data), width=Pt(max(width, 0.5)))
        elif element.payload.get("latex"):
            body.paragraphs[0].add_run(str(element.payload["latex"]))
        number_cell = table.cell(0, 1)
        number_cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        paragraph = number_cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        role = self._body_role(roles)
        self._style_run(paragraph.add_run(number), role, fit_scale)

    def _write_native_table(self, container, element, roles, fit_scale: float, container_width: float) -> None:
        html = element.payload.get("html")
        soup = BeautifulSoup(str(html or "<table><tr><td></td></tr></table>"), "html.parser")
        source = soup.find("table")
        if source is None:
            source = BeautifulSoup("<table><tr><td></td></tr></table>", "html.parser").find("table")
        rows, columns = get_table_dimensions(source)
        rows, columns = max(rows, 1), max(columns, 1)
        table = container.add_table(rows=rows, cols=columns)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_col_widths(table, [container_width / columns] * columns)
        occupied = [[False] * columns for _ in range(rows)]
        role = self._body_role(roles)
        for row_index, row_source in enumerate(get_table_rows(source)):
            column_index = 0
            for cell_source in get_table_columns(row_source):
                while column_index < columns and occupied[row_index][column_index]:
                    column_index += 1
                if column_index >= columns:
                    break
                row_span = min(int(cell_source.get("rowspan", 1)), rows - row_index)
                column_span = min(int(cell_source.get("colspan", 1)), columns - column_index)
                for r in range(row_index, row_index + row_span):
                    for c in range(column_index, column_index + column_span):
                        occupied[r][c] = True
                cell = table.cell(row_index, column_index)
                if row_span > 1 or column_span > 1:
                    cell = cell.merge(table.cell(row_index + row_span - 1, column_index + column_span - 1))
                paragraph = cell.paragraphs[0]
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                self._style_run(paragraph.add_run(cell_source.get_text(" ", strip=True)), role, fit_scale)
                column_index += column_span

    @staticmethod
    def _format_layout_table(table, widths) -> None:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        clear_table_borders(table)
        set_table_col_widths(table, widths)
        for row in table.rows:
            tr_pr = row._tr.get_or_add_trPr()
            cant_split = OxmlElement("w:cantSplit")
            tr_pr.append(cant_split)
            for cell in row.cells:
                set_cell_margins(cell, top=0, bottom=0, start=0, end=0)

    @staticmethod
    def _style_run(run, role, fit_scale: float) -> None:
        if role is None:
            return
        run.font.name = role.western_font_family
        run.font.size = Pt(max(round(role.font_size_pt * fit_scale * 2) / 2.0, 0.5))
        run.bold = role.bold
        run.italic = role.italic
        run.font.color.rgb = RGBColor.from_string(role.color.lstrip("#"))
        fonts = run._element.get_or_add_rPr().find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            run._element.get_or_add_rPr().insert(0, fonts)
        fonts.set(qn("w:eastAsia"), role.font_family)

    @staticmethod
    def _body_role(roles):
        return next((role for role_id, role in roles.items() if role_id.startswith("body_")), next(iter(roles.values()), None))

    @staticmethod
    def _decode_image(value):
        try:
            return base64.b64decode(value) if value else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _usable_width(geometry) -> float:
        return geometry.width_pt - geometry.margin_left_pt - geometry.margin_right_pt

    @staticmethod
    def _clear_container(container) -> None:
        for paragraph in container.paragraphs:
            paragraph._element.getparent().remove(paragraph._element)
