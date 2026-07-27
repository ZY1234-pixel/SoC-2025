"""OOXML helpers for deterministic layout-table geometry."""

from __future__ import annotations

from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def set_cell_margins(cell, top=0, bottom=0, start=0, end=0) -> None:
    properties = cell._element.get_or_add_tcPr()
    margins = properties.find(qn("w:tcMar"))
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for side, value in (("top", top), ("bottom", bottom), ("start", start), ("end", end)):
        node = margins.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            margins.append(node)
        node.set(qn("w:w"), str(int(value)))
        node.set(qn("w:type"), "dxa")


def clear_table_borders(table) -> None:
    properties = table._tbl.tblPr
    borders = properties.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        properties.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "none")
        node.set(qn("w:sz"), "0")
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), "auto")
    for row in table.rows:
        for cell in row.cells:
            set_cell_margins(cell)


def set_horizontal_table_borders(table, header_rows: int = 0) -> None:
    clear_table_borders(table)
    borders = table._tbl.tblPr.find(qn("w:tblBorders"))
    for edge in ("top", "bottom"):
        node = borders.find(qn(f"w:{edge}"))
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "6")
        node.set(qn("w:color"), "000000")
    if not header_rows:
        return
    for cell in table.rows[min(header_rows, len(table.rows)) - 1].cells:
        properties = cell._element.get_or_add_tcPr()
        borders = properties.find(qn("w:tcBorders"))
        if borders is None:
            borders = OxmlElement("w:tcBorders")
            properties.append(borders)
        bottom = borders.find(qn("w:bottom"))
        if bottom is None:
            bottom = OxmlElement("w:bottom")
            borders.append(bottom)
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6")
        bottom.set(qn("w:color"), "000000")


def set_table_col_widths(table, widths_pt) -> None:
    widths = [int(width * 20) for width in widths_pt]
    table_width = table._tbl.tblPr.find(qn("w:tblW"))
    if table_width is None:
        table_width = OxmlElement("w:tblW")
        table._tbl.tblPr.insert(0, table_width)
    table_width.set(qn("w:w"), str(sum(widths)))
    table_width.set(qn("w:type"), "dxa")
    for old in table._tbl.findall(qn("w:tblGrid")):
        table._tbl.remove(old)
    grid = OxmlElement("w:tblGrid")
    for width in widths:
        column = OxmlElement("w:gridCol")
        column.set(qn("w:w"), str(width))
        grid.append(column)
    table._tbl.tblPr.addnext(grid)
    for row in table._tbl.findall(qn("w:tr")):
        for index, cell in enumerate(row.findall(qn("w:tc"))):
            if index >= len(widths):
                continue
            properties = cell.find(qn("w:tcPr"))
            if properties is None:
                properties = OxmlElement("w:tcPr")
                cell.insert(0, properties)
            cell_width = properties.find(qn("w:tcW"))
            if cell_width is None:
                cell_width = OxmlElement("w:tcW")
                properties.insert(0, cell_width)
            cell_width.set(qn("w:w"), str(widths[index]))
            cell_width.set(qn("w:type"), "dxa")
    layout = table._tbl.tblPr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        table._tbl.tblPr.append(layout)
    layout.set(qn("w:type"), "fixed")
