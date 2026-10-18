"""TXT、HTML 和 RTF 输出。"""

from __future__ import annotations

import html as html_lib
import math
from pathlib import Path

from bs4 import BeautifulSoup
from openpyxl import Workbook
from openpyxl.styles import Alignment

from docflow.model.stages import INLINE_FORMULA_MARKER_PATTERN


def _replace_formulas(element) -> str:
    formulas = element.payload.get("inline_formulas") or ()

    def replace(match) -> str:
        index = int(match.group(1))
        if index >= len(formulas):
            return match.group(0)
        return "$" + str(formulas[index].get("latex") or "").strip("$") + "$"

    return INLINE_FORMULA_MARKER_PATTERN.sub(replace, element.text)


def _table_rows(html: str) -> list[list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = []
        for cell in tr.find_all(["td", "th"], recursive=False):
            lines = [line.strip() for line in cell.get_text("\n", strip=True).splitlines() if line.strip()]
            cells.append("\n".join(lines))
        if cells:
            rows.append(cells)
    return rows


class PlainTextRenderer:
    def render(self, analysis, output_path: str) -> None:
        parts = []
        for page in analysis.pages:
            for element in page.elements:
                if element.kind in {"header", "footer", "page_number"}:
                    continue
                if element.kind == "table_group" and element.payload.get("html"):
                    parts.extend("\t".join(row) for row in _table_rows(str(element.payload["html"])))
                elif element.kind == "equation_group" and element.payload.get("latex"):
                    parts.append(str(element.payload["latex"]))
                elif element.text.strip():
                    parts.append(_replace_formulas(element))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


class HtmlRenderer:
    def render(self, analysis, output_path: str) -> None:
        parts = ['<!doctype html><html><head><meta charset="utf-8">',
                 '<style>body{font-family:serif;line-height:1.5;max-width:960px;margin:2em auto;}'
                 'table{border-collapse:collapse;width:100%;table-layout:fixed;}'
                 'td,th{border:1px solid #999;padding:4px;vertical-align:top;white-space:pre-wrap;word-wrap:break-word;}'
                 '.formula{overflow-x:auto;}</style></head><body>']
        for page in analysis.pages:
            for element in page.elements:
                if element.kind in {"header", "footer", "page_number"}:
                    continue
                if element.kind == "heading":
                    parts.append(f"<h2>{html_lib.escape(_replace_formulas(element))}</h2>")
                elif element.kind == "table_group" and element.payload.get("html"):
                    parts.append(str(element.payload["html"]))
                elif element.kind == "equation_group" and element.payload.get("latex"):
                    parts.append(f'<div class="formula">\\[{html_lib.escape(str(element.payload["latex"]))}\\]</div>')
                else:
                    image = element.payload.get("image_base64")
                    if image and element.kind in {"figure_group", "table_group", "equation_group"}:
                        parts.append(f'<img src="data:image/png;base64,{image}" alt="">')
                    if element.text.strip():
                        parts.append(f"<p>{html_lib.escape(_replace_formulas(element))}</p>")
                    caption = str(element.payload.get("caption") or "").strip()
                    if caption:
                        parts.append(f"<p><em>{html_lib.escape(caption)}</em></p>")
        parts.append("</body></html>")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _rtf_escape(text: str, line_break: str = "\\par ") -> str:
    out = []
    for char in str(text):
        if char == "\\":
            out.append("\\\\")
        elif char == "{":
            out.append("\\{")
        elif char == "}":
            out.append("\\}")
        elif char == "\n":
            out.append(line_break)
        elif ord(char) < 128:
            out.append(char)
        else:
            value = ord(char)
            if value > 32767:
                value -= 65536
            out.append(f"\\u{value}?")
    return "".join(out)


def _rtf_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    columns = max(len(row) for row in rows)
    cell_width = max(9000 // max(columns, 1), 1)
    parts = []
    for row in rows:
        parts.append("\\trowd\\trgaph108\\trleft0")
        for index in range(columns):
            parts.append(f"\\clvertalt\\cellx{(index + 1) * cell_width}")
        padded = row + [""] * (columns - len(row))
        parts.append("".join(_rtf_escape(cell, "\\line ") + "\\cell " for cell in padded))
        parts.append("\\row ")
    return "".join(parts) + "\\pard "


class RtfRenderer:
    def render(self, analysis, output_path: str) -> None:
        body = []
        for page in analysis.pages:
            for element in page.elements:
                if element.kind in {"header", "footer", "page_number"}:
                    continue
                if element.kind == "table_group" and element.payload.get("html"):
                    body.append(_rtf_table(_table_rows(str(element.payload["html"]))))
                elif element.kind == "equation_group" and element.payload.get("latex"):
                    body.append(_rtf_escape(str(element.payload["latex"])) + "\\par ")
                elif element.text.strip():
                    body.append(_rtf_escape(_replace_formulas(element)) + "\\par ")
            body.append("\\page ")
        content = "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil\\fcharset134 SimSun;}}\\viewkind4\\uc1\\pard\\f0\\fs21 " + "".join(body) + "}"
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="ascii", errors="backslashreplace")


class XlsxRenderer:
    def render(self, analysis, output_path: str) -> None:
        workbook = Workbook()
        text_sheet = workbook.active
        text_sheet.title = "正文"
        text_sheet.append(["页码", "类型", "内容"])
        table_index = 0
        for page in analysis.pages:
            for element in page.elements:
                if element.kind in {"header", "footer", "page_number"}:
                    continue
                if element.kind == "table_group" and element.payload.get("html"):
                    table_index += 1
                    sheet = workbook.create_sheet(f"表{table_index}")
                    for row in _table_rows(str(element.payload["html"])):
                        sheet.append(row)
                    for row in sheet.iter_rows():
                        for cell in row:
                            cell.alignment = Alignment(wrap_text=True, vertical="top")
                    for column in sheet.columns:
                        sheet.column_dimensions[column[0].column_letter].width = 40
                    for row_index, row in enumerate(sheet.iter_rows(values_only=True), start=1):
                        estimated_lines = max(
                            (
                                math.ceil(len(str(value or "")) / 40) + str(value or "").count("\n")
                                for value in row
                            ),
                            default=1,
                        )
                        sheet.row_dimensions[row_index].height = min(max(estimated_lines * 15, 30), 300)
                elif element.kind == "equation_group" and element.payload.get("latex"):
                    text_sheet.append([page.page_index + 1, "公式", str(element.payload["latex"])])
                elif element.text.strip():
                    text_sheet.append([page.page_index + 1, element.kind, _replace_formulas(element)])
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        workbook.save(output)
