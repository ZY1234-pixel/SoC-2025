from __future__ import annotations

import base64
import io

import pytest
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from PIL import Image

from docflow.model.stages import AnalysisPage, DocumentAnalysis, Rect, SemanticElement, TypographicRole
from docflow.planning import ReflowPlanner
from docflow.renderer.reflow_docx_renderer import ReflowDocxRenderer


def _png_base64() -> str:
    image = Image.new("RGB", (120, 40), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_reflow_docx_keeps_text_tables_and_equation_numbers_editable(tmp_path) -> None:
    roles = (
        TypographicRole("heading", "黑体", "Times New Roman", 18, 1.0, bold=True),
        TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.0),
    )
    elements = (
        SemanticElement("heading", "heading", Rect(100, 100, 900, 160), 1, ("r1",), text="Editable heading", role_id="heading"),
        SemanticElement(
            "table",
            "table_group",
            Rect(100, 220, 900, 500),
            2,
            ("r2",),
            payload={"html": "<table><tr><th>Name</th><th>Value</th></tr><tr><td>A</td><td>7</td></tr></table>"},
        ),
        SemanticElement(
            "equation",
            "equation_group",
            Rect(200, 560, 800, 640),
            3,
            ("r3",),
            payload={"image_base64": _png_base64(), "number": "(7)"},
        ),
    )
    analysis = DocumentAnalysis((AnalysisPage(0, 1000, 1400, elements),), roles)
    plan = ReflowPlanner().plan(analysis)
    output = tmp_path / "result.docx"

    ReflowDocxRenderer().render(plan, str(output))
    document = Document(output)
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    table_text = " ".join(cell.text for table in document.tables for row in table.rows for cell in row.cells)

    assert "Editable heading" in text
    assert "Name" in table_text and "Value" in table_text
    assert "(7)" in table_text
    assert len(document.tables) >= 2


def test_reflow_docx_creates_one_unlinked_section_per_source_page(tmp_path) -> None:
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.0)
    pages = tuple(
        AnalysisPage(
            index,
            1000,
            1400,
            (
                SemanticElement(f"header-{index}", "header", Rect(100, 20, 900, 60), 1, (f"h{index}",), text=f"H{index}", role_id="body"),
                SemanticElement(f"body-{index}", "paragraph_group", Rect(100, 100, 900, 1200), 2, (f"b{index}",), text=f"B{index}", role_id="body"),
            ),
        )
        for index in range(2)
    )
    output = tmp_path / "two-pages.docx"

    ReflowDocxRenderer().render(ReflowPlanner().plan(DocumentAnalysis(pages, (role,))), str(output))
    document = Document(output)

    assert len(document.sections) == 2
    first_header = " ".join(paragraph.text for paragraph in document.sections[0].header.paragraphs)
    second_header = " ".join(paragraph.text for paragraph in document.sections[1].header.paragraphs)
    assert "H0" in first_header
    assert "H1" in second_header
    assert not document.sections[1].header.is_linked_to_previous


def test_layout_table_gutter_preserves_planned_content_width() -> None:
    table = Document().add_table(rows=1, cols=3)

    ReflowDocxRenderer._format_layout_table(table, (100, 100, 100), 20)

    assert [cell.width.pt for cell in table.rows[0].cells] == [110, 120, 110]


def test_nested_table_trailing_paragraph_is_collapsed() -> None:
    cell = Document().add_table(rows=1, cols=1).cell(0, 0)
    cell.add_table(rows=1, cols=1)

    ReflowDocxRenderer._collapse_trailing_paragraph(cell)

    assert cell.paragraphs[-1].paragraph_format.line_spacing.pt == 1


def test_reflow_docx_writes_planned_vertical_spacing(tmp_path) -> None:
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.0)
    elements = (
        SemanticElement("first", "heading", Rect(100, 100, 900, 200), 1, ("r1",), text="First", role_id="body"),
        SemanticElement("second", "paragraph_group", Rect(100, 300, 900, 400), 2, ("r2",), text="Second", role_id="body"),
    )
    plan = ReflowPlanner().plan(DocumentAnalysis((AnalysisPage(0, 1000, 1400, elements),), (role,)))
    output = tmp_path / "spacing.docx"

    ReflowDocxRenderer().render(plan, str(output))
    paragraph = next(item for item in Document(output).paragraphs if item.text == "Second")

    assert plan.pages[0].elements[1].payload["space_before_pt"] > 0
    assert paragraph.paragraph_format.space_before.pt == pytest.approx(
        plan.pages[0].elements[1].payload["space_before_pt"] * plan.pages[0].fit_scale,
        abs=0.1,
    )


def test_reflow_docx_writes_source_bbox_paragraph_indents(tmp_path) -> None:
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.0)
    element = SemanticElement(
        "author",
        "paragraph_group",
        Rect(300, 100, 700, 150),
        1,
        ("r1",),
        text="Author Name",
        role_id="body",
        payload={"lines": ("Author Name",)},
    )
    body = SemanticElement("body", "paragraph_group", Rect(100, 200, 900, 300), 2, ("r2",), text="Body", role_id="body")
    plan = ReflowPlanner().plan(DocumentAnalysis((AnalysisPage(0, 1000, 1400, (element, body)),), (role,)))
    output = tmp_path / "indents.docx"

    ReflowDocxRenderer().render(plan, str(output))
    paragraph = next(item for item in Document(output).paragraphs if item.text == "Author Name")

    assert paragraph.alignment == WD_ALIGN_PARAGRAPH.CENTER
    assert paragraph.paragraph_format.left_indent.pt == pytest.approx(paragraph.paragraph_format.right_indent.pt, abs=0.1)


def test_reflow_docx_uses_deterministic_line_height(tmp_path) -> None:
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.0)
    element = SemanticElement(
        "body",
        "paragraph_group",
        Rect(100, 100, 900, 200),
        1,
        ("r1",),
        text="First line Second line",
        role_id="body",
        payload={"lines": ("First line", "Second line"), "line_heights_px": (20, 20)},
    )
    plan = ReflowPlanner().plan(DocumentAnalysis((AnalysisPage(0, 1000, 1400, (element,)),), (role,)))
    output = tmp_path / "source-lines.docx"

    ReflowDocxRenderer().render(plan, str(output))
    paragraph = next(item for item in Document(output).paragraphs if "First line" in item.text)

    assert plan.pages[0].elements[0].payload["line_height_pt"] == pytest.approx(20 * 841.89 / 1400)
    assert paragraph.text == "First line Second line"
    assert paragraph.paragraph_format.line_spacing_rule == WD_LINE_SPACING.EXACTLY
    assert paragraph.paragraph_format.widow_control is False
    assert paragraph.paragraph_format.keep_together is False
    assert paragraph.paragraph_format.keep_with_next is False
