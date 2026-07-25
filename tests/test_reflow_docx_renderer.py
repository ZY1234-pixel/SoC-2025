from __future__ import annotations

import base64
import io

import pytest
from docx import Document
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
    first_header = " ".join(cell.text for table in document.sections[0].header.tables for row in table.rows for cell in row.cells)
    second_header = " ".join(cell.text for table in document.sections[1].header.tables for row in table.rows for cell in row.cells)
    assert "H0" in first_header
    assert "H1" in second_header
    assert not document.sections[1].header.is_linked_to_previous


def test_layout_table_gutter_preserves_planned_content_width() -> None:
    table = Document().add_table(rows=1, cols=3)

    ReflowDocxRenderer._format_layout_table(table, (100, 100, 100), 20)

    assert [cell.width.pt for cell in table.rows[0].cells] == [110, 120, 110]


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
