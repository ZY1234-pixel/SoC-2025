import base64
import io

import fitz
import numpy as np
import pytest
from bs4 import BeautifulSoup
from PIL import Image

from docflow.model.stages import (
    AnalysisPage,
    DocumentAnalysis,
    Rect,
    SemanticElement,
    TextRow,
    TypographicRole,
)
from docflow.renderer.reflow_pdf_renderer import ReflowPdfRenderer
from docflow.renderer.docx_utils.html_table import get_table_cell_placements
from docflow.renderer.formula_mathjax import MathJaxFormulaRenderer


def _png_base64() -> str:
    image = Image.new("RGB", (20, 10), "red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_reflow_pdf_places_text_and_media_at_source_coordinates(tmp_path) -> None:
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.2)
    text = SemanticElement(
        "text",
        "paragraph_group",
        Rect(10, 10, 90, 30),
        0,
        ("raw-text",),
        text="测试文本",
        role_id="body",
        text_rows=(TextRow("测试文本", Rect(10, 10, 90, 30)),),
    )
    figure = SemanticElement(
        "figure",
        "figure_group",
        Rect(100, 10, 180, 60),
        1,
        ("raw-figure",),
        payload={"image_base64": _png_base64(), "source_bbox": (100, 10, 180, 60)},
    )
    analysis = DocumentAnalysis((AnalysisPage(0, 200, 100, (text, figure)),), (role,))
    output = tmp_path / "direct.pdf"

    ReflowPdfRenderer().render(analysis, [np.zeros((100, 200, 3), dtype=np.uint8)], str(output))

    with fitz.open(output) as document:
        assert len(document) == 1
        assert "测试文本" in document[0].get_text()
        assert document[0].get_images()


def test_reflow_pdf_removes_inline_formula_marker(tmp_path) -> None:
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.2)
    text = SemanticElement(
        "text",
        "paragraph_group",
        Rect(10, 10, 90, 30),
        0,
        ("raw-text",),
        text="结果 {{DOCFLOW_FORMULA_0}}",
        role_id="body",
        text_rows=(TextRow("结果 {{DOCFLOW_FORMULA_0}}", Rect(10, 10, 90, 30)),),
        payload={"inline_formulas": ({"bbox": (40, 10, 70, 30), "image_base64": _png_base64()},)},
    )
    analysis = DocumentAnalysis((AnalysisPage(0, 100, 100, (text,)),), (role,))
    output = tmp_path / "inline.pdf"

    ReflowPdfRenderer().render(analysis, [np.zeros((100, 100, 3), dtype=np.uint8)], str(output))

    with fitz.open(output) as document:
        extracted = document[0].get_text()
        assert "DOCFLOW_FORMULA" not in extracted
        assert document[0].get_images()


def test_inline_formula_follows_rendered_text_edge(tmp_path) -> None:
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.2)
    element = SemanticElement(
        "text",
        "paragraph_group",
        Rect(0, 0, 500, 20),
        0,
        ("raw-text",),
        text="before {{DOCFLOW_FORMULA_0}}after",
        role_id="body",
        payload={"inline_formulas": ({"bbox": (300, 0, 340, 20), "latex": "x"},)},
        text_rows=(TextRow("before {{DOCFLOW_FORMULA_0}}after", Rect(0, 0, 500, 20)),),
    )
    analysis = DocumentAnalysis((AnalysisPage(0, 842, 595, (element,)),), (role,))
    output = tmp_path / "inline-flow.pdf"

    ReflowPdfRenderer().render(analysis, [np.zeros((595, 842, 3), dtype=np.uint8)], str(output))

    with fitz.open(output) as document:
        page = document[0]
        words = {word[4]: word for word in page.get_text("words")}
        drawings = page.get_drawings()
        formula_x0 = min(drawing["rect"].x0 for drawing in drawings)
        formula_x1 = max(drawing["rect"].x1 for drawing in drawings)
        assert formula_x0 == pytest.approx(words["before"][2], abs=5.0)
        assert words["after"][0] == pytest.approx(formula_x1, abs=5.0)


def test_reflow_pdf_draws_table_as_vector_lines_and_text(tmp_path) -> None:
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.2)
    table = SemanticElement(
        "table",
        "table_group",
        Rect(10, 10, 190, 90),
        0,
        ("raw-table",),
        role_id="body",
        payload={
            "html": "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>",
            "primary_bbox": (10, 10, 190, 90),
        },
    )
    analysis = DocumentAnalysis((AnalysisPage(0, 200, 100, (table,)),), (role,))
    output = tmp_path / "table.pdf"

    ReflowPdfRenderer().render(analysis, [np.zeros((100, 200, 3), dtype=np.uint8)], str(output))

    with fitz.open(output) as document:
        assert document[0].get_drawings()
        assert "A" in document[0].get_text()


def test_table_grid_uses_original_cell_and_formula_bounds() -> None:
    table = BeautifulSoup(
        """
        <table>
          <tr>
            <td data-page-bbox="100,100,200,140"><span class="docflow-formula" data-page-bbox="110,100,190,140"></span></td>
            <td data-page-bbox="220,100,300,140"></td>
          </tr>
          <tr><td data-page-bbox="100,160,200,240"></td><td data-page-bbox="220,160,300,240"></td></tr>
        </table>
        """,
        "html.parser",
    ).find("table")
    placements = get_table_cell_placements(table)

    table_bbox = Rect(90, 90, 320, 250)
    y_edges = ReflowPdfRenderer._table_axis_edges(table_bbox, placements, 2, "y")
    x_edges = ReflowPdfRenderer._table_axis_edges(table_bbox, placements, 2, "x")

    assert y_edges == pytest.approx([90.0, 150.0, 250.0])
    assert x_edges == pytest.approx([90.0, 210.0, 320.0])


def test_non_final_paragraph_row_reaches_right_edge(tmp_path) -> None:
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.2)
    element = SemanticElement(
        "text",
        "paragraph_group",
        Rect(0, 0, 200, 40),
        0,
        ("raw-text",),
        text="This line should end aligned\nlast line",
        role_id="body",
        text_rows=(
            TextRow("This line should end aligned", Rect(0, 0, 200, 20)),
            TextRow("last line", Rect(0, 20, 80, 40)),
        ),
    )
    analysis = DocumentAnalysis((AnalysisPage(0, 842, 595, (element,)),), (role,))
    output = tmp_path / "justified.pdf"

    ReflowPdfRenderer().render(analysis, [np.zeros((595, 842, 3), dtype=np.uint8)], str(output))

    with fitz.open(output) as document:
        first_row = [word for word in document[0].get_text("words") if word[1] < 15]
        assert max(word[2] for word in first_row) == pytest.approx(200.0, abs=1.0)


def test_reflow_pdf_vectorizes_formula_with_offline_mathjax(tmp_path) -> None:
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.2)
    formula = SemanticElement(
        "formula",
        "equation_group",
        Rect(10, 10, 90, 50),
        0,
        ("raw-formula",),
        role_id="body",
        payload={
            "latex": r"\frac{C}{C_{in}}=\frac{1}{2}\mathrm{erfc}(\sqrt{\xi})",
            "image_base64": _png_base64(),
        },
    )
    analysis = DocumentAnalysis((AnalysisPage(0, 100, 100, (formula,)),), (role,))
    output = tmp_path / "formula.pdf"

    ReflowPdfRenderer().render(analysis, [np.zeros((100, 100, 3), dtype=np.uint8)], str(output))

    with fitz.open(output) as document:
        assert document[0].get_drawings()
        assert not document[0].get_images()


def test_mathjax_pdf_glyphs_are_fill_only() -> None:
    pdf = MathJaxFormulaRenderer.render_pdf(r"K_{1}=\frac{1}{8}")
    assert pdf
    with fitz.open(stream=pdf, filetype="pdf") as document:
        assert document[0].get_drawings()
        assert all(drawing["type"] == "f" for drawing in document[0].get_drawings())


def test_reflow_pdf_falls_back_to_formula_crop_when_vector_rendering_fails(tmp_path) -> None:
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.2)
    formula = SemanticElement(
        "formula",
        "equation_group",
        Rect(10, 10, 90, 50),
        0,
        ("raw-formula",),
        role_id="body",
        payload={"latex": r"\def\broken{", "image_base64": _png_base64()},
    )
    analysis = DocumentAnalysis((AnalysisPage(0, 100, 100, (formula,)),), (role,))
    output = tmp_path / "formula-fallback.pdf"

    ReflowPdfRenderer().render(analysis, [np.zeros((100, 100, 3), dtype=np.uint8)], str(output))

    with fitz.open(output) as document:
        assert document[0].get_images()
