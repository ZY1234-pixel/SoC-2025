import fitz
import numpy as np
import pytest

from docflow.model.stages import (
    AnalysisPage,
    DocumentAnalysis,
    Rect,
    SemanticElement,
    TextRow,
    TypographicRole,
)
from docflow.renderer.fidelity_pdf_renderer import FidelityPdfRenderer


def test_fidelity_pdf_keeps_page_image_and_searchable_text(tmp_path) -> None:
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.2)
    text = SemanticElement(
        "text",
        "paragraph_group",
        Rect(10, 10, 190, 30),
        0,
        ("raw-text",),
        text="可搜索文本",
        role_id="body",
        text_rows=(TextRow("可搜索文本", Rect(10, 10, 190, 30)),),
    )
    formula = SemanticElement(
        "formula",
        "equation_group",
        Rect(20, 40, 80, 60),
        1,
        ("raw-formula",),
        payload={"latex": "x^2"},
    )
    table = SemanticElement(
        "table",
        "table_group",
        Rect(10, 65, 190, 95),
        2,
        ("raw-table",),
        payload={
            "caption": "表注内容",
            "caption_position": "before",
            "primary_bbox": (10, 75, 190, 95),
        },
    )
    analysis = DocumentAnalysis((AnalysisPage(0, 200, 100, (text, formula, table)),), (role,))
    output = tmp_path / "fidelity.pdf"

    FidelityPdfRenderer().render(analysis, [np.zeros((100, 200, 3), dtype=np.uint8)], str(output))

    with fitz.open(output) as document:
        assert document[0].get_images()
        assert "可搜索文本" in document[0].get_text()
        assert "x^2" in document[0].get_text()
        assert "表注内容" in document[0].get_text()
        text_span = next(
            span
            for block in document[0].get_text("dict")["blocks"]
            for line in block.get("lines", ())
            for span in line.get("spans", ())
            if span.get("text") == "可搜索文本"
        )
        assert text_span["bbox"][2] - text_span["bbox"][0] == pytest.approx(180 * 841.89 / 200, abs=2.0)
