from __future__ import annotations

from docflow.analysis import DocumentAnalyzer
from docflow.model.stages import AnalysisPage, RecognitionEvidence, RecognitionItem, RecognitionPage, Rect, SemanticElement, TextEvidence


def _item(identifier, category, bbox, order, text="", raw_type=None, html=None):
    lines = (TextEvidence(text, polygon=((bbox[0], bbox[1]), (bbox[2], bbox[3]))),) if text else ()
    return RecognitionItem(
        identifier,
        category,
        Rect(*bbox),
        order,
        text_lines=lines,
        raw_type=raw_type,
        html=html,
    )


def test_analyzer_groups_figure_content_and_caption_without_reordering() -> None:
    evidence = RecognitionEvidence(
        (
            RecognitionPage(
                0,
                1000,
                1400,
                (
                    _item("title", "title", (100, 80, 900, 140), 1, "Report"),
                    _item("figure", "figure", (100, 200, 900, 700), 2),
                    _item("chart-label", "text", (180, 300, 300, 340), 3, "2026"),
                    _item("caption", "figure_caption", (200, 720, 800, 760), 4, "Figure 1 Results"),
                    _item("body", "text", (100, 800, 900, 920), 5, "First line",),
                ),
            ),
        )
    )

    analysis = DocumentAnalyzer().analyze(evidence)
    elements = analysis.pages[0].elements

    assert [element.kind for element in elements] == ["heading", "figure_group", "paragraph_group"]
    assert elements[1].source_ids == ("figure", "chart-label", "caption")
    assert elements[1].payload["caption"] == "Figure 1 Results"
    assert [element.model_order for element in elements] == [1, 2, 5]


def test_analyzer_joins_visual_lines_and_groups_editable_formula_number() -> None:
    paragraph = RecognitionItem(
        "body",
        "text",
        Rect(100, 100, 900, 220),
        1,
        text_lines=(TextEvidence("A visual"), TextEvidence("line break")),
    )
    evidence = RecognitionEvidence(
        (
            RecognitionPage(
                0,
                1000,
                1400,
                (
                    paragraph,
                    _item("formula", "formula", (200, 300, 400, 360), 2),
                    _item("number", "formula", (850, 300, 900, 360), 3, "(7)", raw_type="formula_number"),
                ),
            ),
        )
    )

    analysis = DocumentAnalyzer().analyze(evidence)
    paragraph_group, equation_group = analysis.pages[0].elements

    assert paragraph_group.text == "A visual line break"
    assert equation_group.kind == "equation_group"
    assert equation_group.source_ids == ("formula", "number")
    assert equation_group.payload["number"] == "(7)"
    assert all(element.role_id for element in analysis.pages[0].elements if element.kind == "paragraph_group")


def test_duplicate_semantic_element_keeps_all_evidence_provenance() -> None:
    evidence = RecognitionEvidence(
        (
            RecognitionPage(
                0,
                1000,
                1400,
                (
                    _item("short", "text", (100, 100, 900, 200), 1, "same"),
                    _item("long", "text", (100, 100, 900, 200), 2, "same paragraph"),
                ),
            ),
        )
    )

    page = DocumentAnalyzer().analyze(evidence).pages[0]

    assert len(page.elements) == 1
    assert set(page.elements[0].source_ids) == {"short", "long"}
    assert page.diagnostics[0].code == "duplicate_evidence_merged"


def test_vertical_cjk_sidebar_keeps_visual_orientation() -> None:
    evidence = RecognitionEvidence(
        (RecognitionPage(0, 1000, 1400, (_item("aside", "text", (20, 100, 60, 700), 1, "公司证券研究报告"),)),)
    )

    element = DocumentAnalyzer().analyze(evidence).pages[0].elements[0]

    assert element.kind == "figure_group"
    assert element.text == ""


def test_font_size_uses_source_pitch_capped_by_ink_height() -> None:
    item = RecognitionItem(
        "body",
        "text",
        Rect(100, 100, 900, 220),
        1,
        text_lines=(TextEvidence("first", polygon=((100, 100), (900, 100), (900, 120), (100, 120))),),
    )

    role = DocumentAnalyzer().analyze(RecognitionEvidence((RecognitionPage(0, 1000, 1400, (item,)),))).roles[0]

    assert role.font_size_pt == 13.5


def test_style_clustering_absorbs_an_isolated_font_prediction() -> None:
    elements = tuple(
        SemanticElement(
            f"body-{index}",
            "paragraph_group",
            Rect(100, 100 + index * 30, 900, 100 + index * 30 + height),
            index,
            (f"r{index}",),
            text="正文文本",
            payload={"lines": ("正文文本",), "font_family": font},
        )
        for index, (font, height) in enumerate((("宋体", 18), ("宋体", 19), ("楷体", 18), ("宋体", 24)))
    )

    roles, assignments = DocumentAnalyzer()._infer_roles((AnalysisPage(0, 1000, 1400, elements),))

    assert len(roles) == 2
    assert all(role.font_family == "宋体" for role in roles)
    assert len({assignments[f"body-{index}"] for index in range(3)}) == 1
    assert assignments["body-3"] != assignments["body-0"]
