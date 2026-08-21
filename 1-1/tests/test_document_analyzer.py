from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from docflow.analysis import DocumentAnalyzer
from docflow.model.stages import AnalysisPage, RecognitionEvidence, RecognitionItem, RecognitionPage, Rect, SemanticElement, TextEvidence


def _item(identifier, category, bbox, order, text="", raw_type=None, html=None, attributes=None):
    lines = (TextEvidence(text, polygon=((bbox[0], bbox[1]), (bbox[2], bbox[3]))),) if text else ()
    return RecognitionItem(
        identifier,
        category,
        Rect(*bbox),
        order,
        text_lines=lines,
        raw_type=raw_type,
        html=html,
        attributes=attributes or {},
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


def test_analyzer_preserves_caption_of_figure_embedded_in_table() -> None:
    evidence = RecognitionEvidence(
        (
            RecognitionPage(
                0,
                1000,
                1400,
                (
                    _item("table", "table", (50, 50, 950, 700), 1, html="<table><tr><td>A</td></tr></table>"),
                    _item("figure", "figure", (100, 100, 300, 300), 2),
                    _item("model", "figure_caption", (100, 310, 300, 340), 3, "Model A"),
                ),
            ),
        )
    )

    table = DocumentAnalyzer().analyze(evidence).pages[0].elements[0]

    assert table.kind == "table_group"
    assert table.source_ids == ("table", "figure", "model")
    assert table.payload["caption"] == ""


def test_analyzer_preserves_fused_table_row_proportions() -> None:
    table = _item(
        "table",
        "table",
        (0, 0, 100, 80),
        1,
        html="<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>",
        attributes={
            "source_attributes": {"table_content_fit": True},
            "table_cells": [
                {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "bbox": [0, 0, 70, 60]},
                {"row": 0, "col": 1, "rowspan": 1, "colspan": 1, "bbox": [70, 0, 100, 60]},
                {"row": 1, "col": 0, "rowspan": 1, "colspan": 1, "bbox": [0, 60, 70, 80]},
                {"row": 1, "col": 1, "rowspan": 1, "colspan": 1, "bbox": [70, 60, 100, 80]},
            ],
        },
    )

    payload = DocumentAnalyzer().analyze(
        RecognitionEvidence((RecognitionPage(0, 100, 80, (table,)),))
    ).pages[0].elements[0].payload

    assert payload["table_row_height_ratios"] == (0.75, 0.25)
    assert payload["table_column_width_ratios"] == (0.7, 0.3)
    assert payload["table_row_styles"] == ()


def test_analyzer_recovers_lineless_columns_from_inter_column_gaps() -> None:
    table = _item(
        "table",
        "table",
        (0, 0, 100, 20),
        1,
        html="<table><tr><td>A</td><td>B</td><td>C</td></tr></table>",
        attributes={
            "table_cells": [
                {"row": 0, "col": 0, "bbox": [10, 2, 30, 18]},
                {"row": 0, "col": 1, "bbox": [50, 2, 60, 18]},
                {"row": 0, "col": 2, "bbox": [90, 2, 100, 18]},
            ],
        },
    )

    payload = DocumentAnalyzer().analyze(
        RecognitionEvidence((RecognitionPage(0, 100, 20, (table,)),))
    ).pages[0].elements[0].payload

    assert payload["table_column_width_ratios"] == pytest.approx((1 / 3, 7 / 18, 5 / 18))


def test_table_header_bold_requires_row_consensus(monkeypatch) -> None:
    cells = [
        {
            "row": row,
            "col": column,
            "role": "column_header" if row == 0 else "body",
            "bbox": [column * 10, row * 10, (column + 1) * 10, (row + 1) * 10],
            "ocr_objects": [{"text": "x", "bbox": [column * 10 + 1, row * 10 + 1, (column + 1) * 10 - 1, (row + 1) * 10 - 1]}],
        }
        for row in range(2)
        for column in range(3)
    ]
    strokes = iter((0.12, 0.12, 0.05, 0.05, 0.05, 0.05))
    monkeypatch.setattr(
        "docflow.analysis.document_analyzer.infer_text_stroke_ratio",
        lambda _images: next(strokes),
    )

    styles = DocumentAnalyzer._table_cell_styles(cells, (0, 10, 20, 30), Image.new("RGB", (30, 20), "white"))

    assert not any(bold for row, _column, _alignment, bold in styles if row == 0)


def test_formula_table_cell_is_centered_from_visual_semantics() -> None:
    cells = [
        {
            "row": 0,
            "col": 0,
            "role": "body",
            "bbox": [0, 0, 100, 20],
            "layout_objects": [{"label": "inline_formula", "bbox": [40, 5, 60, 15]}],
            "ocr_objects": [],
        }
    ]

    styles = DocumentAnalyzer._table_cell_styles(cells, (0, 100))

    assert styles == ((0, 0, "center", False),)


def test_analyzer_merges_split_misclassified_table_captions_without_moving_the_table() -> None:
    evidence = RecognitionEvidence(
        (
            RecognitionPage(
                0,
                1000,
                1400,
                (
                    _item("duplicate", "text", (100, 220, 500, 245), 1, "Table 7 Results"),
                    _item("table-six", "table", (100, 100, 900, 180), 20, html="<table><tr><td>6</td></tr></table>"),
                    _item("label", "figure_caption", (100, 220, 180, 225), 28, "Table 7"),
                    _item("caption", "figure_caption", (100, 228, 500, 255), 29, "Results"),
                    _item("table-seven", "table", (100, 260, 900, 290), 30, html="<table><tr><td>7</td></tr></table>"),
                ),
            ),
        )
    )

    tables = [element for element in DocumentAnalyzer().analyze(evidence).pages[0].elements if element.kind == "table_group"]

    assert [element.model_order for element in tables] == [20, 30]
    assert tables[1].payload["caption"] == "Table 7 Results"
    assert set(tables[1].source_ids) == {"duplicate", "label", "caption", "table-seven"}


def test_analyzer_discards_ocr_lines_outside_caption_box() -> None:
    caption = RecognitionItem(
        "caption",
        "figure_caption",
        Rect(100, 100, 200, 125),
        1,
        text_lines=(
            TextEvidence("TABLE II", polygon=((100, 100), (200, 100), (200, 125), (100, 125))),
            TextEvidence("SPILLED SUBTITLE", polygon=((20, 124), (300, 124), (300, 149), (20, 149))),
        ),
    )

    assert DocumentAnalyzer._text(caption) == "TABLE II"


def test_analyzer_preserves_visible_gaps_between_same_row_text_tokens() -> None:
    item = RecognitionItem(
        "authors",
        "text",
        Rect(100, 100, 440, 125),
        1,
        text_lines=(
            TextEvidence("本报记者", polygon=((100, 100), (200, 100), (200, 125), (100, 125))),
            TextEvidence("沈小晓", polygon=((200, 100), (277, 100), (277, 125), (200, 125))),
            TextEvidence("任彦", polygon=((283, 100), (359, 100), (359, 125), (283, 125))),
            TextEvidence("黄培昭", polygon=((370, 100), (440, 100), (440, 125), (370, 125))),
        ),
    )

    element = DocumentAnalyzer().analyze(
        RecognitionEvidence((RecognitionPage(0, 1000, 1400, (item,)),))
    ).pages[0].elements[0]

    assert element.text == "本报记者沈小晓 任彦 黄培昭"


def test_analyzer_relates_short_caption_label_through_its_subtitle() -> None:
    evidence = RecognitionEvidence(
        (
            RecognitionPage(
                0,
                1000,
                1400,
                (
                    _item("previous", "table", (500, 100, 900, 500), 1, html="<table><tr><td>A</td></tr></table>"),
                    _item("label", "figure_caption", (650, 540, 750, 565), 2, "TABLE II"),
                    _item("subtitle", "figure_caption", (500, 564, 900, 610), 3, "Results"),
                    _item("target", "table", (500, 640, 900, 900), 4, html="<table><tr><td>B</td></tr></table>"),
                ),
            ),
        )
    )

    tables = [element for element in DocumentAnalyzer().analyze(evidence).pages[0].elements if element.kind == "table_group"]

    assert tables[1].payload["caption"] == "TABLE II\nResults"


def test_analyzer_keeps_same_baseline_caption_fragments_on_one_line() -> None:
    captions = (
        _item("label", "figure_caption", (100, 100, 150, 125), 1, "Table 7"),
        _item("title", "figure_caption", (300, 99, 700, 126), 2, "Measured results"),
    )

    assert DocumentAnalyzer._merge_caption_text(captions) == "Table 7\tMeasured results"


def test_analyzer_keeps_a_shared_pair_caption_outside_individual_figures() -> None:
    evidence = RecognitionEvidence(
        (
            RecognitionPage(
                0,
                1000,
                1400,
                (
                    _item("left", "figure", (100, 100, 450, 400), 1),
                    _item("left-label", "figure_caption", (220, 410, 330, 435), 2, "(a) Before"),
                    _item("right", "figure", (550, 100, 900, 400), 3),
                    _item("right-label", "figure_caption", (670, 410, 790, 435), 4, "(b) After"),
                    _item("shared", "figure_caption", (350, 470, 650, 500), 5, "Figure 3 Comparison"),
                ),
            ),
        )
    )

    elements = DocumentAnalyzer().analyze(evidence).pages[0].elements

    assert [element.kind for element in elements] == ["figure_group", "figure_group", "caption"]
    assert [element.payload.get("caption") for element in elements[:2]] == ["(a) Before", "(b) After"]
    assert elements[2].text == "Figure 3 Comparison"


def test_analyzer_preserves_independently_centered_source_rows() -> None:
    item = RecognitionItem(
        "affiliations",
        "text",
        Rect(100, 100, 900, 220),
        1,
        text_lines=(
            TextEvidence("First unit", polygon=((250, 100), (750, 100), (750, 130), (250, 130))),
            TextEvidence("Second longer unit", polygon=((150, 145), (850, 145), (850, 175), (150, 175))),
            TextEvidence("Third unit", polygon=((275, 190), (725, 190), (725, 220), (275, 220))),
        ),
    )

    element = DocumentAnalyzer().analyze(
        RecognitionEvidence((RecognitionPage(0, 1000, 1400, (item,)),))
    ).pages[0].elements[0]

    assert element.text_structure.preserve_source_lines is True
    assert tuple(row.bbox.width for row in element.text_rows) == (500, 700, 450)


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


def test_analyzer_uses_consistent_table_row_font_predictions() -> None:
    image = Image.new("RGB", (120, 90), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    labels = iter(("仿宋", "宋体", "仿宋"))

    class Classifier:
        def predict_image(self, _image):
            return SimpleNamespace(label=next(labels), confidence=0.9, margin=0.8, accepted=True)

    item = RecognitionItem(
        "table",
        "table",
        Rect(0, 0, 120, 90),
        1,
        image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        html="<table><tr><td>甲</td></tr><tr><td>乙</td></tr><tr><td>丙</td></tr></table>",
    )

    style = DocumentAnalyzer(Classifier())._infer_visual_style(item)

    assert style["font_family"] == "仿宋"


def test_analyzer_classifies_multiline_text_one_line_at_a_time() -> None:
    image = Image.new("RGB", (300, 90), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    labels = iter(("黑体", "黑体", "仿宋"))

    class Classifier:
        def predict_image(self, _image):
            return SimpleNamespace(label=next(labels), confidence=0.9, margin=0.8, accepted=True)

    item = RecognitionItem(
        "body",
        "text",
        Rect(0, 0, 300, 90),
        1,
        text_lines=tuple(
            TextEvidence(
                text,
                polygon=((0, top), (300, top), (300, top + 20), (0, top + 20)),
            )
            for text, top in zip(("第一行", "第二行", "第三行"), (0, 30, 60))
        ),
        image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )

    style = DocumentAnalyzer(Classifier())._infer_visual_style(item)

    assert style["font_family"] == "黑体"


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
        (RecognitionPage(0, 1000, 1400, (_item("aside", "text", (20, 100, 60, 420), 1, "公司证券研究报告"),)),)
    )

    analysis = DocumentAnalyzer().analyze(evidence)
    element = analysis.pages[0].elements[0]

    assert element.kind == "paragraph_group"
    assert element.text == "公司证券研究报告"
    assert element.text_structure.orientation == "vertical"
    assert analysis.roles[0].font_size_pt == 23.0


def test_analyzer_records_list_lines_and_geometry_once() -> None:
    item = RecognitionItem(
        "question",
        "text",
        Rect(100, 100, 900, 300),
        1,
        text_lines=(
            TextEvidence("Question", polygon=((100, 100), (300, 100), (300, 130), (100, 130))),
            TextEvidence("A. First", polygon=((140, 150), (300, 150), (300, 180), (140, 180))),
            TextEvidence("B. Second", polygon=((140, 200), (320, 200), (320, 230), (140, 230))),
        ),
    )

    element = DocumentAnalyzer().analyze(
        RecognitionEvidence((RecognitionPage(0, 1000, 1400, (item,)),))
    ).pages[0].elements[0]

    assert element.text_structure.preserve_source_lines is True
    assert element.text_structure.is_list is True
    assert element.text_structure.hanging_indent_px == 40


def test_analyzer_reflows_inline_numbered_list_across_ocr_rows() -> None:
    item = RecognitionItem(
        "risks",
        "text",
        Rect(100, 100, 900, 180),
        1,
        text_lines=(
            TextEvidence(
                "1) First; 2) Second; 3) Third;",
                polygon=((100, 100), (900, 100), (900, 130), (100, 130)),
            ),
            TextEvidence("4) Fourth.", polygon=((100, 145), (300, 145), (300, 175), (100, 175))),
        ),
    )

    element = DocumentAnalyzer().analyze(
        RecognitionEvidence((RecognitionPage(0, 1000, 1400, (item,)),))
    ).pages[0].elements[0]

    assert element.text_structure.is_list is True
    assert element.text_structure.preserve_source_lines is False


def test_analyzer_uses_polygon_thickness_for_rotated_text_height() -> None:
    item = RecognitionItem(
        "rotated",
        "text",
        Rect(0, 497, 116, 623),
        1,
        text_lines=(
            TextEvidence("星学", polygon=((0, 558), (77, 497), (116, 562), (13, 623))),
        ),
    )

    element = DocumentAnalyzer().analyze(
        RecognitionEvidence((RecognitionPage(0, 1000, 1400, (item,)),))
    ).pages[0].elements[0]

    assert element.text_rows[0].ink_height_px < element.bbox.height


def test_ocr_lines_are_clipped_to_their_layout_region() -> None:
    shared = TextEvidence(
        "侧注正文继续",
        polygon=((50, 100), (350, 100), (350, 130), (50, 130)),
    )
    evidence = RecognitionEvidence(
        (
            RecognitionPage(
                0,
                500,
                700,
                (
                    RecognitionItem("note", "footnote", Rect(50, 100, 150, 180), 1, text_lines=(shared, TextEvidence("内容", polygon=((50, 140), (150, 140), (150, 170), (50, 170))))),
                    RecognitionItem("body", "text", Rect(150, 100, 350, 180), 2, text_lines=(shared,)),
                ),
            ),
        )
    )

    note, body = DocumentAnalyzer().analyze(evidence).pages[0].elements

    assert note.text == "侧注内容"
    assert body.text == "正文继续"


def test_font_size_uses_source_pitch_capped_by_ink_height() -> None:
    item = RecognitionItem(
        "body",
        "text",
        Rect(100, 100, 900, 220),
        1,
        text_lines=(TextEvidence("first", polygon=((100, 100), (900, 100), (900, 120), (100, 120))),),
    )

    role = DocumentAnalyzer().analyze(RecognitionEvidence((RecognitionPage(0, 1000, 1400, (item,)),))).roles[0]

    assert role.font_size_pt == 14.0


def test_analyzer_keeps_layout_and_tight_text_geometry_separate() -> None:
    item = RecognitionItem(
        "body",
        "text",
        Rect(90, 90, 910, 230),
        1,
        text_lines=(
            TextEvidence("first", polygon=((120, 100), (880, 100), (880, 120), (120, 120))),
            TextEvidence("second", polygon=((130, 145), (870, 145), (870, 165), (130, 165))),
            TextEvidence("third", polygon=((150, 190), (800, 190), (800, 210), (150, 210))),
        ),
    )

    element = DocumentAnalyzer().analyze(
        RecognitionEvidence((RecognitionPage(0, 1000, 1400, (item,)),))
    ).pages[0].elements[0]

    assert element.bbox == Rect(90, 90, 910, 230)
    assert element.content_bbox == Rect(120, 100, 880, 210)
    assert tuple(
        right.bbox.y1 - left.bbox.y1
        for left, right in zip(element.text_rows, element.text_rows[1:])
    ) == (45, 45)
    assert len(element.text_rows) == 3


def test_visual_rows_merge_same_baseline_fragments_but_not_adjacent_lines() -> None:
    rows = DocumentAnalyzer._visual_row_boxes(
        (
            Rect(0, 0, 40, 20),
            Rect(50, 2, 100, 22),
            Rect(0, 17, 100, 37),
        )
    )

    assert rows == (Rect(0, 0, 100, 22), Rect(0, 17, 100, 37))


def test_analyzer_reflows_full_width_rows_instead_of_preserving_justified_lines() -> None:
    item = RecognitionItem(
        "body",
        "text",
        Rect(100, 100, 900, 250),
        1,
        text_lines=(
            TextEvidence("first line", polygon=((120, 100), (880, 100), (880, 120), (120, 120))),
            TextEvidence("second line", polygon=((125, 145), (875, 145), (875, 165), (125, 165))),
            TextEvidence("third line", polygon=((130, 190), (870, 190), (870, 210), (130, 210))),
        ),
    )

    element = DocumentAnalyzer().analyze(
        RecognitionEvidence((RecognitionPage(0, 1000, 1400, (item,)),))
    ).pages[0].elements[0]

    assert element.text_structure.preserve_source_lines is False


def test_analyzer_preserves_short_centered_rows() -> None:
    item = RecognitionItem(
        "author",
        "text",
        Rect(100, 100, 900, 250),
        1,
        text_lines=(
            TextEvidence("first centered line", polygon=((280, 100), (720, 100), (720, 120), (280, 120))),
            TextEvidence("second centered line", polygon=((350, 145), (650, 145), (650, 165), (350, 165))),
        ),
    )

    element = DocumentAnalyzer().analyze(
        RecognitionEvidence((RecognitionPage(0, 1000, 1400, (item,)),))
    ).pages[0].elements[0]

    assert element.text_structure.preserve_source_lines is True


def test_analyzer_reflows_hanging_rows_that_only_roughly_share_a_center() -> None:
    item = RecognitionItem(
        "question",
        "text",
        Rect(100, 100, 900, 250),
        1,
        text_lines=(
            TextEvidence("question lead", polygon=((120, 100), (880, 100), (880, 120), (120, 120))),
            TextEvidence("continued text", polygon=((175, 145), (875, 145), (875, 165), (175, 165))),
            TextEvidence("continued text", polygon=((175, 190), (875, 190), (875, 210), (175, 210))),
        ),
    )

    element = DocumentAnalyzer().analyze(
        RecognitionEvidence((RecognitionPage(0, 1000, 1400, (item,)),))
    ).pages[0].elements[0]

    assert element.text_structure.preserve_source_lines is False


def test_analyzer_preserves_split_rows_as_left_and_right_fields() -> None:
    item = RecognitionItem(
        "contact",
        "text",
        Rect(100, 100, 900, 180),
        1,
        text_lines=(
            TextEvidence("phone", polygon=((100, 100), (250, 100), (250, 125), (100, 125))),
            TextEvidence("email", polygon=((600, 100), (900, 100), (900, 125), (600, 125))),
            TextEvidence("license", polygon=((100, 145), (260, 145), (260, 170), (100, 170))),
            TextEvidence("number", polygon=((650, 145), (900, 145), (900, 170), (650, 170))),
        ),
    )

    element = DocumentAnalyzer().analyze(
        RecognitionEvidence((RecognitionPage(0, 1000, 1400, (item,)),))
    ).pages[0].elements[0]

    assert element.text_structure.tabular_rows is True
    assert tuple(tuple(span.text for span in row.spans) for row in element.text_rows) == (
        ("phone", "email"),
        ("license", "number"),
    )
    assert element.text_structure.preserve_source_lines is True


def test_heading_font_size_counts_overlapping_ocr_lines_as_one_visual_row() -> None:
    element = SemanticElement(
        "chapter",
        "heading",
        Rect(100, 100, 900, 250),
        1,
        ("raw",),
        text="2 Chapter title Continued",
        payload={
            "lines": ("2", "Chapter title", "Continued"),
            "line_tops_px": (100, 120, 180),
            "line_heights_px": (80, 60, 50),
        },
    )

    roles, _assignments = DocumentAnalyzer()._infer_roles((AnalysisPage(0, 1000, 1400, (element,)),))

    assert roles[0].font_size_pt > 35


def test_body_font_size_counts_same_baseline_ocr_fragments_as_one_visual_row() -> None:
    element = SemanticElement(
        "byline",
        "paragraph_group",
        Rect(100, 100, 900, 130),
        1,
        ("raw",),
        text="Reporter One Two",
        payload={
            "lines": ("Reporter", "One", "Two"),
            "line_tops_px": (100, 100, 100),
            "line_heights_px": (24, 24, 24),
        },
    )

    roles, _assignments = DocumentAnalyzer()._infer_roles((AnalysisPage(0, 1000, 1400, (element,)),))

    assert roles[0].font_size_pt > 10


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


def test_style_clustering_infers_bold_from_document_stroke_baseline() -> None:
    bodies = tuple(
        SemanticElement(
            f"body-{index}",
            "paragraph_group",
            Rect(100, 100 + index * 80, 900, 160 + index * 80),
            index,
            (f"body-raw-{index}",),
            text="Regular body paragraph content " * 3,
            payload={"lines": ("first", "second"), "line_heights_px": (20, 20), "stroke_ratio": 0.08},
        )
        for index in range(3)
    )
    headings = (
        SemanticElement(
            "regular-heading",
            "heading",
            Rect(100, 400, 700, 440),
            4,
            ("regular-heading-raw",),
            text="Regular heading",
            payload={"lines": ("Regular heading",), "line_heights_px": (30,), "stroke_ratio": 0.085},
        ),
        SemanticElement(
            "bold-heading",
            "heading",
            Rect(100, 460, 700, 500),
            5,
            ("bold-heading-raw",),
            text="Bold heading",
            payload={"lines": ("Bold heading",), "line_heights_px": (30,), "stroke_ratio": 0.12},
        ),
        SemanticElement(
            "unmeasured-heading",
            "heading",
            Rect(100, 520, 700, 560),
            6,
            ("unmeasured-heading-raw",),
            text="Unmeasured heading",
            payload={"lines": ("Unmeasured heading",), "line_heights_px": (30,)},
        ),
    )

    roles, assignments = DocumentAnalyzer()._infer_roles((AnalysisPage(0, 1000, 1400, bodies + headings),))
    by_id = {role.role_id: role for role in roles}

    assert not by_id[assignments["regular-heading"]].bold
    assert by_id[assignments["bold-heading"]].bold
    assert not by_id[assignments["unmeasured-heading"]].bold


def test_style_clustering_uses_strong_document_font_consensus_across_visual_roles() -> None:
    elements = tuple(
        SemanticElement(
            f"heading-{index}",
            "heading",
            Rect(100, 100 + index * 50, 900, 140 + index * 50),
            index,
            (f"r{index}",),
            text="标题文本",
            payload={
                "lines": ("标题文本",),
                "font_family": font,
                "text_color": color,
                "font_prediction": {"accepted": True, "margin": 0.9},
            },
        )
        for index, (font, color) in enumerate(
            (("楷体", "#000000"), ("楷体", "#335577"), ("楷体", "#FFFFFF"), ("仿宋", "#000000"))
        )
    )

    roles, _assignments = DocumentAnalyzer()._infer_roles((AnalysisPage(0, 1000, 1400, elements),))

    assert all(role.font_family == "楷体" for role in roles)


def test_style_clustering_absorbs_a_noisy_long_paragraph_size() -> None:
    elements = tuple(
        SemanticElement(
            f"body-{index}",
            "paragraph_group",
            Rect(100, 100 + index * 100, 900, 169 + index * 100),
            index,
            (f"r{index}",),
            text="正文段落内容" * 10,
            payload={
                "lines": ("第一行", "第二行", "第三行"),
                "line_heights_px": (line_height,) * 3,
            },
        )
        for index, line_height in enumerate((23, 23, 15, 23))
    )

    roles, assignments = DocumentAnalyzer()._infer_roles((AnalysisPage(0, 1000, 1400, elements),))

    assert len(roles) == 1
    assert roles[0].font_size_pt == 13.0
    assert len(set(assignments.values())) == 1


def test_single_line_heading_size_uses_region_height_instead_of_glyph_ink() -> None:
    headings = tuple(
        SemanticElement(
            f"heading-{index}",
            "heading",
            Rect(100, 100 + index * 60, 700, 140 + index * 60),
            index,
            (f"raw-{index}",),
            text=f"{chr(65 + index)}. Section heading",
            payload={"lines": ("Section heading",), "line_heights_px": (ink_height,)},
        )
        for index, ink_height in enumerate((22, 30, 35))
    )

    roles, assignments = DocumentAnalyzer()._infer_roles((AnalysisPage(0, 1000, 1400, headings),))

    assert len(roles) == 1
    assert len(set(assignments.values())) == 1


def test_style_clustering_raises_a_clipped_single_line_paragraph_to_body_consensus() -> None:
    elements = tuple(
        SemanticElement(
            f"body-{index}",
            "paragraph_group",
            Rect(100, 100 + index * 100, 900, 169 + index * 100),
            index,
            (f"r{index}",),
            text="Body paragraph content " * 5,
            payload={"lines": ("first", "second", "third"), "line_heights_px": (23, 23, 23)},
        )
        for index in range(3)
    ) + (
        SemanticElement(
            "clipped",
            "paragraph_group",
            Rect(100, 500, 500, 508),
            4,
            ("clipped-raw",),
            text="Return to never-ever land",
            payload={"lines": ("Return to never-ever land",), "text_color": "#DE000D"},
        ),
        SemanticElement(
            "clipped-heading",
            "heading",
            Rect(100, 550, 500, 558),
            5,
            ("clipped-heading-raw",),
            text="Virtual reality",
            payload={"lines": ("Virtual reality",), "text_color": "#DE000D"},
        ),
    )

    roles, assignments = DocumentAnalyzer()._infer_roles((AnalysisPage(0, 1000, 1400, elements),))
    by_id = {role.role_id: role for role in roles}

    assert by_id[assignments["clipped"]].font_size_pt == 13.0
    assert by_id[assignments["clipped-heading"]].font_size_pt == 13.0


def test_style_clustering_preserves_reliably_measured_small_text() -> None:
    elements = tuple(
        SemanticElement(
            f"body-{index}",
            "paragraph_group",
            Rect(100, 100 + index * 100, 900, 169 + index * 100),
            index,
            (f"r{index}",),
            text="Body paragraph content " * 5,
            payload={"lines": ("first", "second", "third"), "line_heights_px": (23, 23, 23)},
        )
        for index in range(3)
    ) + (
        SemanticElement(
            "footnote",
            "paragraph_group",
            Rect(100, 500, 500, 512),
            4,
            ("footnote-raw",),
            text="Measured footnote",
            payload={"lines": ("Measured footnote",), "line_heights_px": (11,), "text_color": "#555555"},
        ),
    )

    roles, assignments = DocumentAnalyzer()._infer_roles((AnalysisPage(0, 1000, 1400, elements),))
    by_id = {role.role_id: role for role in roles}

    assert by_id[assignments["footnote"]].font_size_pt < by_id[assignments["body-0"]].font_size_pt


def test_style_clustering_preserves_a_distinct_confident_font_role() -> None:
    elements = tuple(
        SemanticElement(
            f"heading-{index}",
            "heading",
            Rect(100, 100 + index * 50, 900, 140 + index * 50),
            index,
            (f"r{index}",),
            text="标题文本",
            payload={
                "lines": ("标题文本",),
                "font_family": "楷体",
                "font_prediction": {"accepted": True, "margin": 0.9},
            },
        )
        for index in range(3)
    ) + (
        SemanticElement(
            "accent",
            "heading",
            Rect(100, 300, 900, 340),
            4,
            ("accent-raw",),
            text="独立标题",
            payload={
                "lines": ("独立标题",),
                "font_family": "仿宋",
                "text_color": "#335577",
                "font_prediction": {"accepted": True, "margin": 0.9},
            },
        ),
    )

    roles, assignments = DocumentAnalyzer()._infer_roles((AnalysisPage(0, 1000, 1400, elements),))
    by_id = {role.role_id: role for role in roles}

    assert by_id[assignments["accent"]].font_family == "仿宋"


def test_style_clustering_uses_body_font_for_an_uncertain_regular_heading() -> None:
    bodies = tuple(
        SemanticElement(
            f"body-{index}",
            "paragraph_group",
            Rect(100, 100 + index * 100, 900, 170 + index * 100),
            index,
            (f"body-{index}-raw",),
            text="正文段落内容" * 10,
            payload={
                "lines": ("第一行", "第二行", "第三行"),
                "font_family": "楷体",
                "font_prediction": {"accepted": True, "margin": 0.8},
            },
        )
        for index in range(3)
    )
    heading = SemanticElement(
        "uncertain-heading",
        "heading",
        Rect(100, 450, 900, 500),
        4,
        ("heading-raw",),
        text="标题文本",
        payload={"lines": ("标题文本",), "font_prediction": {"accepted": False, "margin": 0.02}},
    )

    roles, assignments = DocumentAnalyzer()._infer_roles((AnalysisPage(0, 1000, 1400, bodies + (heading,)),))
    by_id = {role.role_id: role for role in roles}

    assert by_id[assignments["uncertain-heading"]].font_family == "楷体"


def test_style_clustering_unifies_moderate_body_size_noise() -> None:
    elements = tuple(
        SemanticElement(
            f"body-{index}",
            "paragraph_group",
            Rect(100, 100 + index * 100, 900, 160 + index * 100),
            index,
            (f"r{index}",),
            text="正文段落内容" * 10,
            payload={"lines": ("第一行", "第二行", "第三行"), "line_heights_px": (height,) * 3},
        )
        for index, height in enumerate((18, 20, 22))
    )

    roles, assignments = DocumentAnalyzer()._infer_roles((AnalysisPage(0, 1000, 1400, elements),))

    assert len(set(assignments.values())) == 1
    assert len(roles) == 1


def test_style_clustering_absorbs_one_large_body_outlier() -> None:
    elements = tuple(
        SemanticElement(
            f"body-{index}",
            "paragraph_group",
            Rect(100, 100 + index * 100, 900, 100 + index * 100 + height * 3),
            index,
            (f"r{index}",),
            text="正文段落内容" * 10,
            payload={"lines": ("第一行", "第二行", "第三行"), "line_heights_px": (height,) * 3},
        )
        for index, height in enumerate((22, 22, 22, 29))
    )

    roles, assignments = DocumentAnalyzer()._infer_roles((AnalysisPage(0, 1000, 1400, elements),))

    assert len(set(assignments.values())) == 1
    assert len(roles) == 1
