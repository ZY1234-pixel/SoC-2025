from __future__ import annotations

import pytest

from docflow.model.stages import AnalysisPage, DocumentAnalysis, Rect, SemanticElement, TypographicRole
from docflow.planning import ReflowPlanner


def _element(identifier, bbox, order, text="text", kind="paragraph_group", role="body", payload=None):
    return SemanticElement(
        identifier,
        kind,
        Rect(*bbox),
        order,
        (f"raw-{identifier}",),
        text=text,
        role_id=role,
        payload=payload or {},
    )


def _analysis(elements):
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.0)
    return DocumentAnalysis((AnalysisPage(0, 1000, 1400, tuple(elements)),), (role,))


def test_planner_preserves_page_ratio_and_excludes_header_from_content_frame() -> None:
    analysis = _analysis(
        (
            _element("header", (100, 20, 900, 60), 1, "header", "header"),
            _element("body", (150, 200, 850, 1200), 2),
        )
    )

    page = ReflowPlanner().plan(analysis).pages[0]

    assert round(page.geometry.height_pt, 2) == 841.89
    assert round(page.geometry.width_pt / page.geometry.height_pt, 4) == round(1000 / 1400, 4)
    assert page.geometry.margin_top_pt == pytest.approx(200 * 841.89 / 1400)
    assert page.header_element_ids == ("header",)


def test_planner_uses_sequential_columns_when_model_order_finishes_each_lane() -> None:
    elements = (
        _element("left-1", (100, 100, 430, 300), 1),
        _element("left-2", (100, 320, 430, 500), 2),
        _element("right-1", (570, 100, 900, 300), 3),
        _element("right-2", (570, 320, 900, 500), 4),
    )

    section = ReflowPlanner().plan(_analysis(elements)).pages[0].sections[0]

    assert section.kind.value == "sequential_columns"
    assert section.element_ids == tuple(element.element_id for element in elements)


def test_planner_uses_grid_when_model_order_alternates_parallel_lanes() -> None:
    elements = (
        _element("left-1", (100, 100, 430, 300), 1),
        _element("right-1", (570, 100, 900, 300), 2),
        _element("left-2", (100, 320, 430, 500), 3),
        _element("right-2", (570, 320, 900, 500), 4),
    )

    page = ReflowPlanner(word_safety_factor=0.9).plan(_analysis(elements)).pages[0]

    assert page.sections[0].kind.value == "grid_flow"
    assert page.fit_scale <= 1.0
    assert [element.element_id for element in page.elements] == [element.element_id for element in elements]
    planned = {element.element_id: element for element in page.elements}
    assert planned["left-1"].payload["width_fraction"] == pytest.approx(1.0)


def test_visual_width_uses_primary_bbox_instead_of_grouped_number_bbox() -> None:
    formula = _element(
        "formula",
        (100, 200, 900, 300),
        1,
        text="",
        kind="equation_group",
        payload={"primary_bbox": (300, 200, 500, 300), "number": "(1)"},
    )

    planned = ReflowPlanner().plan(_analysis((formula,))).pages[0].elements[0]

    assert planned.payload["width_fraction"] == pytest.approx(0.25)


def test_single_row_table_uses_body_font_and_wrap_aware_fit() -> None:
    table = _element(
        "table",
        (100, 100, 900, 1300),
        1,
        text="",
        kind="table_group",
        payload={"html": f"<table><tr><td>{'long table content ' * 500}</td></tr></table>"},
    )

    page = ReflowPlanner().plan(_analysis((table,))).pages[0]

    assert page.elements[0].payload["table_font_size_pt"] == pytest.approx(10.5)
    assert 0.5 < page.fit_scale < 0.8


def test_page_geometry_floor_is_counted_once_across_model_order_sections() -> None:
    elements = (
        _element("late-first", (100, 1000, 400, 1050), 1),
        _element("early-second", (100, 100, 400, 150), 2),
        _element("separator", (100, 300, 900, 350), 3, kind="heading"),
        _element("middle-last", (100, 500, 400, 550), 4),
    )

    page = ReflowPlanner(word_safety_factor=0.8).plan(_analysis(elements)).pages[0]

    assert page.fit_scale == pytest.approx(0.8)


def test_cross_column_paragraphs_do_not_merge_stable_lane_anchors() -> None:
    elements = (
        _element("c1", (0, 200, 220, 500), 1),
        _element("c1-more", (0, 600, 220, 800), 2),
        _element("bridge-middle", (180, 100, 620, 150), 2),
        _element("c2", (250, 200, 470, 500), 3),
        _element("c2-more", (250, 600, 470, 800), 4),
        _element("c3", (500, 200, 720, 500), 4),
        _element("c3-more", (500, 600, 720, 800), 5),
        _element("bridge-right", (510, 520, 980, 570), 5),
        _element("c4", (750, 200, 970, 500), 6),
        _element("c4-more", (750, 600, 970, 800), 7),
    )

    lanes = ReflowPlanner._anchor_lanes(elements, 1000)

    assert len(lanes) == 4
