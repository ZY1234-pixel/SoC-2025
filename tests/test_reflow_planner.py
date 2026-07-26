from __future__ import annotations

import pytest

from docflow.model.stages import (
    AnalysisPage,
    DocumentAnalysis,
    FlowKind,
    FlowSection,
    PlannedElement,
    Rect,
    SemanticElement,
    TypographicRole,
)
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


def test_grid_assigns_narrow_elements_by_self_coverage() -> None:
    elements = (
        _element("left-1", (100, 100, 350, 250), 1),
        _element("right-1", (400, 100, 900, 250), 2),
        _element("left-2", (100, 300, 350, 450), 3),
        _element("right-heading", (380, 300, 470, 340), 4, kind="heading"),
        _element("right-2", (400, 350, 900, 500), 5),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]
    planned = {element.element_id: element for element in page.elements}

    assert page.sections[0].kind == FlowKind.GRID
    assert planned["right-heading"].payload["column"] == 1
    assert planned["right-heading"].payload["space_before_pt"] == pytest.approx(50 * 841.89 / 1400)


def test_asymmetric_sidebar_and_main_lane_use_grid() -> None:
    elements = (
        _element("left-late", (100, 400, 350, 500), 1),
        _element("left-top", (100, 100, 350, 250), 2),
        _element("right-top", (400, 100, 900, 250), 3),
        _element("right-bottom", (400, 300, 900, 500), 4),
    )

    section = ReflowPlanner().plan(_analysis(elements)).pages[0].sections[0]

    assert section.kind == FlowKind.GRID


def test_column_geometry_uses_all_assigned_elements_after_anchor_detection() -> None:
    elements = (
        _element("left-anchor-1", (100, 100, 350, 180), 1),
        _element("right-anchor-1", (500, 100, 900, 180), 2),
        _element("left-wide", (100, 250, 450, 400), 3, text="", kind="table_group"),
        _element("right-anchor-2", (480, 250, 900, 400), 4),
        _element("left-anchor-2", (100, 450, 350, 520), 5),
        _element("right-anchor-3", (480, 450, 900, 520), 6),
    )

    section = ReflowPlanner().plan(_analysis(elements)).pages[0].sections[0]

    assert section.gutter_pt < 40
    assert section.column_widths_pt[0] > 180


def test_repeated_edge_rails_form_an_independent_grid_lane() -> None:
    elements = (
        _element("rail-top", (20, 50, 70, 300), 1, text="", kind="figure_group"),
        _element("rail-bottom", (20, 350, 70, 1050), 2, text="", kind="figure_group"),
        _element("main-left-1", (120, 100, 600, 300), 3),
        _element("main-right-1", (650, 100, 950, 300), 4),
        _element("main-left-2", (120, 400, 600, 700), 5),
        _element("main-right-2", (650, 400, 950, 700), 6),
    )

    section = ReflowPlanner().plan(_analysis(elements)).pages[0].sections[0]

    assert section.kind == FlowKind.GRID
    assert len(section.column_widths_pt) == 3


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


def test_primary_bbox_keeps_parallel_visuals_in_the_same_grid_row() -> None:
    elements = (
        _element("left-1", (100, 100, 450, 250), 1, text="", kind="figure_group"),
        _element("right-1", (550, 100, 900, 250), 2, text="", kind="figure_group"),
        _element(
            "left-2",
            (100, 300, 700, 450),
            3,
            text="",
            kind="figure_group",
            payload={"primary_bbox": (100, 300, 450, 450)},
        ),
        _element("right-2", (550, 300, 900, 450), 4, text="", kind="table_group"),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]
    sections = page.sections

    assert len(sections) == 1
    assert sections[0].kind == FlowKind.GRID
    assert {(cell.row, cell.column) for cell in sections[0].grid_cells} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert page.elements[0].payload["width_fraction"] == pytest.approx(1.0)


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
    assert page.elements[0].payload["table_min_font_size_pt"] == pytest.approx(6.5)
    assert 0.5 < page.fit_scale < 0.85


def test_default_page_budget_reserves_incremental_text_box_wrap_error() -> None:
    compact = ReflowPlanner().plan(_analysis((_element("compact", (100, 100, 900, 500), 1, "x" * 1760),))).pages[0]
    fragmented = ReflowPlanner().plan(
        _analysis(tuple(_element(f"part-{index}", (100, 100, 900, 500), index, "x" * 88) for index in range(20)))
    ).pages[0]

    assert compact.fit_scale > fragmented.fit_scale


def test_planner_preserves_vertical_gap_from_nearest_overlapping_predecessor() -> None:
    elements = (
        _element("left-heading", (100, 100, 450, 150), 1, kind="heading"),
        _element("right", (550, 120, 900, 180), 2),
        _element("left-body", (100, 250, 450, 350), 3),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]
    planned = {element.element_id: element for element in page.elements}

    assert planned["left-body"].payload["space_before_pt"] == pytest.approx(100 * 841.89 / 1400)


def test_planner_maps_narrow_centered_text_bbox_to_paragraph_indents() -> None:
    element = _element("author", (300, 100, 700, 150), 1, "Author Name", payload={"lines": ("Author Name",)})
    body = _element("body", (100, 200, 900, 300), 2, "Body")

    planned = ReflowPlanner().plan(_analysis((element, body))).pages[0].elements[0]

    assert planned.payload["alignment"] == "center"
    assert planned.payload["left_indent_pt"] == pytest.approx(planned.payload["right_indent_pt"])
    assert planned.payload["left_indent_pt"] > 0


def test_wide_two_line_paragraph_is_left_aligned() -> None:
    element = _element(
        "risk",
        (100, 100, 900, 220),
        1,
        "Risk statement " * 10,
        payload={"lines": ("first", "second")},
    )

    planned = ReflowPlanner().plan(_analysis((element,))).pages[0].elements[0]

    assert planned.payload["alignment"] == "left"


def test_single_flow_multiline_paragraph_does_not_use_bbox_as_column_width() -> None:
    element = _element(
        "body",
        (300, 100, 700, 500),
        1,
        "Body text",
        payload={"lines": ("one", "two", "three")},
    )

    planned = ReflowPlanner().plan(_analysis((element,))).pages[0].elements[0]

    assert planned.payload["left_indent_pt"] == 0
    assert planned.payload["right_indent_pt"] == 0


def test_single_flow_preserves_substantial_inset_paragraph_width() -> None:
    anchor = _element("figure", (100, 100, 900, 300), 1, text="", kind="figure_group")
    paragraph = _element(
        "body",
        (300, 400, 900, 700),
        2,
        "Body text " * 20,
        payload={"lines": ("one", "two", "three", "four")},
    )

    planned = ReflowPlanner().plan(_analysis((anchor, paragraph))).pages[0].elements[1]

    assert planned.payload["left_indent_pt"] > 0
    assert planned.payload["right_indent_pt"] == 0


def test_single_flow_off_center_heading_does_not_use_bbox_as_text_width() -> None:
    heading = _element("sidebar", (100, 100, 500, 160), 1, "Sidebar", kind="heading", payload={"lines": ("Sidebar",)})
    anchor = _element("anchor", (100, 200, 900, 260), 2, "Anchor", kind="heading")

    planned = ReflowPlanner().plan(_analysis((heading, anchor))).pages[0].elements[0]

    assert planned.payload["left_indent_pt"] == 0
    assert planned.payload["right_indent_pt"] == 0


def test_page_fit_uses_observed_source_lines_as_a_lower_bound() -> None:
    text = "short"
    observed = ReflowPlanner().plan(
        _analysis((_element("observed", (100, 100, 900, 1200), 1, text, payload={"lines": ("a",) * 80}),))
    ).pages[0]
    estimated = ReflowPlanner().plan(_analysis((_element("estimated", (100, 100, 900, 1200), 1, text),))).pages[0]

    assert observed.fit_scale < estimated.fit_scale


def test_page_fit_does_not_trust_underreported_source_lines_over_text_width() -> None:
    text = "x" * 1760
    estimated = ReflowPlanner().plan(_analysis((_element("estimated", (100, 100, 900, 1200), 1, text),))).pages[0]
    underreported = ReflowPlanner().plan(
        _analysis((_element("observed", (100, 100, 900, 1200), 1, text, payload={"lines": ("one",)}),))
    ).pages[0]

    assert underreported.fit_scale <= estimated.fit_scale


def test_page_fit_reserves_word_flow_section_boundaries() -> None:
    planner = ReflowPlanner()
    elements = tuple(PlannedElement(str(index), "paragraph_group", "body", "x" * 50) for index in range(6))
    roles = {"body": TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.0)}
    combined = (FlowSection("combined", FlowKind.SINGLE, tuple(element.element_id for element in elements)),)
    split = tuple(FlowSection(str(index), FlowKind.SINGLE, (element.element_id,)) for index, element in enumerate(elements))

    assert planner._fit_scale(combined, elements, roles, 100, 258) == 1.0
    assert planner._fit_scale(split, elements, roles, 100, 258) < 1.0


def test_page_geometry_floor_is_counted_once_across_model_order_sections() -> None:
    elements = (
        _element("late-first", (100, 1000, 400, 1050), 1),
        _element("early-second", (100, 100, 400, 150), 2),
        _element("early-right", (600, 100, 900, 150), 2.5),
        _element("separator", (100, 300, 900, 350), 3, kind="heading"),
        _element("middle-last", (100, 500, 400, 550), 4),
        _element("right-last", (600, 500, 900, 550), 5),
    )

    page = ReflowPlanner(word_safety_factor=0.8).plan(_analysis(elements)).pages[0]

    assert page.fit_scale == pytest.approx(1.0)
    assert [section.element_ids for section in page.sections] == [
        ("early-second", "early-right"),
        ("separator",),
        ("late-first", "middle-last", "right-last"),
    ]


def test_single_flow_preserves_model_order_without_geometric_partitioning() -> None:
    elements = (
        _element("late-first", (100, 1000, 900, 1050), 1),
        _element("early-second", (100, 100, 900, 150), 2),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]

    assert len(page.sections) == 1
    assert page.sections[0].element_ids == ("late-first", "early-second")


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


def test_repeated_visual_blocks_form_a_grid_without_text_anchors() -> None:
    elements = (
        _element("top-left", (50, 100, 450, 400), 1, text="", kind="figure_group"),
        _element("top-right", (550, 100, 950, 400), 2, text="", kind="figure_group"),
        _element("bottom-left", (50, 500, 450, 800), 3, text="", kind="figure_group"),
        _element("bottom-right", (550, 500, 950, 800), 4, text="", kind="figure_group"),
    )

    section = ReflowPlanner().plan(_analysis(elements)).pages[0].sections[0]

    assert section.kind.value == "grid_flow"
    assert len(section.grid_cells) == 4


def test_repeated_small_icon_text_pairs_form_a_local_grid() -> None:
    elements = tuple(
        item
        for row, top in enumerate((100, 200, 300))
        for item in (
            _element(f"icon-{row}", (50, top, 100, top + 50), row * 2 + 1, text="", kind="figure_group"),
            _element(f"text-{row}", (130, top, 700, top + 50), row * 2 + 2),
        )
    )

    section = ReflowPlanner().plan(_analysis(elements)).pages[0].sections[0]

    assert section.kind.value == "grid_flow"
    assert len(section.column_widths_pt) == 2
    assert len(section.grid_cells) == 6


def test_independent_text_and_visual_lanes_confirm_staggered_columns() -> None:
    elements = (
        _element("left-text-1", (50, 100, 450, 150), 1),
        _element("left-figure-1", (50, 200, 450, 240), 2, text="", kind="figure_group"),
        _element("left-text-2", (50, 500, 450, 550), 3),
        _element("left-figure-2", (50, 600, 450, 640), 4, text="", kind="figure_group"),
        _element("right-text-1", (550, 250, 950, 300), 5),
        _element("right-figure-1", (550, 350, 950, 390), 6, text="", kind="figure_group"),
        _element("right-text-2", (550, 650, 950, 700), 7),
        _element("right-figure-2", (550, 750, 950, 790), 8, text="", kind="figure_group"),
    )

    section = ReflowPlanner().plan(_analysis(elements)).pages[0].sections[0]

    assert section.kind.value == "sequential_columns"
