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
    TextStructure,
    TypographicRole,
)
from docflow.planning import ReflowPlanner


def _element(identifier, bbox, order, text="text", kind="paragraph_group", role="body", payload=None, text_structure=None):
    return SemanticElement(
        identifier,
        kind,
        Rect(*bbox),
        order,
        (f"raw-{identifier}",),
        text=text,
        role_id=role,
        payload=payload or {},
        text_structure=text_structure or TextStructure(),
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


def test_single_flow_repairs_outlying_model_order_with_geometry() -> None:
    elements = (
        _element("bottom", (100, 500, 400, 600), 1),
        _element("top", (100, 100, 900, 200), 2),
    )

    section = ReflowPlanner().plan(_analysis(elements)).pages[0].sections[0]

    assert section.element_ids == ("top", "bottom")


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
    assert section.row_heights_pt == pytest.approx((400 * 841.89 / 1400,))


def test_full_width_paragraphs_do_not_become_a_third_column_lane() -> None:
    elements = (
        _element("wide-1", (100, 50, 900, 120), 1),
        _element("wide-2", (100, 140, 900, 210), 2),
        _element("left-1", (100, 260, 450, 500), 3),
        _element("left-2", (100, 520, 450, 760), 4),
        _element("right-1", (550, 260, 900, 500), 5),
        _element("right-2", (550, 520, 900, 760), 6),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]

    assert [section.kind for section in page.sections] == [FlowKind.SINGLE, FlowKind.SINGLE, FlowKind.SEQUENTIAL_COLUMNS]
    assert len(page.sections[-1].column_widths_pt) == 2


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


def test_partial_width_elements_span_grid_columns_without_splitting_the_section() -> None:
    elements = (
        _element("title", (100, 50, 600, 100), 1, kind="heading"),
        _element("left-1", (100, 150, 300, 300), 2),
        _element("middle-1", (400, 150, 600, 300), 3),
        _element("right-1", (700, 150, 900, 300), 4),
        _element("left-2", (100, 320, 300, 450), 5),
        _element("middle-2", (400, 320, 600, 450), 6),
        _element("right-2", (700, 320, 900, 450), 7),
        _element("right-bridge", (700, 450, 900, 650), 8),
        _element("image", (100, 500, 600, 700), 9, text="", kind="figure_group"),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]
    cells = {identifier: cell for cell in page.sections[0].grid_cells for identifier in cell.element_ids}

    assert len(page.sections) == 1
    assert page.sections[0].kind == FlowKind.GRID
    assert cells["title"].column_span == 2
    assert cells["image"].column_span == 2
    assert cells["right-bridge"].row_span == 2
    assert cells["right-bridge"].row <= cells["image"].row < cells["right-bridge"].row + cells["right-bridge"].row_span
    assert page.fit_scale > 0.8


def test_repeated_icon_list_keeps_local_rows_between_spanning_blocks() -> None:
    elements = (
        _element("top", (100, 0, 900, 50), 1, kind="heading"),
        _element("text-1", (200, 60, 800, 90), 2),
        _element("icon-2", (100, 100, 150, 130), 3, text="", kind="figure_group"),
        _element("text-2", (200, 100, 800, 130), 4),
        _element("icon-3", (100, 140, 150, 170), 5, text="", kind="figure_group"),
        _element("text-3", (200, 140, 800, 170), 6),
        _element("bottom", (100, 200, 900, 250), 7, kind="heading"),
    )
    placement = {"top": 0, "bottom": 0, "icon-2": 0, "icon-3": 0, "text-1": 1, "text-2": 1, "text-3": 1}
    spans = {identifier: (2 if identifier in {"top", "bottom"} else 1) for identifier in placement}

    rows = ReflowPlanner._spanning_grid_rows(elements, placement, spans)

    assert rows["text-1"] < rows["text-2"] < rows["text-3"]
    assert rows["icon-2"] == rows["text-2"]
    assert rows["icon-3"] == rows["text-3"]


def test_grid_rows_do_not_duplicate_structural_paragraph_spacing() -> None:
    elements = (
        _element("left-top", (100, 50, 300, 80), 1),
        _element("middle-top", (400, 50, 600, 80), 2),
        _element("right-top", (700, 50, 900, 80), 3),
        _element("left-anchor", (100, 82, 300, 95), 4),
        _element("heading", (100, 100, 300, 200), 5, kind="heading"),
        _element("middle", (400, 100, 600, 300), 6),
        _element("right", (700, 100, 900, 300), 7),
        _element("bridge", (100, 350, 600, 400), 8),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]
    planned = {element.element_id: element for element in page.elements}

    assert planned["bridge"].payload["space_before_pt"] == 0.0


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


def test_grid_text_frame_ignores_a_wider_figure_in_the_same_column() -> None:
    lines = {"lines": ("one", "two", "three")}
    elements = (
        _element("wide-figure", (680, 0, 980, 80), 1, text="", kind="figure_group"),
        _element("left-1", (50, 100, 270, 250), 2, "Left text " * 8, payload=lines),
        _element("middle-1", (390, 100, 610, 250), 3, "Middle text " * 8, payload=lines),
        _element("right-1", (730, 100, 950, 250), 4, "Right text " * 8, payload=lines),
        _element("left-2", (50, 300, 270, 450), 5, "Left text " * 8, payload=lines),
        _element("middle-2", (390, 300, 610, 450), 6, "Middle text " * 8, payload=lines),
        _element("right-2", (730, 300, 950, 450), 7, "Right text " * 8, payload=lines),
        _element("spanning-figure", (50, 480, 610, 600), 8, text="", kind="figure_group"),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]
    planned = {element.element_id: element for element in page.elements}
    grid = next(section for section in page.sections if len(section.column_widths_pt) == 3)

    assert planned["right-1"].payload["alignment"] == "justify"
    assert planned["right-1"].payload["left_indent_pt"] == 0
    assert planned["right-1"].payload["right_indent_pt"] == 0
    assert grid.column_widths_pt[1] == pytest.approx(grid.column_widths_pt[2])


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
    assert sections[0].row_heights_pt == pytest.approx((200 * 841.89 / 1400, 150 * 841.89 / 1400))
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


def test_planner_preserves_gap_between_text_blocks() -> None:
    elements = (
        _element("first", (100, 100, 900, 200), 1),
        _element("second", (100, 240, 900, 340), 2),
    )

    planned = {element.element_id: element for element in ReflowPlanner().plan(_analysis(elements)).pages[0].elements}

    assert planned["second"].payload["space_before_pt"] == pytest.approx(40 * 841.89 / 1400)


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


def test_bullet_paragraph_at_right_edge_is_left_aligned() -> None:
    element = _element(
        "bullet",
        (400, 100, 900, 220),
        1,
        "• A list item",
        payload={"lines": ("• A list item",)},
        text_structure=TextStructure(is_list=True),
    )
    anchor = _element("anchor", (100, 300, 900, 400), 2)

    planned = ReflowPlanner().plan(_analysis((element, anchor))).pages[0].elements[0]

    assert planned.payload["alignment"] == "left"


def test_partial_width_table_preserves_horizontal_anchor() -> None:
    table = _element("table", (400, 100, 900, 500), 1, text="", kind="table_group")
    anchor = _element("anchor", (100, 600, 900, 700), 2)

    planned = ReflowPlanner().plan(_analysis((table, anchor))).pages[0].elements[0]

    assert planned.payload["left_indent_pt"] > 0
    assert planned.payload["right_indent_pt"] == 0


def test_option_rows_are_planned_as_explicit_line_breaks() -> None:
    question = _element(
        "question",
        (100, 100, 900, 500),
        1,
        "Question A. First B. Second C. Third",
        payload={
            "lines": ("Question", "A. First", "B. Second", "C. Third"),
            "line_lefts_px": (100, 140, 140, 140),
        },
        text_structure=TextStructure(preserve_source_lines=True, is_list=True, hanging_indent_px=40),
    )

    planned = ReflowPlanner().plan(_analysis((question,))).pages[0].elements[0]

    assert planned.text_structure.preserve_source_lines is True
    assert planned.payload["left_indent_pt"] > 0
    assert planned.payload["first_line_indent_pt"] < 0
    assert planned.payload["right_indent_pt"] == 0


def test_multiline_heading_anchored_to_column_left_is_not_centered() -> None:
    heading = _element(
        "heading",
        (100, 100, 880, 180),
        1,
        "3.1 Long heading continued on the next line",
        kind="heading",
        payload={
            "lines": ("3.1 Long heading", "continued on the next line"),
            "line_lefts_px": (100, 180),
        },
    )

    planned = ReflowPlanner().plan(_analysis((heading,))).pages[0].elements[0]

    assert planned.payload["alignment"] == "left"
    assert planned.payload["left_indent_pt"] > 0
    assert planned.payload["first_line_indent_pt"] < 0


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


def test_single_flow_off_center_heading_preserves_its_left_anchor_without_narrowing() -> None:
    heading = _element("sidebar", (200, 100, 500, 160), 1, "Sidebar", kind="heading", payload={"lines": ("Sidebar",)})
    anchor = _element("anchor", (100, 200, 900, 260), 2, "Anchor", kind="heading")

    planned = ReflowPlanner().plan(_analysis((heading, anchor))).pages[0].elements[0]

    assert planned.payload["left_indent_pt"] > 0
    assert planned.payload["right_indent_pt"] == 0


def test_overlapping_margin_note_becomes_stable_sidebar_grid() -> None:
    elements = (
        _element("body-1", (180, 100, 900, 300), 1, "Body " * 30, payload={"lines": ("a", "b", "c")}),
        _element("note", (80, 240, 160, 340), 2, "Margin note", payload={"lines": ("a", "b")}),
        _element("body-2", (180, 350, 900, 500), 3, "Body " * 20, payload={"lines": ("a", "b")}),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]

    grid = next(section for section in page.sections if section.kind == FlowKind.GRID)
    assert grid.column_widths_pt[0] < grid.column_widths_pt[1]
    assert grid.grid_cells[0].element_ids == ("note",)
    assert grid.grid_cells[1].element_ids == ("body-1", "body-2")


def test_narrow_edge_visual_is_page_furniture() -> None:
    elements = (
        _element("rail", (20, 100, 60, 400), 1, text="", kind="figure_group"),
        _element("body", (120, 200, 900, 1000), 2),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]

    assert page.header_element_ids == ("rail",)
    assert page.sections[0].element_ids == ("body",)


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


def test_page_fit_does_not_charge_markup_free_single_section_boundaries() -> None:
    planner = ReflowPlanner()
    elements = tuple(PlannedElement(str(index), "paragraph_group", "body", "x" * 50) for index in range(6))
    roles = {"body": TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.0)}
    combined = (FlowSection("combined", FlowKind.SINGLE, tuple(element.element_id for element in elements)),)
    split = tuple(FlowSection(str(index), FlowKind.SINGLE, (element.element_id,)) for index, element in enumerate(elements))

    assert planner._fit_scale(combined, elements, roles, 100, 258) == 1.0
    assert planner._fit_scale(split, elements, roles, 100, 258) == 1.0


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


def test_single_visual_pair_forms_a_local_grid() -> None:
    elements = (
        _element("intro", (100, 50, 900, 150), 1),
        _element("image", (100, 220, 450, 500), 2, text="", kind="figure_group"),
        _element("description", (550, 220, 900, 500), 3),
        _element("body", (100, 550, 900, 700), 4),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]

    assert [section.kind for section in page.sections] == [FlowKind.SINGLE, FlowKind.SEQUENTIAL_COLUMNS, FlowKind.SINGLE]
    assert len(page.sections[1].column_widths_pt) == 2


def test_edge_figure_with_continuous_text_becomes_wrapped_flow() -> None:
    elements = (
        _element("left-1", (100, 100, 500, 180), 1),
        _element("left-2", (100, 190, 500, 400), 2),
        _element("image", (600, 100, 900, 400), 3, text="", kind="figure_group"),
        _element("full-width", (100, 410, 900, 520), 4),
    )

    section = ReflowPlanner().plan(_analysis(elements)).pages[0].sections[0]

    assert section.kind == FlowKind.WRAPPED
    assert section.floating_element_id == "image"
    assert section.floating_side == "right"
    assert section.element_ids == ("left-1", "left-2", "image", "full-width")


def test_centered_heading_spans_local_visual_lanes() -> None:
    elements = (
        _element("heading", (450, 50, 550, 100), 1, kind="heading"),
        _element("left", (100, 150, 450, 450), 2),
        _element("image", (550, 150, 900, 450), 3, text="", kind="figure_group"),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]
    planned = {element.element_id: element for element in page.elements}

    assert page.sections[0].kind == FlowKind.SINGLE
    assert planned["heading"].payload["alignment"] == "center"


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
