from __future__ import annotations

import pytest

from docflow.model.stages import (
    AnalysisPage,
    DocumentAnalysis,
    FlowKind,
    FlowSection,
    GridCell,
    PlannedElement,
    Rect,
    SemanticElement,
    TextRow,
    TextSpan,
    TextStructure,
    TextParagraphLayout,
    TypographicRole,
)
from docflow.planning import ReflowPlanner
from docflow.planning.reflow_planner import LayoutFitError
from docflow.planning.text_metrics import measure_text_width_pt, resolve_text_layout


def _element(
    identifier,
    bbox,
    order,
    text="text",
    kind="paragraph_group",
    role="body",
    payload=None,
    text_structure=None,
    text_rows=(),
):
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
        text_rows=text_rows,
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


def test_page_fit_can_use_blank_space_before_footer() -> None:
    analysis = _analysis(
        (
            _element("body", (150, 200, 850, 1000), 1),
            _element("footer", (100, 1250, 900, 1300), 2, "footer", "footer"),
        )
    )

    page = ReflowPlanner().plan(analysis).pages[0]

    assert page.geometry.margin_bottom_pt == pytest.approx((1400 - 1250) * 841.89 / 1400)
    assert page.footer_element_ids == ("footer",)


def test_formula_pages_do_not_get_an_extra_global_font_penalty() -> None:
    section = FlowSection("section_0", FlowKind.SINGLE, ("formula",))
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.0)
    base = SemanticElement(
        "formula",
        "equation_group",
        Rect(100, 100, 900, 1100),
        1,
        ("raw",),
        payload={"primary_bbox": (100, 100, 900, 1100)},
    )
    formula = SemanticElement(
        "formula",
        "equation_group",
        Rect(100, 100, 900, 1100),
        1,
        ("raw",),
        payload={"primary_bbox": (100, 100, 900, 1100), "latex": "x^2"},
    )

    roles = {role.role_id: role}
    plain_scale = ReflowPlanner()._fit_scale((section,), (base,), roles, 500.0, 500.0)
    formula_scale = ReflowPlanner()._fit_scale((section,), (formula,), roles, 500.0, 500.0)

    assert formula_scale == pytest.approx(plain_scale)


def test_full_width_body_with_leading_layout_space_is_left_aligned() -> None:
    element = SemanticElement(
        "body",
        "paragraph_group",
        Rect(100, 100, 900, 300),
        1,
        ("raw",),
        text="body " * 30,
        role_id="body",
        payload={"lines": ("line one", "line two", "line three")},
        content_bbox=Rect(140, 105, 895, 295),
    )

    assert ReflowPlanner._alignment(element, (100, 900)) == "left"


def test_preserved_centered_rows_are_center_aligned_even_when_wide() -> None:
    element = _element(
        "author",
        (100, 100, 900, 250),
        1,
        "first author line second author line",
        payload={
            "lines": ("first author line", "second author line"),
            "line_lefts_px": (180, 230),
            "line_widths_px": (640, 540),
        },
        text_structure=TextStructure(preserve_source_lines=True),
    )

    assert ReflowPlanner._alignment(element, (100, 900)) == "center"


def test_distinct_row_geometry_preserves_per_row_alignment() -> None:
    rows = (
        TextRow(
            "centered caption",
            Rect(150, 100, 850, 120),
            (TextSpan("centered caption", Rect(150, 100, 850, 120)),),
        ),
        TextRow("right credit", Rect(700, 130, 900, 150), (TextSpan("right credit", Rect(700, 130, 900, 150)),)),
    )
    element = _element(
        "caption",
        (100, 100, 900, 160),
        1,
        text_rows=rows,
    )

    assert ReflowPlanner._row_alignments(element, (100, 900)) == ("center", "right")


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


def test_sequential_columns_repair_model_order_within_each_lane() -> None:
    elements = (
        _element("left-body", (100, 180, 430, 300), 1),
        _element("left-heading", (100, 140, 430, 170), 2, kind="heading"),
        _element("right-body", (570, 180, 900, 300), 3),
        _element("right-heading", (570, 140, 900, 170), 4, kind="heading"),
    )

    section = ReflowPlanner().plan(_analysis(elements)).pages[0].sections[0]

    assert section.kind == FlowKind.SEQUENTIAL_COLUMNS
    assert section.element_ids == ("left-heading", "left-body", "right-heading", "right-body")


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

    assert [section.kind for section in page.sections] == [
        FlowKind.SINGLE,
        FlowKind.SINGLE,
        FlowKind.SEQUENTIAL_COLUMNS,
    ]
    assert len(page.sections[-1].column_widths_pt) == 2


def test_planner_uses_sequential_columns_when_model_order_alternates_independent_text_lanes() -> None:
    elements = (
        _element("left-1", (100, 100, 430, 300), 1),
        _element("right-1", (570, 100, 900, 300), 2),
        _element("left-2", (100, 320, 430, 500), 3),
        _element("right-2", (570, 320, 900, 500), 4),
    )

    page = ReflowPlanner(word_safety_factor=0.9).plan(_analysis(elements)).pages[0]

    assert page.sections[0].kind.value == "sequential_columns"
    assert page.sections[0].gutter_pt > 0
    assert page.fit_scale <= 1.0
    assert [element.element_id for element in page.elements] == [element.element_id for element in elements]
    planned = {element.element_id: element for element in page.elements}
    assert planned["left-1"].payload["width_fraction"] == pytest.approx(1.0)


def test_unbalanced_independent_text_lanes_remain_sequential_columns() -> None:
    elements = (
        _element("left-heading", (100, 100, 360, 140), 1, kind="heading"),
        _element("left-body", (100, 160, 360, 350), 2),
        _element("left-note", (100, 380, 360, 500), 3),
        _element("right-heading", (450, 100, 900, 140), 4, kind="heading"),
        _element("right-body", (450, 160, 900, 640), 5),
        _element("right-note", (450, 660, 900, 760), 6),
    )

    section = ReflowPlanner().plan(_analysis(elements)).pages[0].sections[0]

    assert section.kind == FlowKind.SEQUENTIAL_COLUMNS
    assert section.column_widths_pt[0] < section.column_widths_pt[1]
    assert section.element_ids == tuple(element.element_id for element in elements)


def test_column_crossing_elements_are_promoted_to_spanning_sections() -> None:
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

    planned = {element.element_id: element for element in page.elements}
    assert planned["title"].payload["right_indent_pt"] > 0
    image = next(cell for section in page.sections for cell in section.grid_cells if "image" in cell.element_ids)
    assert image.column_span == 2
    rendered_ids = [identifier for section in page.sections for identifier in section.element_ids]
    assert sorted(rendered_ids) == sorted(element.element_id for element in elements)
    assert page.fit_scale > 0.8


def test_same_line_icon_and_heading_share_one_grid_row() -> None:
    analysis = _analysis(
        (
            _element("icon", (420, 100, 460, 140), 1, text="", kind="figure_group"),
            _element("heading", (480, 100, 620, 140), 2, text="小标题", kind="heading"),
        )
    )

    page = ReflowPlanner().plan(analysis).pages[0]

    assert len(page.sections) == 1
    assert page.sections[0].kind == FlowKind.GRID
    cells = {cell.element_ids[0]: cell for cell in page.sections[0].grid_cells}
    assert (cells["icon"].row, cells["icon"].column) == (0, 0)
    assert (cells["heading"].row, cells["heading"].column) == (0, 1)


def test_repeated_icon_list_pairs_each_icon_with_its_text_row() -> None:
    analysis = _analysis(
        (
            _element("top", (100, 0, 900, 50), 1, kind="heading"),
            _element("icon-1", (100, 80, 150, 110), 2, text="", kind="figure_group"),
            _element("text-1", (200, 80, 800, 110), 3),
            _element("icon-2", (100, 120, 150, 150), 4, text="", kind="figure_group"),
            _element("text-2", (200, 120, 800, 150), 5),
        )
    )

    page = ReflowPlanner().plan(analysis).pages[0]
    section = next(section for section in page.sections if "icon-1" in section.element_ids)
    by_id = {identifier: cell for cell in section.grid_cells for identifier in cell.element_ids}

    assert by_id["icon-1"].row == by_id["text-1"].row
    assert by_id["icon-2"].row == by_id["text-2"].row


def test_consecutive_list_items_use_one_common_font_size() -> None:
    elements = tuple(
        _element(
            f"item-{index}",
            (100, 100 + index * 40, 700, 135 + index * 40),
            index,
            text=f"{index + 1}. list item",
            role=f"body_{index}",
            text_structure=TextStructure(is_list=True),
        )
        for index in range(3)
    )
    roles = (
        TypographicRole("body_0", "宋体", "Times New Roman", 10, 1.0),
        TypographicRole("body_1", "宋体", "Times New Roman", 14, 1.0),
        TypographicRole("body_2", "宋体", "Times New Roman", 12, 1.0),
    )

    plan = ReflowPlanner().plan(DocumentAnalysis((AnalysisPage(0, 1000, 1400, elements),), roles))
    sizes = [element.text_layout[0].font_size_pt for element in plan.pages[0].elements]

    assert max(sizes) - min(sizes) <= 0.5


def test_grid_first_cell_uses_previous_row_as_spacing_reference() -> None:
    elements = (
        _element("title", (0, 20, 190, 60), 1, kind="heading"),
        _element("right", (200, 110, 290, 150), 2),
    )
    section = FlowSection(
        "grid",
        FlowKind.GRID,
        ("title", "right"),
        (90, 90, 90),
        grid_cells=(
            GridCell(0, 0, ("title",), column_span=2),
            GridCell(1, 2, ("right",)),
        ),
    )

    spacing = ReflowPlanner._vertical_spacing(elements, (section,), {"title": 0, "right": 2}, 1.0)

    assert spacing["right"] == 50


def test_regular_grid_cell_orders_elements_by_source_position() -> None:
    elements = (
        _element("left", (0, 0, 120, 300), 1),
        _element("right-lower", (150, 160, 200, 220), 2),
        _element("right-upper", (150, 80, 200, 140), 3),
    )
    planner = ReflowPlanner()

    section, _placement = planner._narrow_section(
        elements,
        Rect(0, 0, 200, 300),
        200,
        0,
        fallback_lanes=((elements[0],), elements[1:]),
    )

    right = next(cell for cell in section.grid_cells if cell.column == 1)
    assert right.element_ids == ("right-upper", "right-lower")


def test_figure_crossing_one_fifth_of_an_adjacent_lane_spans_both_lanes() -> None:
    elements = (
        _element("left-1", (100, 100, 300, 200), 1),
        _element("middle-1", (400, 100, 600, 200), 2),
        _element("right-1", (700, 100, 900, 200), 3),
        _element("left-2", (100, 220, 300, 320), 4),
        _element("middle-2", (400, 220, 600, 320), 5),
        _element("right-2", (700, 220, 900, 320), 6),
        _element("image", (250, 350, 600, 500), 7, text="", kind="figure_group"),
    )

    sections = ReflowPlanner().plan(_analysis(elements)).pages[0].sections
    image = next(section for section in sections if "image" in section.element_ids)

    assert image.kind == FlowKind.SINGLE
    element = next(element for element in ReflowPlanner().plan(_analysis(elements)).pages[0].elements if element.element_id == "image")
    assert 0 < element.payload["width_fraction"] < 1


def test_grid_rows_preserve_local_spacing_when_tracks_follow_content() -> None:
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

    assert planned["bridge"].payload["space_before_pt"] > 0.0


def test_grid_cell_starts_keep_source_spacing_after_visuals() -> None:
    elements = (
        _element("left-visual", (100, 100, 430, 300), 1, text="", kind="figure_group"),
        _element("right-visual", (570, 100, 900, 300), 2, text="", kind="figure_group"),
        _element("following-heading", (100, 400, 430, 450), 3, kind="heading"),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]
    planned = {element.element_id: element for element in page.elements}

    assert page.sections[0].kind == FlowKind.GRID
    assert planned["following-heading"].payload["space_before_pt"] == pytest.approx(100 * 841.89 / 1400)


def test_grid_without_source_tracks_keeps_cell_start_spacing() -> None:
    elements = (
        _element("left", (50, 100, 350, 500), 1),
        _element("right-top", (450, 100, 950, 200), 2),
        _element("right-bottom", (450, 300, 950, 400), 3),
    )
    section = FlowSection(
        "section",
        FlowKind.GRID,
        tuple(element.element_id for element in elements),
        (200, 300),
        grid_cells=(GridCell(0, 0, ("left",), row_span=2), GridCell(0, 1, ("right-top",)), GridCell(1, 1, ("right-bottom",))),
    )

    spacing = ReflowPlanner._vertical_spacing(
        elements,
        (section,),
        {"left": 0, "right-top": 1, "right-bottom": 1},
        1.0,
    )

    assert spacing["right-bottom"] == 100


def test_grid_keeps_source_gap_after_spanning_figure() -> None:
    elements = (
        _element("figure", (450, 100, 950, 500), 1, kind="figure_group"),
        _element("caption", (450, 520, 950, 560), 2),
    )
    section = FlowSection(
        "section",
        FlowKind.GRID,
        ("figure", "caption"),
        (250, 250),
        grid_cells=(
            GridCell(0, 0, ("figure",), row_span=2, column_span=2),
            GridCell(2, 0, ("caption",), column_span=2),
        ),
        row_heights_pt=(200, 200, 40),
    )

    spacing = ReflowPlanner._vertical_spacing(elements, (section,), {"figure": 0, "caption": 0}, 1.0)

    assert spacing["caption"] == 20


def test_page_budget_preserves_source_whitespace_without_unnecessary_scaling() -> None:
    elements = (
        _element("first", (100, 100, 900, 600), 1, "First " * 300, payload={"lines": ("a",) * 8}),
        _element("second", (100, 900, 900, 1300), 2, "Second " * 300, payload={"lines": ("b",) * 6}),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]

    assert 0.90 < page.fit_scale <= 1.0
    source_gap = 300 * 841.89 / 1400
    assert 0 < page.elements[1].payload["space_before_pt"] <= source_gap


def test_planner_resolves_final_text_layout_from_the_same_height_model() -> None:
    element = _element(
        "body",
        (100, 100, 900, 300),
        1,
        "正文" * 30,
        payload={
            "lines": ("正文" * 10,) * 3,
            "visual_line_count": 3,
            "line_heights_px": (20, 20, 20),
        },
    )

    page = ReflowPlanner().plan(_analysis((element,))).pages[0]
    planned = page.elements[0]
    usable_width = page.geometry.width_pt - page.geometry.margin_left_pt - page.geometry.margin_right_pt

    assert planned.text_layout
    assert isinstance(planned.text_layout[0], TextParagraphLayout)
    measured_height = ReflowPlanner._element_height(
        planned,
        {"body": _analysis(()).roles[0]},
        usable_width,
        page.fit_scale,
    )
    assert measured_height == pytest.approx(sum(layout.measured_height_pt for layout in planned.text_layout))


def test_text_uses_column_width_even_when_source_box_is_narrow() -> None:
    element = PlannedElement(
        "body",
        "paragraph_group",
        "body",
        "中文内容很长" * 20,
        payload={
            "line_height_pt": 24.0,
            "target_width_pt": 80.0,
            "target_height_pt": 30.0,
        },
    )
    role = TypographicRole("body", "宋体", "Times New Roman", 12.0, 1.0)

    layouts = resolve_text_layout(element, role, 200.0, 1.0)

    assert all(
        200.0 - layout.left_indent_pt - layout.right_indent_pt == pytest.approx(200.0)
        for layout in layouts
    )
    assert all(layout.font_size_pt == pytest.approx(12.0) for layout in layouts)


def test_fixed_font_metrics_distinguish_cjk_and_times_widths() -> None:
    assert measure_text_width_pt("中文", 10.0, "宋体", "Times New Roman") == pytest.approx(20.0)
    assert measure_text_width_pt("A", 10.0, "宋体", "Times New Roman") == pytest.approx(7.22168, abs=1e-5)


def test_page_fit_compresses_gaps_before_scaling_typography() -> None:
    elements = (
        PlannedElement("first", "paragraph_group", "body", "first"),
        PlannedElement(
            "second",
            "paragraph_group",
            "body",
            "second",
            payload={"space_before_pt": 100.0},
        ),
    )
    roles = {"body": TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.0)}
    section = FlowSection("section", FlowKind.SINGLE, ("first", "second"))

    fit_scale, gap_scale = ReflowPlanner()._fit_page(
        (section,),
        elements,
        roles,
        100.0,
        100.0,
    )

    assert fit_scale == pytest.approx(1.0)
    assert 0.30 <= gap_scale < 1.0


def test_page_fit_fails_instead_of_going_below_readable_limits() -> None:
    element = PlannedElement("body", "paragraph_group", "body", "text " * 500)
    roles = {"body": TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.0)}
    section = FlowSection("section", FlowKind.SINGLE, ("body",))

    with pytest.raises(LayoutFitError):
        ReflowPlanner()._fit_page((section,), (element,), roles, 100.0, 40.0, page_index=0)


def test_heading_keeps_role_font_size_even_when_source_box_is_narrow() -> None:
    element = PlannedElement(
        "heading",
        "heading",
        "heading_1",
        "C. Quantitative cloud removal results",
        payload={"target_width_pt": 80.0, "target_height_pt": 12.0},
    )
    role = TypographicRole("heading_1", "黑体", "Times New Roman", 11.25, 1.0, bold=True)

    layout = resolve_text_layout(element, role, 200.0, 1.0)[0]

    assert layout.font_size_pt == pytest.approx(11.0)


def test_heading_source_line_width_is_not_rescaled_by_a_narrow_grid_cell() -> None:
    element = PlannedElement(
        "chapter",
        "heading",
        "heading_1",
        "七、温里药",
        payload={"source_scale": 0.516},
        text_rows=(TextRow("七、温里药", Rect(137, 173, 294, 202), ink_height_px=30),),
    )
    role = TypographicRole("heading_1", "黑体", "Times New Roman", 14.5, 1.0, bold=True)

    layout = resolve_text_layout(element, role, 83.0, 1.0)[0]

    assert layout.font_size_pt > 12.0


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
    assert planned["right-heading"].payload["space_before_pt"] > 0.0


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
        _element("left-wide", (100, 250, 400, 400), 3, text="", kind="table_group"),
        _element("right-anchor-2", (480, 250, 900, 400), 4),
        _element("left-anchor-2", (100, 450, 350, 520), 5),
        _element("right-anchor-3", (480, 450, 900, 520), 6),
    )

    section = ReflowPlanner().plan(_analysis(elements)).pages[0].sections[0]

    assert section.gutter_pt < 60
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


def test_figure_width_ignores_tight_foreground_crop() -> None:
    figure = SemanticElement(
        "figure",
        "figure_group",
        Rect(100, 100, 900, 500),
        1,
        ("raw",),
        content_bbox=Rect(700, 120, 850, 480),
    )

    planned = ReflowPlanner().plan(_analysis((figure,))).pages[0].elements[0]

    assert planned.payload["width_fraction"] == pytest.approx(1.0)


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
    assert sections[0].row_heights_pt[1] * page.fit_scale > 150 * 841.89 / 1400
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
    assert page.fit_scale == pytest.approx(1.0)


def test_table_font_size_excludes_grouped_caption_height() -> None:
    table = _element(
        "table",
        (100, 100, 900, 600),
        1,
        text="",
        kind="table_group",
        payload={
            "html": "<table>" + "<tr><td>value</td></tr>" * 10 + "</table>",
            "primary_bbox": (100, 400, 900, 600),
            "caption": "Table caption",
        },
    )

    planned = ReflowPlanner().plan(_analysis((table,))).pages[0].elements[0]

    assert planned.payload["table_font_size_pt"] == pytest.approx(200 * 841.89 / 1400 / 10 / 1.45)


def test_table_font_size_prefers_observed_text_height() -> None:
    table = _element(
        "table",
        (100, 100, 900, 600),
        1,
        text="",
        kind="table_group",
        payload={
            "html": "<table><tr><td>Value</td></tr></table>",
            "table_line_heights_px": (10, 12, 14),
        },
    )

    planned = ReflowPlanner().plan(_analysis((table,))).pages[0].elements[0]

    assert planned.payload["table_font_size_pt"] == pytest.approx(12 * 841.89 / 1400)


def test_table_height_budget_reserves_word_row_box_drift() -> None:
    element = PlannedElement(
        "table",
        "table_group",
        "body",
        payload={
            "html": "<table><tr><td>A</td></tr><tr><td>B</td></tr></table>",
            "table_font_size_pt": 8,
            "width_fraction": 1.0,
        },
    )
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.0)

    height = ReflowPlanner._element_height(element, {"body": role}, 100, 1.0)

    assert height == pytest.approx(25.2)


def test_grid_height_treats_row_span_as_a_total_height_constraint() -> None:
    def image(identifier, height):
        return PlannedElement(
            identifier,
            "figure_group",
            payload={"source_bbox": (0, 0, 100, height), "source_scale": 1.0},
        )

    elements = (image("span", 200), image("top", 150), image("bottom", 50))
    section = FlowSection(
        "grid",
        FlowKind.GRID,
        tuple(element.element_id for element in elements),
        (100, 100),
        grid_cells=(
            GridCell(0, 0, ("span",), row_span=2),
            GridCell(0, 1, ("top",)),
            GridCell(1, 1, ("bottom",)),
        ),
    )

    height = ReflowPlanner()._section_height(section, elements, {}, 200, 1.0)

    assert height == pytest.approx(200)


def test_grid_tracks_minimize_overlapping_row_span_constraints() -> None:
    def image(identifier):
        return PlannedElement(
            identifier,
            "figure_group",
            payload={"source_bbox": (0, 0, 100, 100), "source_scale": 1.0},
        )

    elements = (image("upper"), image("lower"))
    section = FlowSection(
        "grid",
        FlowKind.GRID,
        ("upper", "lower"),
        (100, 100),
        grid_cells=(
            GridCell(0, 0, ("upper",), row_span=2),
            GridCell(1, 1, ("lower",), row_span=2),
        ),
    )

    tracks = ReflowPlanner()._grid_row_heights(section, elements, {}, 1.0)

    assert sum(tracks) < 101


def test_grid_tracks_materialize_content_height_for_word() -> None:
    element = PlannedElement("text", "paragraph_group", "body", "x" * 100)
    section = FlowSection(
        "grid",
        FlowKind.GRID,
        ("text",),
        (100, 100),
        grid_cells=(GridCell(0, 0, ("text",), row_span=2),),
        row_heights_pt=(10, 10),
    )
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.0)

    tracks = ReflowPlanner()._grid_row_heights(section, (element,), {"body": role}, 1.0)

    assert sum(tracks) == pytest.approx(ReflowPlanner()._section_height(section, (element,), {"body": role}, 200, 1.0))
    assert sum(tracks) > sum(section.row_heights_pt)


def test_grid_tracks_do_not_keep_source_whitespace_as_a_height_floor() -> None:
    element = PlannedElement("text", "paragraph_group", "body", "short text")
    section = FlowSection(
        "grid",
        FlowKind.GRID,
        ("text",),
        (200,),
        grid_cells=(GridCell(0, 0, ("text",)),),
        row_heights_pt=(200,),
    )
    role = TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.0)

    tracks = ReflowPlanner()._grid_row_heights(section, (element,), {"body": role}, 1.0)

    assert tracks[0] < section.row_heights_pt[0]


def test_auto_height_grid_does_not_reserve_fixed_track_overhead() -> None:
    element = PlannedElement(
        "image",
        "figure_group",
        payload={"source_bbox": (0, 0, 100, 92), "source_scale": 1.0},
    )
    section = FlowSection(
        "grid",
        FlowKind.GRID,
        ("image",),
        (100,),
        grid_cells=(GridCell(0, 0, ("image",)),),
        row_heights_pt=(92,),
    )

    assert ReflowPlanner()._fit_scale((section,), (element,), {}, 100, 100) == 1.0


def test_default_page_budget_reserves_incremental_text_box_wrap_error() -> None:
    compact = ReflowPlanner().plan(_analysis((_element("compact", (100, 100, 900, 500), 1, "x" * 6600),))).pages[0]
    fragmented = ReflowPlanner().plan(
        _analysis(tuple(_element(f"part-{index}", (100, 100, 900, 500), index, "x" * 330) for index in range(20)))
    ).pages[0]

    assert compact.fit_scale > fragmented.fit_scale


def test_background_text_width_does_not_shrink_with_page_fit() -> None:
    element = PlannedElement(
        "label",
        "heading",
        "heading",
        "投资评级：买入（维持）",
        payload={"background_color": "#4671A6", "width_fraction": 0.8},
    )
    role = TypographicRole("heading", "黑体", "Times New Roman", 16, 1.0)

    full = resolve_text_layout(element, role, 200, 1.0)[0]
    scaled = resolve_text_layout(element, role, 200, 0.8)[0]

    assert full.right_indent_pt == pytest.approx(scaled.right_indent_pt)
    assert scaled.rendered_line_count <= full.rendered_line_count


def test_background_extent_is_part_of_visual_layout_geometry() -> None:
    element = _element(
        "band",
        (100, 100, 200, 140),
        1,
        kind="heading",
        payload={"background_bbox": (80, 90, 500, 150)},
    )

    assert ReflowPlanner._layout_bbox(element) == Rect(80, 90, 500, 150)


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


def test_vertical_gap_uses_geometry_instead_of_model_order() -> None:
    elements = (
        _element("current", (100, 300, 900, 400), 1),
        _element("previous", (100, 100, 900, 200), 2),
    )
    section = FlowSection(
        "section",
        FlowKind.GRID,
        ("current", "previous"),
        (500,),
        grid_cells=(GridCell(1, 0, ("current",)), GridCell(0, 0, ("previous",))),
    )

    spacing = ReflowPlanner._vertical_spacing(elements, (section,), {"current": 0, "previous": 0}, 1.0)

    assert spacing["current"] == 100


def test_grid_gap_uses_a_spanning_predecessor_from_another_anchor_column() -> None:
    elements = (
        _element("right-top", (500, 100, 900, 200), 1),
        _element("spanning", (100, 220, 900, 250), 2),
        _element("right-bottom", (500, 270, 900, 370), 3),
    )
    section = FlowSection(
        "section",
        FlowKind.GRID,
        tuple(element.element_id for element in elements),
        (400, 400),
        grid_cells=(
            GridCell(0, 1, ("right-top",)),
            GridCell(1, 0, ("spanning",), column_span=2),
            GridCell(2, 1, ("right-bottom",)),
        ),
    )

    spacing = ReflowPlanner._vertical_spacing(
        elements,
        (section,),
        {"right-top": 1, "spanning": 0, "right-bottom": 1},
        1.0,
    )

    assert spacing["right-bottom"] == 20


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


def test_single_flow_left_paragraph_ignores_narrow_source_box() -> None:
    element = _element("body", (300, 100, 700, 140), 1, "A short sentence that must flow naturally")
    anchor = _element("anchor", (100, 200, 900, 300), 2, "Anchor")

    planned = ReflowPlanner().plan(_analysis((element, anchor))).pages[0].elements[0]

    assert planned.payload["alignment"] == "left"
    assert planned.payload["right_indent_pt"] == 0
    assert planned.text_layout[0].rendered_line_count == 1


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


def test_split_page_edge_rail_is_grouped_as_furniture() -> None:
    elements = (
        _element("rail-top", (20, 50, 60, 300), 1, text_structure=TextStructure(orientation="vertical")),
        _element("rail-bottom", (20, 350, 60, 700), 2, text_structure=TextStructure(orientation="vertical")),
        _element("body", (120, 200, 900, 1200), 3),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]

    assert set(page.header_element_ids) == {"rail-top", "rail-bottom"}
    assert page.sections[0].element_ids == ("body",)
    assert page.geometry.margin_top_pt == pytest.approx(200 * 841.89 / 1400)


def test_top_edge_side_figure_is_anchored_without_consuming_body_height() -> None:
    elements = (
        _element("figure", (550, 0, 950, 140), 1, text="", kind="figure_group"),
        _element("title", (50, 150, 500, 220), 2, kind="heading"),
        _element("body", (50, 300, 950, 1200), 3),
    )

    page = ReflowPlanner().plan(_analysis(elements)).pages[0]

    assert page.header_element_ids == ("figure",)
    assert all("figure" not in section.element_ids for section in page.sections)
    assert page.geometry.margin_top_pt == pytest.approx(150 * 841.89 / 1400)


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


def test_grid_text_does_not_receive_an_element_local_scale() -> None:
    page = ReflowPlanner().plan(
        _analysis(
            (
                _element("left", (100, 100, 350, 400), 1, "dense " * 40),
                _element("right", (400, 100, 900, 400), 2, "right"),
            )
        )
    ).pages[0]

    assert all("grid_fit_scale" not in element.payload for element in page.elements)


def test_page_fit_does_not_charge_markup_free_single_section_boundaries() -> None:
    planner = ReflowPlanner()
    elements = tuple(PlannedElement(str(index), "paragraph_group", "body", "x" * 50) for index in range(6))
    roles = {"body": TypographicRole("body", "宋体", "Times New Roman", 10.5, 1.0)}
    combined = (FlowSection("combined", FlowKind.SINGLE, tuple(element.element_id for element in elements)),)
    split = tuple(FlowSection(str(index), FlowKind.SINGLE, (element.element_id,)) for index, element in enumerate(elements))

    combined_scale = planner._fit_scale(combined, elements, roles, 100, 258)
    split_scale = planner._fit_scale(split, elements, roles, 100, 258)

    assert combined_scale == pytest.approx(split_scale)
    assert combined_scale <= 1.0


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
        ("middle-last", "late-first", "right-last"),
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

    assert [section.kind for section in page.sections] == [
        FlowKind.SINGLE,
        FlowKind.SEQUENTIAL_COLUMNS,
        FlowKind.SINGLE,
    ]
    assert len(page.sections[1].column_widths_pt) == 2


def test_page_columns_take_priority_over_a_local_visual_pair() -> None:
    anchors = tuple(
        _element(
            f"anchor-{column}-{row}",
            (50 + column * 225, 500 + row * 100, 250 + column * 225, 580 + row * 100),
            row,
        )
        for column in range(4)
        for row in range(2)
    )
    band = (
        _element("left", (50, 100, 250, 300), 1),
        _element("left-lower", (50, 320, 250, 400), 2),
        _element("middle-left", (275, 100, 475, 300), 2),
        _element("middle-left-lower", (275, 320, 475, 400), 3),
        _element("middle-right", (500, 100, 700, 180), 3),
        _element("middle-right-lower", (500, 400, 700, 480), 4),
        _element("chart", (500, 200, 700, 380), 4, text="", kind="figure_group"),
        _element("right", (725, 100, 925, 300), 5),
    )
    fallback_lanes = tuple(
        tuple(item for item in anchors if item.element_id.startswith(f"anchor-{column}-"))
        for column in range(4)
    )

    section, placement = ReflowPlanner()._narrow_section(
        band,
        Rect(50, 0, 925, 700),
        550,
        0,
        fallback_lanes,
    )

    assert len(section.column_widths_pt) == 4
    assert {placement[item.element_id] for item in band if item.kind == "paragraph_group"} == {0, 1, 2, 3}


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


def test_grid_keeps_source_text_blocks_separate_across_other_columns() -> None:
    elements = (
        _element("left-top", (50, 100, 300, 250), 1),
        _element("right-top", (400, 100, 950, 250), 2),
        _element("right-middle", (400, 270, 950, 400), 3),
        _element("left-bottom", (50, 420, 300, 600), 4),
        _element("right-bottom", (400, 420, 950, 600), 5),
    )

    section = ReflowPlanner().plan(_analysis(elements)).pages[0].sections[0]

    assert section.kind == FlowKind.GRID
    left = [cell for cell in section.grid_cells if cell.column == 0]
    assert [cell.element_ids for cell in left] == [("left-top",), ("left-bottom",)]
    assert left[0].row + left[0].row_span == left[1].row


def test_contiguous_text_blocks_merge_across_wide_neighbor_rows() -> None:
    elements = (
        _element("left-top", (50, 100, 300, 200), 1),
        _element("left-bottom", (50, 210, 300, 300), 2),
        _element("right", (400, 100, 950, 300), 3),
    )
    by_id = {element.element_id: element for element in elements}
    cells = (
        GridCell(0, 0, ("left-top",), row_span=4),
        GridCell(4, 0, ("left-bottom",), row_span=2),
        GridCell(0, 1, ("right",), row_span=6),
    )

    merged = ReflowPlanner._merge_contiguous_grid_cells(cells, by_id)

    left = next(cell for cell in merged if cell.column == 0)
    assert left.element_ids == ("left-top", "left-bottom")
    assert left.row_span == 6


def test_grid_cell_fills_unoccupied_rows_in_its_column() -> None:
    elements = (
        _element("rail-top", (20, 50, 60, 150), 1, text_structure=TextStructure(orientation="vertical")),
        _element("main-top", (100, 100, 600, 250), 2),
        _element("right-label", (700, 100, 780, 150), 3),
        _element("right-value", (850, 100, 950, 150), 4),
        _element("rail-bottom", (20, 200, 60, 500), 5, text_structure=TextStructure(orientation="vertical")),
        _element("main-bottom", (100, 270, 600, 900), 6),
        _element("right-wide-1", (700, 200, 950, 300), 7),
        _element("right-wide-2", (700, 350, 950, 450), 8),
        _element("right-wide-3", (700, 500, 950, 600), 9),
    )

    section = ReflowPlanner().plan(_analysis(elements)).pages[0].sections[0]
    row_count = max(cell.row + cell.row_span for cell in section.grid_cells)
    rail = sorted(
        (cell for cell in section.grid_cells if cell.column == 0),
        key=lambda cell: cell.row,
    )

    assert section.kind == FlowKind.GRID
    assert [cell.element_ids for cell in rail] == [("rail-top",), ("rail-bottom",)]
    assert rail[0].row + rail[0].row_span == rail[1].row
    assert rail[1].row + rail[1].row_span == row_count


def test_overlapping_adjacent_blocks_do_not_inherit_spacing_from_an_older_block() -> None:
    elements = (
        _element("older", (100, 100, 500, 200), 1),
        _element("heading", (100, 300, 500, 350), 2, kind="heading"),
        _element("body", (100, 345, 500, 500), 3),
    )
    section = FlowSection("section", FlowKind.SINGLE, tuple(element.element_id for element in elements))

    spacing = ReflowPlanner._vertical_spacing(elements, (section,), {element.element_id: 0 for element in elements}, 1.0)

    assert spacing["body"] == 0


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


def test_long_preserved_line_shrinks_font_without_horizontal_squeeze() -> None:
    element = PlannedElement(
        "narrow",
        "paragraph_group",
        "body",
        "A very long source line that must fit",
        payload={"lines": ("A very long source line that must fit",)},
        text_structure=TextStructure(preserve_source_lines=True),
    )
    role = TypographicRole("body", "宋体", "Times New Roman", 12.0, 1.0)

    layout = resolve_text_layout(element, role, 50.0, 1.0)[0]

    assert layout.font_width_scale_pct == 100
    assert layout.font_size_pt < 12.0


def test_line_specific_text_layout_keeps_paragraph_font_size_and_fits_width() -> None:
    element = PlannedElement(
        "body",
        "paragraph_group",
        "body",
        text="A" * 10 + "\n" + "B" * 20,
        payload={"source_frame_width_px": 100.0},
        text_structure=TextStructure(preserve_source_lines=True),
        text_rows=(
            TextRow("A" * 10, Rect(0, 0, 40, 20)),
            TextRow("B" * 20, Rect(0, 20, 82, 40)),
        ),
        row_alignments=("left", "right"),
    )
    role = TypographicRole("body", "宋体", "Times New Roman", 10.0, 1.0)

    layouts = resolve_text_layout(element, role, 100.0, 1.0)

    assert len(layouts) == 2
    assert layouts[0].font_size_pt == layouts[1].font_size_pt
    assert all(90 <= layout.font_width_scale_pct <= 110 for layout in layouts)


def test_wide_centered_paragraph_is_justified_not_centered() -> None:
    element = SemanticElement(
        "body",
        "paragraph_group",
        Rect(185, 343, 1189, 477),
        1,
        ("source",),
        text="第一行中文正文内容足够长需要两端对齐\n第二行中文正文内容足够长需要两端对齐\n第三行中文正文内容足够长需要两端对齐",
        role_id="body",
        text_rows=(
            TextRow("第一行中文正文内容足够长需要两端对齐", Rect(185, 343, 1189, 387)),
            TextRow("第二行中文正文内容足够长需要两端对齐", Rect(185, 387, 1189, 431)),
            TextRow("第三行中文正文内容足够长需要两端对齐", Rect(185, 431, 1189, 475)),
        ),
    )

    assert ReflowPlanner._alignment(element, (0, 1386)) == "justify"


def test_centered_multiline_heading_is_not_classified_as_left_from_bbox_edge() -> None:
    element = SemanticElement(
        "title",
        "heading",
        Rect(113, 419, 667, 535),
        1,
        ("source",),
        text="广东省自然资源厅广东省农业农村厅关于\n妥善处理农村不动产登记历史\n遗留问题的若干意见",
        content_bbox=Rect(118, 420, 662, 533),
        text_rows=(
            TextRow("广东省自然资源厅广东省农业农村厅关于", Rect(118, 420, 662, 454)),
            TextRow("妥善处理农村不动产登记历史", Rect(199, 460, 583, 494)),
            TextRow("遗留问题的若干意见", Rect(257, 499, 525, 533)),
        ),
    )

    assert ReflowPlanner._alignment(element, (87, 697)) == "center"


def test_vertical_edge_label_is_excluded_from_body_frame() -> None:
    page = AnalysisPage(0, 1386, 1859, ())
    element = SemanticElement(
        "label",
        "paragraph_group",
        Rect(63, 155, 108, 344),
        1,
        ("source",),
        text="故事中的科学",
        role_id="body",
        text_structure=TextStructure(orientation="vertical"),
    )

    assert ReflowPlanner._is_edge_text_label(element, page) is True


def test_list_paragraph_does_not_use_per_row_alignment() -> None:
    element = _element(
        "list",
        (100, 100, 900, 160),
        1,
        "b) first line\nsecond line",
        text_structure=TextStructure(is_list=True),
        text_rows=(
            TextRow("b) first line", Rect(160, 100, 900, 130)),
            TextRow("second line", Rect(100, 130, 700, 160)),
        ),
    )

    assert ReflowPlanner._row_alignments(element, (100, 900)) == ("left", "left")


def test_caption_gap_preserves_minimum_source_spacing_when_compressed() -> None:
    body = _element("body", (0, 0, 100, 20), 1, payload={"space_before_pt": 15.0})
    caption = _element("caption", (0, 30, 100, 50), 2, kind="caption", payload={"space_before_pt": 15.0})

    adjusted = ReflowPlanner._with_compressed_gaps((body, caption), 0.30)

    assert adjusted[0].payload["space_before_pt"] == pytest.approx(4.5)
    assert adjusted[1].payload["space_before_pt"] == pytest.approx(12.0)


def test_table_embedded_text_requires_vertical_overlap() -> None:
    table = _element(
        "table",
        (100, 500, 900, 900),
        2,
        kind="table_group",
        payload={"html": "<table><tr><td>不施肥</td><td>第一组</td></tr></table>"},
    )
    above = _element("above", (100, 300, 900, 480), 1, "美人蕉最终高度不施肥组")
    below = _element("below", (100, 920, 900, 1100), 3, "经测量发现不施肥组")
    inside = _element("inside", (150, 550, 800, 650), 2.5, "不施肥")

    embedded = ReflowPlanner._embedded_table_text_ids((above, table, below, inside))

    assert embedded == {"inside"}
