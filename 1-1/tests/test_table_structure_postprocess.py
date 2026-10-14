from docflow.vendor.table_rec.fusion import LayoutTableFusion, LayoutObject, TableCell
from docflow.vendor.table_rec.mixed_rules import Rect
from docflow.vendor.table_rec.structure_postprocess import StructurePostProcessor


def _cell(row: int, col: int, bbox: tuple[float, float, float, float], **kwargs) -> TableCell:
    return TableCell(row=row, col=col, bbox=Rect(*bbox), **kwargs)


def test_wide_multi_image_header_is_split_by_stable_body_columns() -> None:
    images = [
        LayoutObject("image", Rect(x + 10, 20, x + 90, 100), role="visual_asset")
        for x in (100, 200, 300, 400)
    ]
    cells = [
        _cell(0, 0, (0, 0, 500, 200), rowspan=2, layout_objects=list(images)),
        *[_cell(0, col, (col * 100, 150, (col + 1) * 100, 200)) for col in range(1, 5)],
        *[_cell(1, col, (col * 100, 150, (col + 1) * 100, 200)) for col in range(5)],
        *[_cell(2, col, (col * 100, 200, (col + 1) * 100, 250), text=f"body-{col}") for col in range(5)],
    ]
    ocr = [
        LayoutObject("ocr_text", Rect(10, 70, 80, 100), role="ocr_text", text="型号"),
        *[
            LayoutObject(
                "ocr_text",
                Rect(col * 100 + 10, 110, col * 100 + 90, 135),
                role="ocr_text",
                text=f"MODEL-{col}",
            )
            for col in range(1, 5)
        ],
        LayoutObject("ocr_text", Rect(10, 160, 80, 190), role="ocr_text", text="屏幕尺寸"),
        *[
            LayoutObject(
                "ocr_text",
                Rect(col * 100 + 10, 160, col * 100 + 90, 190),
                role="ocr_text",
                text=f"{20 + col}英寸",
            )
            for col in range(1, 5)
        ],
    ]

    diagnostics = StructurePostProcessor().process(
        cells,
        images,
        Rect(0, 0, 500, 250),
        {"width": 500, "height": 250},
        {"_ocr_objects": ocr},
    )

    assert diagnostics["wide_image_header_splits"] == 1
    for row in (0, 1):
        assert sorted(cell.col for cell in cells if cell.row == row) == [0, 1, 2, 3, 4]
    for col in range(1, 5):
        header = next(cell for cell in cells if cell.row == 0 and cell.col == col)
        assert len([obj for obj in header.layout_objects if obj.label == "image"]) == 1
        assert f"MODEL-{col}" in header.text
        value = next(cell for cell in cells if cell.row == 1 and cell.col == col)
        assert f"{20 + col}英寸" in value.text


def test_rule_bounded_multi_line_header_is_merged_into_one_row() -> None:
    cells = [
        _cell(0, 0, (0, 5, 400, 35), colspan=4, text="model header"),
        *[_cell(1, col, (col * 100, 45, (col + 1) * 100, 95), text=f"model sub-{col}") for col in range(4)],
        *[_cell(2, col, (col * 100, 105, (col + 1) * 100, 145), text=f"body-{col}") for col in range(4)],
    ]
    ocr = [
        LayoutObject("ocr_text", Rect(10, 50, 45, 65), role="ocr_text", text="地"),
        LayoutObject("ocr_text", Rect(60, 50, 95, 65), role="ocr_text", text="区"),
    ]
    for col in range(1, 4):
        ocr.extend(
            [
                LayoutObject(
                    "ocr_text",
                    Rect(col * 100 + 15, 50, col * 100 + 80, 65),
                    role="ocr_text",
                    text=f"列{col}",
                ),
                LayoutObject(
                    "ocr_text",
                    Rect(col * 100 + 15, 75, col * 100 + 80, 88),
                    role="ocr_text",
                    text=f"单位{col}",
                ),
            ]
        )

    diagnostics = StructurePostProcessor().process(
        cells,
        [],
        Rect(0, 5, 400, 145),
        {"width": 400, "height": 145},
        {
            "_ocr_objects": ocr,
            "_horizontal_rule_ys": [5, 100, 145],
        },
    )

    assert diagnostics["header_rule_rows_merged"] == 2
    assert diagnostics["row_count"] == 2
    header = {cell.col: cell.text for cell in cells if cell.row == 0}
    assert header[0] == "地区"
    assert header[1] == "列1\n单位1"
    assert {cell.row for cell in cells if cell.text.startswith("body-")} == {1}


def test_ocr_reassignment_does_not_replace_longer_cell_text() -> None:
    cell = _cell(
        0,
        0,
        (0, 0, 300, 120),
        text="First complete line\nSecond complete line\nThird complete line",
    )
    ocr = [LayoutObject("ocr_text", Rect(10, 10, 280, 35), role="ocr_text", text="First")]

    changed, skipped = StructurePostProcessor()._reassign_text_from_ocr([cell], ocr)

    assert changed == 0
    assert skipped == 1
    assert "Second complete line" in cell.text
    assert "Third complete line" in cell.text


def test_ocr_reassignment_keeps_cell_evidence_in_sync() -> None:
    cell = _cell(0, 0, (0, 0, 100, 50), text="总")
    ocr = [
        LayoutObject("ocr_text", Rect(10, 10, 40, 40), role="ocr_text", text="总"),
        LayoutObject("ocr_text", Rect(60, 10, 90, 40), role="ocr_text", text="计"),
    ]

    changed, _skipped = StructurePostProcessor()._reassign_text_from_ocr([cell], ocr)

    assert changed == 1
    assert cell.text == "总计"
    assert [obj.text for obj in cell.ocr_objects] == ["总", "计"]


def test_wide_ocr_line_inside_cell_is_not_rejected_by_width_alone() -> None:
    cell = _cell(0, 0, (0, 0, 300, 120), rowspan=2)
    ocr = LayoutObject(
        "ocr_text",
        Rect(10, 10, 285, 40),
        role="ocr_text",
        text="A long line that is fully inside the spanning table cell",
    )

    assert not StructurePostProcessor._is_ocr_too_wide_for_cell(ocr, cell)


def test_empty_edge_rows_are_removed_from_table_cells() -> None:
    cells = [
        *[_cell(0, col, (col * 100, 0, (col + 1) * 100, 20)) for col in range(2)],
        *[_cell(1, col, (col * 100, 20, (col + 1) * 100, 60), text=f"header-{col}") for col in range(2)],
        *[_cell(2, col, (col * 100, 60, (col + 1) * 100, 100), text=f"body-{col}") for col in range(2)],
    ]

    removed = LayoutTableFusion._remove_empty_edge_rows(cells)

    assert removed == 1
    assert sorted(cell.row for cell in cells) == [0, 0, 1, 1]
    assert [cell.text for cell in cells] == ["header-0", "header-1", "body-0", "body-1"]


def test_empty_row_header_cells_merge_into_previous_header() -> None:
    cells = [
        _cell(0, 0, (0, 0, 100, 200), text="高度/cm", role="row_header"),
        _cell(1, 0, (0, 50, 100, 100), role="row_header"),
        _cell(2, 0, (0, 100, 100, 150), role="row_header"),
        _cell(3, 0, (0, 150, 100, 200), role="row_header"),
        _cell(0, 1, (100, 0, 200, 50), text="第一组"),
    ]

    merged = LayoutTableFusion._merge_empty_row_header_runs(cells)

    assert merged == 3
    assert len([cell for cell in cells if cell.col == 0]) == 1
    assert cells[0].rowspan == 4


def test_short_header_fragments_merge_across_columns() -> None:
    cells = [
        _cell(0, 0, (0, 0, 100, 100), text="参", rowspan=2, role="row_header"),
        _cell(0, 1, (100, 0, 200, 100), text="数", rowspan=2),
        _cell(0, 2, (200, 0, 500, 50), text="美人蕉", colspan=3, rowspan=1, role="column_header"),
    ]

    merged = LayoutTableFusion._merge_short_header_fragments(cells)

    assert merged == 1
    assert cells[0].text == "参数"
    assert cells[0].colspan == 2
    assert cells[0].bbox.to_list() == [0, 0, 200, 100]


def test_empty_edge_column_is_removed() -> None:
    cells = [
        _cell(row, 0, (0, row * 30, 40, (row + 1) * 30))
        for row in range(3)
    ] + [
        _cell(row, 1, (40, row * 30, 300, (row + 1) * 30), text=f"value-{row}")
        for row in range(3)
    ]
    diagnostics = {"ghost_columns_removed": 0}

    StructurePostProcessor()._remove_ghost_columns(cells, diagnostics)

    assert diagnostics["ghost_columns_removed"] == 1
    assert [cell.col for cell in cells] == [0, 0, 0]


def test_column_text_is_rebalanced_when_one_row_contains_two_values() -> None:
    cells = [
        _cell(0, 0, (0, 0, 100, 20), text="header"),
        _cell(1, 0, (0, 20, 100, 40), text=""),
        _cell(2, 0, (0, 40, 100, 60), text="1\n2"),
        _cell(3, 0, (0, 60, 100, 80), text="3"),
    ]

    changed = StructurePostProcessor._rebalance_column_text(cells)

    assert changed == 2
    assert [cells[index].text for index in range(1, 4)] == ["1", "2", "3"]
