from docflow.vendor.table_rec.fusion import LayoutObject, TableCell
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
