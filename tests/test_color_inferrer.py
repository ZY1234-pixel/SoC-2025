import base64

import cv2
import numpy as np

from docflow.layout.color_inferrer import infer_crop_style, infer_table_row_fills, infer_table_rule_style


def test_crop_style_and_table_fill_are_inferred_from_region_pixels():
    image = np.full((60, 180, 3), 255, dtype=np.uint8)
    image[30:, :] = (130, 75, 35)
    cv2.putText(image, "HEAD", (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    encoded = base64.b64encode(cv2.imencode(".png", image)[1]).decode("ascii")

    style = infer_crop_style(encoded)

    assert style is not None
    row_styles = infer_table_row_fills(encoded, 2)
    assert row_styles[0][:2] == (1, "#234B82")


def test_table_rule_style_distinguishes_horizontal_rules_from_a_grid():
    horizontal = np.full((100, 300, 3), 255, dtype=np.uint8)
    for y in (5, 35, 95):
        cv2.line(horizontal, (0, y), (299, y), (0, 0, 0), 2)
    grid = horizontal.copy()
    for x in (5, 150, 295):
        cv2.line(grid, (x, 0), (x, 99), (0, 0, 0), 2)
    encode = lambda image: base64.b64encode(cv2.imencode(".png", image)[1]).decode("ascii")

    assert infer_table_rule_style(encode(horizontal)) == "horizontal"
    assert infer_table_rule_style(encode(grid)) == "grid"
