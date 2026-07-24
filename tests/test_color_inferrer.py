import base64

import cv2
import numpy as np

from docflow.layout.color_inferrer import infer_crop_style, infer_table_row_fills


def test_crop_style_and_table_fill_are_inferred_from_region_pixels():
    image = np.full((60, 180, 3), 255, dtype=np.uint8)
    image[30:, :] = (130, 75, 35)
    cv2.putText(image, "HEAD", (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    encoded = base64.b64encode(cv2.imencode(".png", image)[1]).decode("ascii")

    style = infer_crop_style(encoded)

    assert style is not None
    row_styles = infer_table_row_fills(encoded, 2)
    assert row_styles[0][:2] == (1, "#234B82")
