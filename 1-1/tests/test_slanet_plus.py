import numpy as np

from docflow.vendor.table_rec.slanet_plus import SLANetPlusTableEngine


def test_slanet_plus_reuses_project_ocr_results() -> None:
    ocr_result = [
        (
            np.asarray([[10, 20], [90, 20], [90, 50], [10, 50]], dtype=np.float32),
            "cell text",
            0.98,
        )
    ]

    boxes, results = SLANetPlusTableEngine._format_ocr(ocr_result, (200, 300))

    assert boxes.tolist() == [[10.0, 20.0, 90.0, 50.0]]
    assert results == [("cell text", 0.98)]
