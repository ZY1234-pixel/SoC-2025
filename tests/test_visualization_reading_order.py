from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code" / "docflow_src"))

from docflow.utils.visualization import draw_reading_order_comparison, draw_reading_order_map, draw_sorted_layout


def _blocks():
    return [
        {
            "id": "a",
            "type": "title",
            "bbox": [20, 20, 180, 80],
            "col_count": 2,
            "col_index": 0,
            "spanned_cols": [0, 1],
        },
        {
            "id": "b",
            "type": "text",
            "bbox": [20, 100, 90, 180],
            "col_count": 2,
            "col_index": 0,
            "spanned_cols": [0],
        },
        {
            "id": "c",
            "type": "text",
            "bbox": [110, 100, 180, 180],
            "col_count": 2,
            "col_index": 1,
            "spanned_cols": [1],
        },
    ]


def test_draw_reading_order_map_returns_image():
    image = np.full((240, 220, 3), 255, dtype=np.uint8)
    vis = draw_reading_order_map(image, _blocks(), title="XY-Cut++ Reading Order")
    assert vis.shape[0] > image.shape[0]
    assert vis.shape[1] > image.shape[1]
    assert vis.dtype == image.dtype
    assert np.any(vis != 255)


def test_draw_reading_order_comparison_concatenates_panels():
    image = np.full((240, 220, 3), 255, dtype=np.uint8)
    vis = draw_reading_order_comparison(image, _blocks(), list(reversed(_blocks())))
    single = draw_reading_order_map(image, _blocks(), title="Legacy Reading Order")
    assert vis.shape[0] == single.shape[0]
    assert vis.shape[1] == single.shape[1] * 2 + 24
    assert np.any(vis != 255)


def test_draw_reading_order_map_keeps_overlapped_spanning_border_visible():
    image = np.full((260, 240, 3), 255, dtype=np.uint8)
    blocks = [
        {
            "id": "span",
            "type": "title",
            "bbox": [20, 20, 210, 150],
            "col_count": 2,
            "col_index": 0,
            "spanned_cols": [0, 1],
        },
        {
            "id": "left",
            "type": "text",
            "bbox": [20, 95, 105, 200],
            "col_count": 2,
            "col_index": 0,
            "spanned_cols": [0],
        },
    ]

    vis = draw_reading_order_map(image, blocks, title="XY-Cut++ Reading Order")

    # Large spanning block's left border should remain visible above the overlap.
    assert np.any(vis[128:158, 30:38] != 255)
    # The local block should also remain visible in the overlap region.
    assert np.any(vis[160:178, 32:47] != 255)


def test_draw_sorted_layout_adds_top_margin_for_page_edge_title():
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    blocks = [
        {
            "id": "title",
            "type": "title",
            "bbox": [10, 0, 170, 40],
            "col_count": 2,
            "col_index": 0,
            "spanned_cols": [0, 1],
        }
    ]

    vis = draw_sorted_layout(image, blocks)

    assert vis.shape[0] > image.shape[0]
    assert vis.shape[1] > image.shape[1]
    # Extra top band should exist and contain annotation pixels.
    assert np.any(vis[:24, 8:120] != 255)
    # The title interior should remain mostly untouched; only the border should be present.
    assert np.all(vis[36:64, 24:156] == 255)
