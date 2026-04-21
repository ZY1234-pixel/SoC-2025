from __future__ import annotations

from pathlib import Path

from model import RuntimePaths


def test_runtime_paths_default_layout_model_points_to_headfloat100() -> None:
    paths = RuntimePaths.discover()
    assert paths.layout_model == (
        Path(__file__).resolve().parents[1]
        / "Code"
        / "models"
        / "layout"
        / "doclayout_yolo_docstructbench_headfloat100_runtime"
    )
