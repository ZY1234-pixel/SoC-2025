"""Deployment contracts, exercised without loading GPU model weights."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import unittest
import tempfile
from unittest.mock import patch
import torch
from detectron2.structures import Boxes, Instances

from instance_seg.cli import parse_args
from instance_seg.geometry import touches_internal_crop_edge
from instance_seg.io import input_images
from instance_seg.refinement import should_drop_unconfirmed_low_score
from instance_seg.settings import DEFAULT_INPUT, PACKAGE_ROOT


def test_cli_is_independent_of_cwd_and_model_imports(tmp_path):
    assert parse_args([]).input == DEFAULT_INPUT
    assert parse_args([], output_mode="production").output_mode == "production"
    assert parse_args(["--output-mode", "debug"], output_mode="production").output_mode == "debug"
    assert DEFAULT_INPUT == PACKAGE_ROOT.parent / "TEST_dewarp"
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from instance_seg.cli import parse_args; parse_args([]); "
        'assert "torch" not in sys.modules; '
        'assert "detectron2" not in sys.modules'
    )
    subprocess.run([sys.executable, "-c", code, str(PACKAGE_ROOT)], check=True, cwd=tmp_path)


def test_invalid_arguments_fail_before_model_loading(args):
    with unittest.TestCase().assertRaises(ValueError):
        parse_args(args)


def test_recursive_discovery_ignores_directories_with_image_extensions(tmp_path):
    (tmp_path / "folder.jpg").mkdir()
    image = tmp_path / "folder.jpg" / "scan.PNG"
    image.touch()
    (tmp_path / "notes.txt").touch()
    assert list(input_images(tmp_path)) == [image]


def test_roi_edges_distinguish_image_boundary_from_internal_crop():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:3, 3:7] = True
    assert not touches_internal_crop_edge(mask, (0, 0, 10, 10), (10, 10))
    assert touches_internal_crop_edge(mask, (0, 5, 10, 15), (20, 20))


def test_low_score_removal_requires_a_contradicting_roi_candidate():
    diagnostic = dict(
        attempted=True,
        supports_detection_verification=True,
        accepted=False,
        candidate_count=1,
        reason="low_match_iou",
        iou=0.1,
    )
    assert should_drop_unconfirmed_low_score(0.3, diagnostic, 0.5, 0.2)
    assert not should_drop_unconfirmed_low_score(0.9, diagnostic, 0.5, 0.2)
    diagnostic["candidate_count"] = 0
    assert not should_drop_unconfirmed_low_score(0.3, diagnostic, 0.5, 0.2)


def test_batch_outputs_keep_nested_names_masks_and_overlays(
    tmp_path, mode
):
    from instance_seg import pipeline

    inputs = tmp_path / "input"
    for folder in ["a", "b"]:
        (inputs / folder).mkdir(parents=True)
        cv2.imwrite(
            str(inputs / folder / "same.jpg"), np.full((32, 48, 3), 128, np.uint8)
        )
    checkpoint = tmp_path / "placeholder.pth"
    checkpoint.touch()

    class FakeModel:
        def __call__(self, records):
            record = records[0]
            h, w = record["height"], record["width"]
            instances = Instances((h, w))
            masks = torch.zeros((1, h, w), dtype=torch.bool)
            masks[:, 8:24, 12:36] = True
            instances.pred_masks = masks
            instances.pred_boxes = Boxes(torch.tensor([[12.0, 8.0, 36.0, 24.0]]))
            instances.scores = torch.tensor([0.9])
            return [{"instances": instances}]

    def fake_refinement(**kwargs):
        debug = kwargs["image_bgr"].copy() if kwargs["create_debug_image"] else None
        return kwargs["coarse_mask"], {"attempted": True, "accepted": True}, debug

    output = tmp_path / mode
    args = parse_args(
        [
            str(inputs),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--output-mode",
            mode,
            "--opts",
            "MODEL.DEVICE",
            "cpu",
        ]
    )
    with patch.object(pipeline, "load_model", lambda *args: FakeModel()), patch.object(
        pipeline, "refine_instance_with_roi", fake_refinement
    ), patch.object(torch.cuda, "is_available", lambda: False):
        pipeline.run(args)
    for folder in ["a", "b"]:
        mask = output / "instances" / folder / "same_instance_001.png"
        overlay = output / "overlays" / folder / "same.jpg"
        debug = output / "roi_debug" / folder / "same_instance_001.jpg"
        assert mask.exists() == (mode != "debug")
        assert overlay.exists() == (mode != "production")
        assert debug.exists() == (mode == "debug")
        if mask.exists():
            pixels = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
            assert pixels.shape == (32, 48)
            assert int((pixels == 255).sum()) == 16 * 24
        if overlay.exists():
            pixels = cv2.imread(str(overlay))
            assert pixels.shape == (32, 48, 3)
            assert not np.all(pixels == 128)


class DeploymentTests(unittest.TestCase):
    def test_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            test_cli_is_independent_of_cwd_and_model_imports(Path(folder))

    def test_invalid_arguments(self):
        for args in [["--score-threshold", "nan"], ["--score-threshold", "1.1"],
                     ["--max-images", "-1"], ["--roi-refine-size", "33"]]:
            with self.subTest(args=args):
                test_invalid_arguments_fail_before_model_loading(args)

    def test_recursive_discovery(self):
        with tempfile.TemporaryDirectory() as folder:
            test_recursive_discovery_ignores_directories_with_image_extensions(Path(folder))

    def test_crop_boundaries(self):
        test_roi_edges_distinguish_image_boundary_from_internal_crop()

    def test_low_score_gates(self):
        test_low_score_removal_requires_a_contradicting_roi_candidate()

    def test_output_modes(self):
        for mode in ["production", "debug"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                test_batch_outputs_keep_nested_names_masks_and_overlays(Path(folder), mode)


if __name__ == "__main__":
    unittest.main()
