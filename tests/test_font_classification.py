from __future__ import annotations

from docflow.config import RecoveryConfig
from docflow.layout.font_classifier import (
    FixedWidthLineTransform,
    FontClassifier,
    FontPrediction,
    apply_font_prediction,
)
from docflow.model.base import BBox, BlockType
from docflow.model.blocks.text_block import TextBlock, TextLine
from docflow.model.page import Page
from docflow.pipeline import RecoveryPipeline
from docflow.schema.models import BlockStyle


def _text_block(style=None) -> TextBlock:
    return TextBlock(
        bbox=BBox(0, 0, 100, 30),
        block_type=BlockType.TEXT,
        lines=[TextLine(text="测试文本", text_region=[[0, 0], [100, 0], [100, 30], [0, 30]])],
        style=style,
    )


def test_apply_font_prediction_sets_block_style_font_family():
    block = _text_block()
    prediction = FontPrediction(
        label="楷体",
        confidence=0.91,
        margin=0.42,
        scores={"楷体": 0.91},
        accepted=True,
    )

    apply_font_prediction(block, prediction)

    assert block.style is not None
    assert block.style.font_family == "楷体"
    assert block.attributes["font_prediction"]["label"] == "楷体"
    assert block.attributes["font_prediction"]["font_family"] == "楷体"


def test_apply_font_prediction_other_does_not_set_font_family():
    block = _text_block()
    prediction = FontPrediction(
        label="其他",
        confidence=0.95,
        margin=0.5,
        scores={"其他": 0.95},
        accepted=True,
    )

    apply_font_prediction(block, prediction)

    assert block.style is not None
    assert block.style.font_family is None
    assert block.attributes["font_prediction"]["label"] == "其他"
    assert block.attributes["font_prediction"]["font_family"] is None


def test_apply_font_prediction_rejected_records_scores_without_setting_font():
    block = _text_block()
    prediction = FontPrediction(
        label="宋体",
        confidence=0.65,
        margin=0.2,
        scores={"宋体": 0.65, "黑体": 0.45},
        accepted=False,
    )

    apply_font_prediction(block, prediction)

    assert block.style is not None
    assert block.style.font_family is None
    assert block.attributes["font_prediction"]["label"] == "宋体"
    assert block.attributes["font_prediction"]["font_family"] == "宋体"
    assert block.attributes["font_prediction"]["accepted"] is False
    assert block.attributes["font_prediction"]["scores"] == {"宋体": 0.65, "黑体": 0.45}


def test_pipeline_font_classifier_preserves_explicit_font(monkeypatch):
    explicit = _text_block(style=BlockStyle(font_family="宋体"))
    inferred = _text_block()
    page = Page(index=0, image_width=200, image_height=100)

    class FakeClassifier:
        @classmethod
        def from_config(cls, config):
            return cls()

        def classify_page(self, page, blocks):
            for block in blocks:
                if isinstance(block, TextBlock) and not (block.style and block.style.font_family):
                    apply_font_prediction(
                        block,
                        FontPrediction(
                            label="黑体",
                            confidence=0.95,
                            margin=0.5,
                            scores={"黑体": 0.95},
                            accepted=True,
                        ),
                    )
            return {"enabled": True, "available": True, "applied": 1, "accepted": 1}

    monkeypatch.setattr("docflow.pipeline.FontClassifier", FakeClassifier)
    pipeline = RecoveryPipeline(RecoveryConfig(font_classification_enabled=True))
    stats = pipeline._classify_block_fonts(page, [explicit, inferred])

    assert stats["accepted"] == 1
    assert explicit.style.font_family == "宋体"
    assert inferred.style.font_family == "黑体"


def test_font_classifier_grayscale_is_default_and_removes_color(tmp_path):
    import pytest
    import torch
    from PIL import Image, ImageDraw

    pytest.importorskip("torchvision")

    config = RecoveryConfig()
    classifier = FontClassifier.from_config(config)
    assert classifier.transform.grayscale is True
    assert classifier.binarize is False
    assert classifier.contrast == config.font_classifier_contrast

    image = Image.new("RGB", (96, 32), (246, 235, 220))
    draw = ImageDraw.Draw(image)
    draw.text((8, 6), "红字", fill=(186, 34, 48))

    tensor = classifier.transform(classifier._preprocess_crop(image))
    if tensor.ndim == 4:
        tensor = tensor[0]

    assert tensor.shape[0] == 3
    assert torch.equal(tensor[0], tensor[1])
    assert torch.equal(tensor[1], tensor[2])
    assert torch.unique(tensor).numel() > 2


def test_font_classifier_default_acceptance_thresholds():
    config = RecoveryConfig()
    classifier = FontClassifier.from_config(config)

    assert classifier.reject_threshold == 0.6
    assert classifier.margin_threshold == 0.25


def test_fixed_width_transform_matches_training_demo(monkeypatch):
    import sys

    import pytest
    import torch
    from PIL import Image, ImageDraw

    pytest.importorskip("torchvision")

    source_root = "/home/lyq/projects/YuzuMarker.FontDetection"
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from train_4class_mobilenetv3 import FixedWidthLineTransform as DemoTransform

    image = Image.new("RGB", (320, 46), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 8), "DocFlow 字体测试 123", fill="black")

    ours = FixedWidthLineTransform(height=48, width=768, grayscale=False, eval_crops=5)
    demo = DemoTransform(height=48, width=768, train=False, grayscale=False, eval_crops=5)

    assert torch.equal(ours(image), demo(image))
