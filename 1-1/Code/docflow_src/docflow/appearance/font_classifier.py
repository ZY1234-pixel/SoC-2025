"""Lazy four-font classification for recognition crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


LABELS: Sequence[str] = ("仿宋", "宋体", "楷体", "黑体", "其他")
FONT_FAMILY_BY_LABEL: Dict[str, Optional[str]] = {
    "仿宋": "仿宋",
    "宋体": "宋体",
    "楷体": "楷体",
    "黑体": "黑体",
    "其他": None,
}
FONT_INK_HEIGHT_RATIO = {"宋体": 0.86, "黑体": 0.90, "楷体": 0.84, "仿宋": 0.86}


@dataclass(frozen=True)
class FontPrediction:
    label: str
    confidence: float
    margin: float
    scores: Dict[str, float]
    accepted: bool


class FixedWidthLineTransform:
    """Match the preprocessing used to train the font classifier."""

    def __init__(
        self,
        height: int = 48,
        width: int = 768,
        fill: int = 255,
        grayscale: bool = False,
        eval_crops: int = 5,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.fill = int(fill)
        self.grayscale = bool(grayscale)
        self.eval_crops = max(1, int(eval_crops))

    def _window_to_tensor(self, image: Image.Image, left: int, crop_width: int):
        import torchvision.transforms.functional as transform

        source_width, source_height = image.size
        crop_width = min(crop_width, source_width)
        if source_width > crop_width:
            image = transform.crop(image, 0, left, source_height, crop_width)
            source_width = crop_width

        resized_width = max(1, round(source_width * self.height / source_height))
        image = transform.resize(image, [self.height, resized_width], antialias=True)
        if resized_width < self.width:
            padding = self.width - resized_width
            image = transform.pad(image, [padding // 2, 0, padding - padding // 2, 0], fill=self.fill)
        elif resized_width > self.width:
            image = transform.crop(image, 0, 0, self.height, self.width)
        tensor = transform.to_tensor(image)
        return tensor.repeat(3, 1, 1) if self.grayscale else tensor

    def __call__(self, image: Image.Image):
        import torch

        image = image.convert("L" if self.grayscale else "RGB")
        source_width, source_height = image.size
        if source_width <= 0 or source_height <= 0:
            raise ValueError("font crop has empty size")
        crop_width = max(1, round(self.width * source_height / self.height))
        if source_width <= crop_width:
            tensor = self._window_to_tensor(image, 0, crop_width)
            return torch.stack([tensor.clone() for _ in range(self.eval_crops)])

        max_left = source_width - crop_width
        positions = [max_left // 2] if self.eval_crops == 1 else [
            round(max_left * index / (self.eval_crops - 1))
            for index in range(self.eval_crops)
        ]
        return torch.stack([self._window_to_tensor(image, int(left), crop_width) for left in positions])


class FontClassifier:
    """MobileNetV3 classifier exposed only through immutable image predictions."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        height: int = 48,
        width: int = 768,
        eval_crops: int = 5,
        temperature: float = 1.0,
        reject_threshold: float = 0.6,
        margin_threshold: float = 0.25,
        grayscale: bool = True,
        binarize: bool = False,
        contrast: float = 1.0,
        invert: bool = False,
    ) -> None:
        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        self.device_name = device
        self.temperature = max(float(temperature), 1e-6)
        self.reject_threshold = float(reject_threshold)
        self.margin_threshold = float(margin_threshold)
        self.binarize = bool(binarize)
        self.contrast = max(float(contrast), 0.0)
        self.invert = bool(invert)
        self.transform = FixedWidthLineTransform(height, width, grayscale=grayscale, eval_crops=eval_crops)
        self._model = None

    @staticmethod
    def _resolve_checkpoint(checkpoint_path: str) -> Path:
        checkpoint_text = str(checkpoint_path or "").strip()
        candidates = []
        if checkpoint_text:
            raw = Path(checkpoint_text).expanduser()
            candidates.append(raw if raw.is_absolute() else Path.cwd() / raw)
        candidates.append(Path(__file__).resolve().parents[3] / "models_openvino" / "font_openvino" / "mobilenetv3.xml")
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"font checkpoint not found: {checkpoint_path}")

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from docflow.inference import OpenVINOInferSession

        self._model = OpenVINOInferSession(self.checkpoint_path)
        return self._model

    def predict_image(self, image: Image.Image) -> FontPrediction:
        model = self._ensure_model()
        batch = self.transform(self._preprocess_crop(image)).numpy()
        logits = model({"images": batch})[0].mean(axis=0) / self.temperature
        logits -= logits.max()
        probability = np.exp(logits)
        probability /= probability.sum()
        scores = {label: float(probability[index]) for index, label in enumerate(LABELS)}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        label, confidence = ranked[0]
        margin = confidence - ranked[1][1]
        return FontPrediction(
            label,
            confidence,
            margin,
            scores,
            confidence >= self.reject_threshold and margin >= self.margin_threshold,
        )

    def _preprocess_crop(self, image: Image.Image) -> Image.Image:
        processed = image.convert("RGB")
        if self.invert:
            processed = ImageOps.invert(processed)
        if self.binarize:
            gray = ImageOps.autocontrast(ImageOps.grayscale(processed))
            if self.contrast and abs(self.contrast - 1.0) > 1e-6:
                gray = ImageEnhance.Contrast(gray).enhance(self.contrast)
            threshold = _otsu_threshold(gray)
            gray = gray.point(lambda pixel: 255 if pixel > threshold else 0)
            processed = Image.merge("RGB", (gray, gray, gray))
        return processed


def _otsu_threshold(image: Image.Image) -> int:
    histogram = image.histogram()
    total = sum(histogram)
    if total <= 0:
        return 180
    sum_total = sum(level * count for level, count in enumerate(histogram))
    sum_background = 0.0
    weight_background = 0
    best_threshold = 180
    best_variance = -1.0
    for level, count in enumerate(histogram):
        weight_background += count
        if not weight_background:
            continue
        weight_foreground = total - weight_background
        if not weight_foreground:
            break
        sum_background += level * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = level
    return int(best_threshold)
