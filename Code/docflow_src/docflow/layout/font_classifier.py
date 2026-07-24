"""Font classification support for scanned text blocks.

The classifier is intentionally lazy: importing this module does not import
torch/torchvision. If those optional dependencies are unavailable, the recovery
pipeline can continue with its normal font defaults.
"""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING

from PIL import Image, ImageEnhance, ImageOps

from docflow.model.blocks.text_block import TextBlock
from docflow.schema.models import BlockStyle

if TYPE_CHECKING:
    from docflow.config import RecoveryConfig
    from docflow.model.base import Block
    from docflow.model.page import Page


LABELS: Sequence[str] = ("仿宋", "宋体", "楷体", "黑体", "其他")

# Labels are model classes; DOCX needs concrete CJK font family names.  The
# catch-all class is intentionally not rendered as a font name and falls back to
# the document default during style inference/rendering.
FONT_FAMILY_BY_LABEL: Dict[str, Optional[str]] = {
    "仿宋": "仿宋",
    "宋体": "宋体",
    "楷体": "楷体",
    "黑体": "黑体",
    "其他": None,
}


@dataclass(frozen=True)
class FontPrediction:
    label: str
    confidence: float
    margin: float
    scores: Dict[str, float]
    accepted: bool


class FixedWidthLineTransform:
    """Match the inference preprocessing used by the training demo."""

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
        import torch
        import torchvision.transforms.functional as TF

        source_width, source_height = image.size
        crop_width = min(crop_width, source_width)
        if source_width > crop_width:
            image = TF.crop(image, 0, left, source_height, crop_width)
            source_width = crop_width

        resized_width = max(1, round(source_width * self.height / source_height))
        image = TF.resize(image, [self.height, resized_width], antialias=True)

        if resized_width < self.width:
            pad_total = self.width - resized_width
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            image = TF.pad(image, [pad_left, 0, pad_right, 0], fill=self.fill)
        elif resized_width > self.width:
            image = TF.crop(image, 0, 0, self.height, self.width)

        tensor = TF.to_tensor(image)
        if self.grayscale:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

    def __call__(self, image: Image.Image):
        import torch

        image = image.convert("L" if self.grayscale else "RGB")
        source_width, source_height = image.size
        if source_width <= 0 or source_height <= 0:
            raise ValueError("font crop has empty size")
        crop_source_width = max(1, round(self.width * source_height / self.height))

        if source_width <= crop_source_width:
            tensor = self._window_to_tensor(image, 0, crop_source_width)
            return torch.stack([tensor.clone() for _ in range(self.eval_crops)])

        max_left = source_width - crop_source_width
        if self.eval_crops == 1:
            left_positions = [max_left // 2]
        else:
            left_positions = [
                round(max_left * i / (self.eval_crops - 1))
                for i in range(self.eval_crops)
            ]
        crops = [
            self._window_to_tensor(image, int(left), crop_source_width)
            for left in left_positions
        ]
        return torch.stack(crops)


class FontClassifier:
    """MobileNetV3 font classifier wrapper."""

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
        grayscale: bool = False,
        crop_padding_px: int = 3,
        max_line_crops_per_block: int = 5,
        binarize: bool = False,
        contrast: float = 1.0,
        invert: bool = False,
    ) -> None:
        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        self.device_name = device
        self.temperature = max(float(temperature), 1e-6)
        self.reject_threshold = float(reject_threshold)
        self.margin_threshold = float(margin_threshold)
        self.crop_padding_px = max(0, int(crop_padding_px))
        self.max_line_crops_per_block = max(1, int(max_line_crops_per_block))
        self.binarize = bool(binarize)
        self.contrast = max(float(contrast), 0.0)
        self.invert = bool(invert)
        self.transform = FixedWidthLineTransform(
            height=height,
            width=width,
            grayscale=grayscale,
            eval_crops=eval_crops,
        )
        self._model = None
        self._device = None

    @classmethod
    def from_config(cls, config: "RecoveryConfig") -> "FontClassifier":
        return cls(
            checkpoint_path=getattr(config, "font_model_path", ""),
            device=getattr(config, "font_classifier_device", "auto"),
            height=getattr(config, "font_classifier_height", 48),
            width=getattr(config, "font_classifier_width", 768),
            eval_crops=getattr(config, "font_classifier_eval_crops", 5),
            temperature=getattr(config, "font_classifier_temperature", 1.0),
            reject_threshold=getattr(config, "font_classifier_reject_threshold", 0.6),
            margin_threshold=getattr(config, "font_classifier_margin_threshold", 0.25),
            grayscale=getattr(config, "font_classifier_grayscale", False),
            crop_padding_px=getattr(config, "font_classifier_crop_padding_px", 3),
            max_line_crops_per_block=getattr(config, "font_classifier_max_line_crops_per_block", 5),
            binarize=getattr(config, "font_classifier_binarize", False),
            contrast=getattr(config, "font_classifier_contrast", 1.0),
            invert=getattr(config, "font_classifier_invert", False),
        )

    @staticmethod
    def _resolve_checkpoint(checkpoint_path: str) -> Path:
        checkpoint_text = str(checkpoint_path or "").strip()
        candidates = []
        if checkpoint_text:
            raw = Path(checkpoint_text).expanduser()
            candidates.append(raw)
            if not raw.is_absolute():
                candidates.append(Path.cwd() / raw)
        code_root = Path(__file__).resolve().parents[3]
        candidates.append(code_root / "models" / "font" / "mobilenetv3.ckpt")
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(f"font checkpoint not found: {checkpoint_path}")

    def _select_device(self):
        import torch

        name = str(self.device_name or "auto").lower()
        if name == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if name.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(name)

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        import torch
        import torch.nn as nn

        try:
            import torchvision
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "font classification requires torchvision; install it from Code/requirement.txt"
            ) from exc

        model = torchvision.models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(LABELS))

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        expected_classes = int(
            checkpoint.get("hyper_parameters", {}).get("num_classes", len(LABELS))
        )
        if expected_classes != len(LABELS):
            raise RuntimeError(
                "font checkpoint class count mismatch: "
                f"checkpoint has {expected_classes}, DocFlow expects {len(LABELS)} "
                f"labels={list(LABELS)}"
            )
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_state = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
        if not model_state:
            model_state = dict(state_dict)
        model.load_state_dict(model_state, strict=True)
        self._device = self._select_device()
        model.to(self._device)
        model.eval()
        self._model = model
        return model

    def classify_page(self, page: "Page", blocks: Iterable["Block"]) -> dict:
        image = _load_page_image(page)
        if image is None:
            return {"enabled": True, "available": False, "reason": "page_image_missing", "applied": 0}
        try:
            self._ensure_model()
        except Exception as exc:
            return {
                "enabled": True,
                "available": False,
                "reason": str(exc),
                "checkpoint": str(self.checkpoint_path),
                "applied": 0,
            }

        applied = 0
        accepted = 0
        skipped = 0
        unavailable_reason = None
        for block in blocks:
            if not isinstance(block, TextBlock):
                continue
            if block.style is not None and block.style.font_family:
                skipped += 1
                continue
            try:
                prediction = self.predict_block(image, block)
            except RuntimeError as exc:
                unavailable_reason = str(exc)
                break
            except Exception:
                skipped += 1
                continue
            if prediction is None:
                skipped += 1
                continue
            applied += 1
            apply_font_prediction(block, prediction)
            if prediction.accepted:
                accepted += 1

        if unavailable_reason:
            return {
                "enabled": True,
                "available": False,
                "reason": unavailable_reason,
                "checkpoint": str(self.checkpoint_path),
                "applied": 0,
            }
        return {
            "enabled": True,
            "available": True,
            "checkpoint": str(self.checkpoint_path),
            "labels": list(LABELS),
            "applied": applied,
            "accepted": accepted,
            "skipped": skipped,
        }

    def predict_block(self, page_image: Image.Image, block: TextBlock) -> Optional[FontPrediction]:
        crops = self._line_crops(page_image, block)
        if not crops:
            return None
        return self._predict_crops(crops)

    def predict_image(self, image: Image.Image) -> FontPrediction:
        return self._predict_crops([image])

    def _predict_crops(self, crops: Sequence[Image.Image]) -> FontPrediction:
        import torch

        model = self._ensure_model()
        tensors = [self.transform(self._preprocess_crop(crop)) for crop in crops]
        batch = torch.stack(tensors).to(self._device or self._select_device())
        with torch.inference_mode():
            batch = batch.to(self._device)
            batch_size, num_crops, channels, height, width = batch.shape
            logits = model(batch.reshape(batch_size * num_crops, channels, height, width))
            logits = logits.view(batch_size, num_crops, -1).mean(dim=1)
            prob = torch.softmax(logits / self.temperature, dim=1).mean(dim=0).detach().cpu()

        scores = {label: float(prob[index]) for index, label in enumerate(LABELS)}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_label, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score - second_score
        accepted = top_score >= self.reject_threshold and margin >= self.margin_threshold
        return FontPrediction(
            label=top_label,
            confidence=top_score,
            margin=margin,
            scores=scores,
            accepted=accepted,
        )

    def _preprocess_crop(self, image: Image.Image) -> Image.Image:
        processed = image.convert("RGB")
        if self.invert:
            processed = ImageOps.invert(processed)
        if self.binarize:
            gray = ImageOps.grayscale(processed)
            gray = ImageOps.autocontrast(gray)
            if self.contrast and abs(self.contrast - 1.0) > 1e-6:
                gray = ImageEnhance.Contrast(gray).enhance(self.contrast)
            threshold = _otsu_threshold(gray)
            gray = gray.point(lambda p: 255 if p > threshold else 0)
            processed = Image.merge("RGB", (gray, gray, gray))
        return processed

    def _line_crops(self, page_image: Image.Image, block: TextBlock) -> List[Image.Image]:
        boxes = []
        for line in block.lines:
            text = (line.text or "").strip()
            if not text or line.x1 is None or line.y1 is None or line.x2 is None or line.y2 is None:
                continue
            boxes.append((line.x1, line.y1, line.x2, line.y2))
        if not boxes:
            boxes.append((block.bbox.x1, block.bbox.y1, block.bbox.x2, block.bbox.y2))

        boxes = _select_representative_boxes(boxes, self.max_line_crops_per_block)
        crops = []
        for box in boxes:
            crop_box = _expand_box(box, self.crop_padding_px, page_image.size)
            if crop_box[2] - crop_box[0] < 4 or crop_box[3] - crop_box[1] < 4:
                continue
            crops.append(page_image.crop(crop_box))
        return crops


def apply_font_prediction(block: TextBlock, prediction: FontPrediction) -> None:
    if block.style is None:
        block.style = BlockStyle()
    font_family = FONT_FAMILY_BY_LABEL.get(prediction.label)
    if prediction.accepted and block.style.font_family is None and font_family:
        block.style.font_family = font_family
    if block.attributes is None:
        block.attributes = {}
    block.attributes["font_prediction"] = {
        "label": prediction.label,
        "font_family": font_family,
        "confidence": round(prediction.confidence, 4),
        "margin": round(prediction.margin, 4),
        "accepted": bool(prediction.accepted),
        "scores": {
            label: round(score, 4)
            for label, score in prediction.scores.items()
        },
    }


def _select_representative_boxes(
    boxes: Sequence[tuple[float, float, float, float]],
    limit: int,
) -> List[tuple[float, float, float, float]]:
    if len(boxes) <= limit:
        return list(boxes)
    step = (len(boxes) - 1) / max(limit - 1, 1)
    return [boxes[round(index * step)] for index in range(limit)]


def _expand_box(
    box: tuple[float, float, float, float],
    padding: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    return (
        max(0, int(round(x1)) - padding),
        max(0, int(round(y1)) - padding),
        min(width, int(round(x2)) + padding),
        min(height, int(round(y2)) + padding),
    )


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
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += level * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )
        if variance > best_variance:
            best_variance = variance
            best_threshold = level
    return int(best_threshold)


def _load_page_image(page: "Page") -> Optional[Image.Image]:
    image_path = getattr(page, "image_path", None)
    if image_path:
        path = Path(str(image_path)).expanduser()
        if path.exists():
            return Image.open(path).convert("RGB")

    image_base64 = getattr(page, "image_base64", None)
    if image_base64:
        try:
            if "," in image_base64[:80]:
                image_base64 = image_base64.split(",", 1)[1]
            data = base64.b64decode(image_base64, validate=False)
            return Image.open(io.BytesIO(data)).convert("RGB")
        except (binascii.Error, OSError, ValueError):
            return None
    return None
