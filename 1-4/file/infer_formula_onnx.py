#!/usr/bin/env python3
"""PP-FormulaNet_plus-S ONNX FP16 inference example.

This file is a self-contained integration example for
PP-FormulaNet_plus-S_pure_onnx_fp16. It does not import PaddleOCR code.

Install dependencies:
    pip install onnxruntime numpy pillow opencv-python albumentations tokenizers ftfy

Single image:
    python infer_formula_onnx.py --image test/images/arrow_1.png

Directory:
    python infer_formula_onnx.py --image-dir test/images --output-json output.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import onnxruntime as ort
import albumentations as A
from PIL import Image, ImageOps
from tokenizers import Tokenizer

try:
    from ftfy import fix_text
except Exception:  # ftfy is recommended but not mandatory for basic decoding.
    fix_text = lambda text: text


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class UniMERNetTokenizerDecode:
    """Decode PP-FormulaNet token ids to LaTeX text."""

    def __init__(self, tokenizer_dir: Path):
        tokenizer_file = tokenizer_dir / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Missing tokenizer file: {tokenizer_file}")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))

    def _normalize_infer(self, text: str) -> str:
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = r"[\W_^\d]"
        names = []
        for matched in re.findall(text_reg, text):
            pattern = r"(\\[a-zA-Z]+)\s(?=\w)|\\[a-zA-Z]+\s(?=})"
            for item in re.findall(pattern, matched[0]):
                if item not in [
                    "\\operatorname",
                    "\\mathrm",
                    "\\text",
                    "\\mathbf",
                ] and item.strip():
                    text = text.replace(item, item + "XXXXXXX")
                    text = text.replace(" ", "")
                    names.append(text)
        if names:
            text = re.sub(text_reg, lambda _: str(names.pop(0)), text)
        new_text = text
        while True:
            text = new_text
            new_text = re.sub(
                rf"(?!\\ )({noletter})\s+?({noletter})", r"\1\2", text
            )
            new_text = re.sub(
                rf"(?!\\ )({noletter})\s+?({letter})", r"\1\2", new_text
            )
            new_text = re.sub(rf"({letter})\s+?({noletter})", r"\1\2", new_text)
            if new_text == text:
                break
        return text.replace("XXXXXXX", " ")

    @staticmethod
    def _remove_chinese_text_wrapping(formula: str) -> str:
        pattern = re.compile(r"\\text\s*{\s*([^}]*?[\u4e00-\u9fff]+[^}]*?)\s*}")
        return pattern.sub(lambda match: match.group(1), formula).replace('"', "")

    def decode_one(self, token_ids: np.ndarray) -> str:
        token_ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        eos_positions = np.argwhere(token_ids == 2)
        if len(eos_positions) > 0:
            token_ids = token_ids[: int(eos_positions[0][0]) + 1]
        text = self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        text = self._remove_chinese_text_wrapping(text)
        text = fix_text(text)
        return self._normalize_infer(text)

    def __call__(self, batch_token_ids: np.ndarray) -> List[str]:
        token_ids = np.asarray(batch_token_ids)
        if token_ids.ndim == 1:
            token_ids = token_ids[None, :]
        return [self.decode_one(row) for row in token_ids]


class FormulaONNXPredictor:
    """ONNX Runtime predictor for PP-FormulaNet_plus-S pure ONNX FP16."""

    def __init__(
        self,
        model_path: Path,
        tokenizer_dir: Path,
        provider: str = "cpu",
        threads: int = 4,
    ):
        self.model_path = Path(model_path)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.decoder = UniMERNetTokenizerDecode(self.tokenizer_dir)
        self.session = self._make_session(provider=provider, threads=threads)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _make_session(self, provider: str, threads: int) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.log_severity_level = 3

        if provider.lower() == "cuda":
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDAExecutionProvider is not available. "
                    f"Available providers: {available}"
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=providers
        )

    @staticmethod
    def _crop_margin(image: Image.Image) -> Image.Image:
        data = np.asarray(image.convert("L")).astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return image
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)
        coords = cv2.findNonZero(gray)
        if coords is None:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        if w == 0 or h == 0 or max(w, h) / min(w, h) > 200:
            return image
        return image.crop((x, y, x + w, y + h))

    @staticmethod
    def _resize_short_edge(image: Image.Image, size: int) -> Image.Image:
        width, height = image.size
        if width <= height:
            new_width = size
            new_height = int(size * height / width)
        else:
            new_height = size
            new_width = int(size * width / height)
        return image.resize((new_width, new_height), resample=Image.BILINEAR)

    @classmethod
    def preprocess(cls, image_path: Path, input_size=(384, 384)) -> np.ndarray:
        """Return float32 tensor with shape [1, 1, 384, 384]."""
        image = Image.open(image_path).convert("RGB")
        image = cls._crop_margin(image)
        if image.height == 0 or image.width == 0:
            raise ValueError(f"Invalid image size: {image_path}")

        image = cls._resize_short_edge(image, min(input_size))
        # Keep PIL's default thumbnail resampling, matching PaddleOCR.
        image.thumbnail((input_size[1], input_size[0]))
        delta_width = input_size[1] - image.width
        delta_height = input_size[0] - image.height
        pad_left = delta_width // 2
        pad_top = delta_height // 2
        padding = (
            pad_left,
            pad_top,
            delta_width - pad_left,
            delta_height - pad_top,
        )
        # PaddleOCR UniMERNetImgDecode uses ImageOps.expand without a fill
        # argument, so the padded area is black. Keep this behavior exactly.
        image = ImageOps.expand(image, padding)

        # Match PaddleOCR UniMERNetTestTransform and LatexImageFormat.
        transform = A.Compose(
            [
                A.ToGray(p=1.0),
                A.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
            ]
        )
        array = transform(image=np.asarray(image))["image"]
        tensor = array[:, :, 0][None, None, :, :].astype("float32")
        return tensor

    def predict_tokens(self, image_path: Path) -> tuple[np.ndarray, float]:
        image_tensor = self.preprocess(Path(image_path))
        start = time.perf_counter()
        token_ids = self.session.run([self.output_name], {self.input_name: image_tensor})[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        return token_ids, elapsed_ms

    def predict(self, image_path: Path) -> dict:
        token_ids, elapsed_ms = self.predict_tokens(image_path)
        formula = self.decoder(token_ids)[0]
        return {
            "input_path": str(image_path),
            "rec_formula": formula,
            "elapsed_ms": round(elapsed_ms, 3),
            "token_count": int(np.asarray(token_ids).shape[-1]),
        }


def collect_images(image_paths: Iterable[str], image_dir: str | None) -> List[Path]:
    images: List[Path] = [Path(item) for item in image_paths]
    if image_dir:
        images.extend(
            path
            for path in sorted(Path(image_dir).rglob("*"))
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
    missing = [str(path) for path in images if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing image(s): {missing}")
    if not images:
        raise ValueError("Pass --image or --image-dir.")
    return images


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "inference.onnx")
    parser.add_argument("--tokenizer-dir", type=Path, default=root / "tokenizer")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--image-dir")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictor = FormulaONNXPredictor(
        model_path=args.model,
        tokenizer_dir=args.tokenizer_dir,
        provider=args.provider,
        threads=args.threads,
    )
    rows = []
    for image_path in collect_images(args.image, args.image_dir):
        result = predictor.predict(image_path)
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
