"""Image restoration API."""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

os.environ["XFORMERS_DISABLED"] = "1"
warnings.filterwarnings("ignore", message="xFormers is.*")

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from pixrestore.build import build_model
from pixrestore.encoder import extract_features, load_encoder
from pixrestore.flow import PixelDiffusion

ROOT = Path(__file__).resolve().parent
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def offsets(length, tile, stride):
    if length <= tile:
        return [0]
    result = list(range(0, length - tile + 1, stride))
    if result[-1] != length - tile:
        result.append(length - tile)
    return result


def tile_plan(width, height, tile, overlap):
    padded = max(width, tile), max(height, tile)
    boxes = [(top, left) for top in offsets(padded[1], tile, tile - overlap)
             for left in offsets(padded[0], tile, tile - overlap)]
    return padded, boxes


def folder_tile_indices(paths, tile, overlap):
    names = []
    for path in paths:
        with Image.open(path) as image:
            _, boxes = tile_plan(*image.size, tile, overlap)
        names.extend(f"{path.stem}__{top:06d}_{left:06d}.png" for top, left in boxes)
    if len(names) != len(set(names)):
        raise ValueError("Input filenames must have distinct stems.")
    return {name: index for index, name in enumerate(sorted(names))}


class WatermarkRemover:
    """Load once, then restore images on the selected device."""

    def __init__(self, device="cuda:0", precision="fp16", weights_dir=None,
                 tile=None, overlap=128, steps=4, seed=0):
        self.config = SimpleNamespace(**json.loads((ROOT / "config.json").read_text()))
        self.device = torch.device(device)
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError("Use a CUDA device or cpu.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install CUDA-enabled PyTorch and check the GPU driver.")
        self.tile = self.config.resolution if tile is None else tile
        self.overlap, self.steps, self.seed = overlap, steps, seed
        if self.tile <= 0 or self.tile % self.config.patch_size or not 0 <= overlap < self.tile:
            raise ValueError("Tile size must be a positive multiple of 8, with 0 <= overlap < tile.")
        if steps < 1:
            raise ValueError("Steps must be positive.")
        if precision == "auto":
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    precision = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
            else:
                precision = "fp32"
        if precision not in ("bf16", "fp16", "fp32"):
            raise ValueError("Precision must be auto, bf16, fp16 or fp32.")
        if self.device.type == "cpu" and precision != "fp32":
            raise ValueError("CPU execution uses fp32.")
        if self.device.type == "cuda" and precision == "bf16":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("This GPU does not support bf16; use fp32.")
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                      "fp32": torch.float32}[precision]
        weights = ROOT / "weights" if weights_dir is None else Path(weights_dir).expanduser().resolve()
        self.model = build_model(self.config)
        self.model.load_state_dict(load_file(str(weights / "restorer.safetensors"), device="cpu"), strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.encoder = load_encoder(weights / "encoder.safetensors", self.device)
        self.flow = PixelDiffusion(self.config)
        window = np.hanning(self.tile).astype(np.float64)
        self.window = np.maximum(np.outer(window, window), 1e-3)

    @torch.inference_mode()
    def _restore_tile(self, image, index):
        torch.manual_seed(self.seed + index)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed + index)
        lq = pil_to_tensor(image).unsqueeze(0).to(self.device).float().div_(127.5).sub_(1)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype,
                            enabled=self.dtype != torch.float32):
            features = extract_features(self.encoder, lq, self.config.encoder_layers,
                                        self.config.encoder_input_size)
            restored = self.flow.sample_multistep_fm(
                self.model, lq, venc_fea=features, n_steps=self.steps, schedule="linear")
        restored = restored[0].float().cpu()
        if not torch.isfinite(restored).all().item():
            raise RuntimeError("Non-finite restoration output; try precision='bf16' or 'fp32'.")
        restored = restored.add(1).mul(0.5).clamp(0, 1)
        return np.asarray(to_pil_image(restored).convert("RGB"))

    def restore(self, image, *, tile_indices=None):
        """Return original and restored PIL images."""
        if isinstance(image, Image.Image):
            orientation = image.getexif().get(274, 1)
            original = image.convert("RGB")
        else:
            with Image.open(image) as source:
                orientation = source.getexif().get(274, 1)
                original = source.convert("RGB")
        width, height = original.size
        padded, boxes = tile_plan(width, height, self.tile, self.overlap)
        canvas = ImageOps.expand(original, (0, 0, padded[0] - width, padded[1] - height), fill=0)
        if tile_indices is None:
            tile_indices = list(range(len(boxes)))
        if len(tile_indices) != len(boxes):
            raise ValueError("Tile index count does not match the image.")
        restored_sum = np.zeros((padded[1], padded[0], 3), dtype=np.float64)
        weight_sum = np.zeros((padded[1], padded[0]), dtype=np.float64)
        for index, (top, left) in zip(tile_indices, boxes):
            tile = canvas.crop((left, top, left + self.tile, top + self.tile))
            restored = self._restore_tile(tile, index)
            area = np.s_[top:top + self.tile, left:left + self.tile]
            restored_sum[area] += restored * self.window[..., None]
            weight_sum[area] += self.window
        def finish(array):
            value = array / weight_sum[..., None]
            return Image.fromarray(np.clip(value[:height, :width] + 0.5, 0, 255).astype(np.uint8))
        restored = finish(restored_sum)
        transpose = TRANSPOSE.get(orientation)
        if transpose is not None:
            original, restored = [im.transpose(transpose) for im in (original, restored)]
        return dict(original=original, restored=restored)
