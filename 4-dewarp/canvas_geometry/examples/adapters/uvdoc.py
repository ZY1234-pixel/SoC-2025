"""Lazy UVDoc loader so the core estimator remains model-independent."""

from __future__ import annotations

import sys
from pathlib import Path


class UVDocAdapter:
    def __init__(self, uvdoc_root, checkpoint=None, device=None):
        import torch

        self.root = Path(uvdoc_root).resolve()
        if not (self.root / "utils.py").is_file():
            raise FileNotFoundError(f"Not a UVDoc checkout: {self.root}")
        sys.path.insert(0, str(self.root))
        from utils import IMG_SIZE, bilinear_unwarping, load_model

        self.torch = torch
        self.img_size = IMG_SIZE
        self.unwarp_fn = bilinear_unwarping
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = Path(checkpoint) if checkpoint else self.root / "model/best_model.pkl"
        self.model = load_model(checkpoint, map_location=self.device).to(self.device).eval()

    def predict(self, rgb_float):
        import cv2
        import numpy as np

        resized = cv2.resize(rgb_float, tuple(self.img_size))
        tensor = self.torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1)))[None]
        with self.torch.inference_mode():
            return self.model(tensor.to(self.device))

    def unwarp(self, rgb_float, grid2d, output_size):
        import numpy as np

        tensor = self.torch.from_numpy(np.ascontiguousarray(rgb_float.transpose(2, 0, 1)))[None]
        with self.torch.inference_mode():
            output = self.unwarp_fn(tensor.to(self.device), grid2d[:1], output_size)[0]
        return np.clip(output.cpu().numpy().transpose(1, 2, 0), 0.0, 1.0)
