"""Shared utilities for training and inference."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

DEFAULT_TILE = 512
DEFAULT_OVERLAP = 128
DEFAULT_NCNN_THREADS = 4
MAX_SAFE_TILE = 1024

_BLEND_CACHE_TORCH = {}
_BLEND_CACHE_NP = {}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_checkpoint_state_dict(model_path: str):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        return checkpoint
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    for key in ('params_ema', 'params'):
        if key in checkpoint:
            return checkpoint[key]
    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise ValueError(f'Unrecognized checkpoint format: {list(checkpoint.keys())}')


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _make_blend_vector_torch(size: int, overlap: int) -> torch.Tensor:
    vector = torch.ones(size, dtype=torch.float32)
    if overlap <= 0:
        return vector
    coord = torch.arange(overlap, dtype=torch.float32)
    ramp = 0.5 - 0.5 * torch.cos(torch.pi * (coord + 1.0) / (overlap + 1.0))
    vector[:overlap] = ramp
    vector[-overlap:] = torch.flip(ramp, dims=[0])
    return vector


def get_blend_window_torch(size: int, overlap: int) -> torch.Tensor:
    key = (size, overlap)
    if key not in _BLEND_CACHE_TORCH:
        vector = _make_blend_vector_torch(size, overlap)
        window = vector.unsqueeze(1) * vector.unsqueeze(0)
        _BLEND_CACHE_TORCH[key] = window.unsqueeze(0).unsqueeze(0)
    return _BLEND_CACHE_TORCH[key]


def _make_blend_vector_np(size: int, overlap: int) -> np.ndarray:
    vector = np.ones(size, dtype=np.float32)
    if overlap <= 0:
        return vector
    coord = np.arange(overlap, dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(np.pi * (coord + 1.0) / (overlap + 1.0))
    vector[:overlap] = ramp
    vector[-overlap:] = ramp[::-1]
    return vector


def get_blend_window_np(size: int, overlap: int) -> np.ndarray:
    key = (size, overlap)
    if key not in _BLEND_CACHE_NP:
        vector = _make_blend_vector_np(size, overlap)
        _BLEND_CACHE_NP[key] = np.outer(vector, vector).astype(np.float32)
    return _BLEND_CACHE_NP[key]


def pad_tile_tensor(tile: torch.Tensor, tile_size: int) -> tuple[torch.Tensor, int, int]:
    _, _, height, width = tile.shape
    if height == tile_size and width == tile_size:
        return tile, height, width
    pad_right = tile_size - width
    pad_bottom = tile_size - height
    pad_mode = 'reflect' if (width > 1 and height > 1 and pad_right < width and pad_bottom < height) else 'replicate'
    padded = torch.nn.functional.pad(tile, (0, pad_right, 0, pad_bottom), mode=pad_mode)
    return padded, height, width


def pad_tile_array(tile: np.ndarray, tile_size: int) -> tuple[np.ndarray, int, int]:
    _, height, width = tile.shape
    if height == tile_size and width == tile_size:
        return np.ascontiguousarray(tile, dtype=np.float32), height, width
    pad_right = tile_size - width
    pad_bottom = tile_size - height
    pad_mode = 'reflect' if (width > 1 and height > 1 and pad_right < width and pad_bottom < height) else 'edge'
    padded = np.pad(tile, ((0, 0), (0, pad_bottom), (0, pad_right)), mode=pad_mode)
    return np.ascontiguousarray(padded, dtype=np.float32), height, width


class NcnnModel:
    """Thin wrapper around the Python NCNN binding."""

    def __init__(self, param_path: str, bin_path: str, num_threads: int = DEFAULT_NCNN_THREADS, use_vulkan: bool = False):
        import ncnn

        self._ncnn = ncnn
        self.padder_size = 8
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = use_vulkan
        if not use_vulkan and num_threads > 0:
            self.net.opt.num_threads = num_threads
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

    def infer_chw(self, image: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(image, dtype=np.float32)
        with self.net.create_extractor() as extractor:
            extractor.input('in0', self._ncnn.Mat(image).clone())
            _, output = extractor.extract('out0')
        return np.array(output, dtype=np.float32)
