"""Image loading and saving helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def list_image_files(input_dir: str | Path) -> list[Path]:
    input_dir = Path(input_dir)
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def img_to_tensor(path: str | Path) -> torch.Tensor:
    image = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def save_tensor_image(tensor: torch.Tensor, path: str | Path, image_format: str = 'png') -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {'quality': 95} if image_format.lower() == 'jpg' else {}
    tensor_to_image(tensor).save(path, **save_kwargs)
