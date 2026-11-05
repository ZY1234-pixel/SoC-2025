"""Recursive image discovery and output files."""

from typing import Iterable
from pathlib import Path
import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def input_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
    else:
        raise FileNotFoundError(path)


def output_key(image_path: Path, input_path: Path) -> Path:
    """Return a collision-free relative path without an image suffix."""
    if input_path.is_dir():
        relative = image_path.relative_to(input_path)
    else:
        relative = Path(image_path.name)
    return relative.with_suffix("")


def output_file(directory: Path, key: Path, suffix: str) -> Path:
    """Build an output path and mirror source subdirectories when needed."""
    path = directory / key.parent / f"{key.name}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")
