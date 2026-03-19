"""图片/PDF 输入加载与预处理工具。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import fitz
import numpy as np


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> List[np.ndarray]:
    doc = fitz.open(pdf_path)
    images: List[np.ndarray] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    doc.close()
    return images


def expand_to_pages(path: Path, dpi: int = 200) -> List[Tuple[str, np.ndarray]]:
    """将单个样本展开为若干页面图像：(页面名, 图像数组)。"""
    if path.suffix.lower() == ".pdf":
        pages = pdf_to_images(path, dpi=dpi)
        return [(f"{path.stem}_p{i}", img) for i, img in enumerate(pages)]
    return [(path.stem, load_image(path))]
