"""测试样本发现与文件类型判断工具。"""

from __future__ import annotations

from pathlib import Path
from typing import List


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}


def collect_samples(input_path: Path) -> List[Path]:
    """从文件或目录收集受支持的测试样本。"""
    if input_path.is_file():
        ext = input_path.suffix.lower()
        if ext in IMAGE_EXTS or ext in PDF_EXTS:
            return [input_path]
        return []

    if not input_path.is_dir():
        return []

    items = [
        p
        for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS.union(PDF_EXTS)
    ]
    return sorted(items, key=lambda p: p.name.lower())


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_pdf_file(path: Path) -> bool:
    return path.suffix.lower() in PDF_EXTS
