"""图像编码与解码工具。"""

import base64
import io
from typing import Optional, Tuple

from PIL import Image


def decode_image_base64(data: str) -> bytes:
    """将 base64 编码的图片字符串解码为原始 PNG 字节。"""
    return base64.b64decode(data)


def encode_image_base64(image_bytes: bytes) -> str:
    """将原始图片字节编码为 base64 字符串。"""
    return base64.b64encode(image_bytes).decode("ascii")


def load_image_bytes(path: str) -> bytes:
    """加载图片文件并返回其字节内容。"""
    with open(path, "rb") as f:
        return f.read()


def get_image_size(image_bytes: bytes) -> Tuple[int, int]:
    """根据图片字节返回 (宽度, 高度)。"""
    with Image.open(io.BytesIO(image_bytes)) as img:
        return img.size


def crop_image_bytes(
    page_image_bytes: bytes,
    bbox: Tuple[int, int, int, int],
    output_format: str = "PNG",
) -> Optional[bytes]:
    """从页面图像中裁剪指定区域并返回 PNG 字节。

    Args:
        page_image_bytes: 完整页面图像字节。
        bbox: (x1, y1, x2, y2) 裁剪区域（像素坐标）。
        output_format: 输出图像格式。

    Returns:
        裁剪后的图像字节，失败时返回 None。
    """
    try:
        with Image.open(io.BytesIO(page_image_bytes)) as img:
            x1, y1, x2, y2 = bbox
            cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
            buf = io.BytesIO()
            cropped.save(buf, format=output_format)
            return buf.getvalue()
    except Exception:
        return None


def constrain_image_size(
    original_width: float,
    original_height: float,
    max_width: float,
    max_height: float,
) -> Tuple[float, float]:
    """在保持宽高比的前提下缩放尺寸以适应最大边界。

    Returns:
        满足约束的 (宽度, 高度)。
    """
    if original_width <= 0 or original_height <= 0:
        return max_width, max_height

    scale = min(max_width / original_width, max_height / original_height, 1.0)
    return original_width * scale, original_height * scale
