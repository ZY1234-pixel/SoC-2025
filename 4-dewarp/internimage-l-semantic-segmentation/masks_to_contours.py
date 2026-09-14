import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INPUT = Path("outputs/masks")
DEFAULT_OUTPUT = Path("outputs/contours")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_grayscale(path):
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像: {path}")
    return image


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"无法编码 PNG: {path}")
    encoded.tofile(str(path))


def mask_to_contour(mask, thickness=2, mode="inner"):
    """Return a black image with a white 0/255 contour.

    inner:
        Keep only pixels inside the source mask boundary. This is the safest
        default when the contour must remain a subset of the original mask.
    centered:
        Keep pixels on both sides of the boundary for a more visible contour.
    """
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    radius = max(int(thickness), 1)
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(
        binary, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT,
        borderValue=0)

    if mode == "inner":
        contour = cv2.subtract(binary, eroded)
    elif mode == "centered":
        dilated = cv2.dilate(
            binary, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT,
            borderValue=0)
        contour = cv2.subtract(dilated, eroded)
    else:
        raise ValueError(f"不支持的轮廓模式: {mode}")

    return np.where(contour > 0, 255, 0).astype(np.uint8)


def collect_images(input_dir):
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量把二值 mask 转换为只保留边缘轮廓的 0/255 PNG。")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="轮廓向内或向两侧扩展的像素半径，默认 2。")
    parser.add_argument(
        "--mode",
        choices=("inner", "centered"),
        default="inner",
        help="inner 只保留 mask 内侧轮廓；centered 保留边界两侧。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖输出目录中已经存在的同名轮廓图。")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="仅处理前 N 张，适合测试脚本。")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if args.thickness < 1:
        raise ValueError("--thickness 必须大于或等于 1")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("输出目录不能等于或位于输入目录内部")

    image_paths = collect_images(input_dir)
    if args.max_samples is not None:
        image_paths = image_paths[:max(args.max_samples, 0)]
    if not image_paths:
        raise RuntimeError(f"没有在输入目录中找到 mask: {input_dir}")

    converted = 0
    skipped = 0
    empty = 0
    failed = 0
    for index, source_path in enumerate(image_paths, 1):
        relative = source_path.relative_to(input_dir).with_suffix(".png")
        output_path = output_dir / relative
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            mask = read_grayscale(source_path)
            contour = mask_to_contour(
                mask, thickness=args.thickness, mode=args.mode)
            if not np.any(contour):
                empty += 1
            write_png(output_path, contour)
            converted += 1
        except Exception as error:
            failed += 1
            print(f"[失败] {source_path}: {error}")

        if index % 100 == 0 or index == len(image_paths):
            print(
                f"[{index}/{len(image_paths)}] 已转换={converted}, "
                f"跳过={skipped}, 空轮廓={empty}, 失败={failed}",
                flush=True)

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"轮廓模式: {args.mode}, 厚度: {args.thickness}")
    print(
        f"完成: 转换={converted}, 跳过={skipped}, "
        f"空轮廓={empty}, 失败={failed}")


if __name__ == "__main__":
    main()
