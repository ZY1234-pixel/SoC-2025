import cv2
from cv2.gapi import mask
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from typing import Optional
import time

from scipy.linalg import svd


def sauvola_binarize(gray: np.ndarray, window_size: int = 31, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    """
    Sauvola 局部阈值二值化。
    threshold(x,y) = m(x,y) * (1 + k * (s(x,y)/r - 1))
    - gray: uint8 灰度图（2D）
    - window_size: 邻域窗口大小（建议奇数）
    - k: Sauvola 参数，常用 0.2~0.5
    - r: 动态范围参数，8-bit 图像常用 128
    """
    if gray.ndim != 2:
        raise ValueError("sauvola_binarize expects a 2D grayscale image.")

    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1

    gray_f = gray.astype(np.float32)

    mean = cv2.boxFilter(
        gray_f,
        ddepth=-1,
        ksize=(window_size, window_size),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )
    mean_sq = cv2.boxFilter(
        gray_f * gray_f,
        ddepth=-1,
        ksize=(window_size, window_size),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )

    var = mean_sq - mean * mean
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    thresh = mean * (1.0 + k * ((std / float(r)) - 1.0))
    binary = np.where(gray_f > thresh, 255, 0).astype(np.uint8)
    return binary


def test1(output_dir, image_files, use_fast=True):    
    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"警告: 无法读取图像 {img_path}，跳过。")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask2 = sauvola_binarize(gray, 31, 0.1, 128)
        # 保留原图，只在mask为白的区域变成白色
        result = image.copy()
        result[mask2 == 255] = [255, 255, 255]  # BGR白色

        # 保存结果
        out_path = output_dir / f"{img_path.stem}_result.png"
        cv2.imwrite(str(out_path), result)
        
        print(f"已处理并保存: {out_path}")
        cv2.imwrite(str(out_path), result)

def main(input_dir: Path, output_dir: Path):
    """
    主函数
    
    参数:
    - input_dir: 输入目录
    - output_dir: 输出目录
    """
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"错误: 输入路径不是有效的文件夹: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 支持的图像后缀
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    image_files = [p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in exts]

    if not image_files:
        print(f"在文件夹中未找到图像文件: {input_dir}")
        return

    print(f"将在文件夹 {input_dir} 中处理 {len(image_files)} 张图像。")

    # 记录开始时间
    start_time = time.time()

    # 执行处理
    test1(output_dir, image_files)
    
    # 记录结束时间并计算运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n处理完成！总耗时: {elapsed_time:.2f} 秒")
    print(f"平均每张图像: {elapsed_time/len(image_files):.2f} 秒")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='对文件夹中的所有图像进行颜色阈值分割测试（不遍历子目录）')
    parser.add_argument('--input_dir', '-i', type=str, default='D:/workspace/ImageDecontaminationV2/data/test7',help='输入图像文件夹路径（不遍历子目录）')
    parser.add_argument('--output_dir', '-o', type=str, default='./text_restore_results',
                        help='输出目录（默认: ./color_correction_results）')
    args = parser.parse_args()
    
    # 如果命令行没有指定输入目录，使用默认值
    # args.input_dir = "D:/workspace/ImageDecontaminationV2/text_restore_results/sauvola"
    main(Path(args.input_dir), Path(args.output_dir))