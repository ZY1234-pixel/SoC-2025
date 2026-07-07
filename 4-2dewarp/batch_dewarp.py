"""
批量文档图像矫正脚本。
从三个文件夹中读取文件（原始图像、边界特征图、文本行特征图），
批量进行文档几何矫正。

用法：
  python batch_dewarp.py --input ./images --boundary ./masks/boundary --textline ./masks/textline --output ./results

输入文件夹：
  --input    : 原始图像文件夹
  --boundary : 边界特征图文件夹
  --textline : 文本行特征图文件夹
  --output   : 矫正结果输出文件夹
  --debug    : 是否保存形变场可视化（默认 False）

文件匹配规则：只比较文件名前缀（不含扩展名），不比较后缀。
只有在三个输入文件夹中前缀同时存在的文件才会被处理。
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optim.opt import opt
from utils.line import line as extract_line
from utils.line1 import line1 as extract_line1

# 导入原脚本中的核心函数
from dewarp_from_mask import (
    extract_boundary_edges_full,
    unwarp,
    visualize_grid_on_image,
    visualize_grid_mapping,
)


# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def list_image_filenames(folder):
    """
    列出文件夹中所有图片文件名（不含路径），返回 dict[str, str]。
    键为文件名前缀（不含扩展名），值为完整文件名。
    例如 {'test5': 'test5.jpg', 'page1': 'page1.png'}
    """
    filenames = {}
    if not os.path.isdir(folder):
        print(f"警告：文件夹不存在: {folder}")
        return filenames
    for f in os.listdir(folder):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
            prefix = os.path.splitext(f)[0]
            filenames[prefix] = f
    return filenames


def dewarp_single(input_path, boundary_path, textline_path, output_path, debug=False):
    """
    对单张图像执行文档矫正（与原 dewarp_from_mask.py 的 dewarp 函数逻辑一致）。
    """
    img = cv2.imread(input_path)
    if img is None:
        print(f"  [错误] 无法读取原始图像: {input_path}")
        return False

    b_mask = cv2.imread(boundary_path)
    if b_mask is None:
        print(f"  [错误] 无法读取边界特征图: {boundary_path}")
        return False

    t_mask = cv2.imread(textline_path)
    if t_mask is None:
        print(f"  [错误] 无法读取文本行特征图: {textline_path}")
        return False

    # 提取边界
    print("  提取完整边缘点（保留弯曲特征）...")
    boundary_edges = extract_boundary_edges_full(b_mask, n=128)

    # 提取文本行
    print("  提取文本行 ...")
    if len(t_mask.shape) == 3:
        t_gray = cv2.cvtColor(t_mask, cv2.COLOR_BGR2GRAY)
    else:
        t_gray = t_mask.copy()
    _, t_bin = cv2.threshold(t_gray, 10, 255, cv2.THRESH_BINARY)
    t_bin = cv2.resize(t_bin, (512, 512))

    img_name = os.path.basename(input_path)
    textline_np = extract_line(t_bin)
    print(f"    水平文本行数: {len(textline_np)}")

    os.makedirs("result/vertical_line", exist_ok=True)
    line1_np = extract_line1(textline_np, img_name)
    print(f"    垂直文本行数: {len(line1_np)}")

    # 求解形变场
    grid = opt(None, textline_np, line1_np, boundary_edges=boundary_edges)
    print("  opt 完成")

    # debug 可视化（仅保留两项）
    if debug:
        # 1. 形变场叠加在原图上显示
        grid_on_img_path = os.path.splitext(output_path)[0] + "_grid_on_image.png"
        visualize_grid_on_image(grid, img, step=1, save_path=grid_on_img_path)

        # 2. 可视化网格映射对比
        fig, axes = visualize_grid_mapping(grid, sample_step=4)
        grid_mapping_path = os.path.splitext(output_path)[0] + "_grid_mapping.png"
        fig.savefig(grid_mapping_path, dpi=150)
        print(f"  网格映射可视化已保存: {grid_mapping_path}")
        plt.close(fig)

    # 矫正图像
    print("  矫正图像 ...")
    result = unwarp(img, grid)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"  矫正结果已保存: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="批量文档图像几何矫正")
    parser.add_argument("--input",    "-i", default="./images1", help="原始图像文件夹路径")
    parser.add_argument("--boundary", "-b", default="./mask_b", help="边界特征图文件夹路径")
    parser.add_argument("--textline", "-t", default="./mask_t", help="文本行特征图文件夹路径")
    parser.add_argument("--output",   "-o", default="./results_dewarp", help="矫正结果输出文件夹路径")
    parser.add_argument("--debug",    "-d", action="store_true", help="保存形变场可视化图")
    args = parser.parse_args()

    # 获取三个文件夹中的文件名（前缀 -> 完整文件名）
    input_names = list_image_filenames(args.input)
    boundary_names = list_image_filenames(args.boundary)
    textline_names = list_image_filenames(args.textline)

    print(f"原始图像文件夹 ({args.input}): {len(input_names)} 个文件")
    print(f"边界特征图文件夹 ({args.boundary}): {len(boundary_names)} 个文件")
    print(f"文本行特征图文件夹 ({args.textline}): {len(textline_names)} 个文件")

    # 求交集：只处理三个文件夹中前缀同时存在的文件
    common_prefixes = sorted(set(input_names.keys()) & set(boundary_names.keys()) & set(textline_names.keys()))

    if not common_prefixes:
        print("没有在三个文件夹中同时存在的同名前缀文件，退出。")
        return

    all_prefixes = set(input_names.keys()) | set(boundary_names.keys()) | set(textline_names.keys())

    print(f"\n共 {len(common_prefixes)} 个文件待处理（按前缀匹配）：{common_prefixes}\n")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 逐个处理
    success_count = 0
    fail_count = 0
    for idx, prefix in enumerate(common_prefixes, 1):
        input_filename = input_names[prefix]
        boundary_filename = boundary_names[prefix]
        textline_filename = textline_names[prefix]

        input_path = os.path.join(args.input, input_filename)
        boundary_path = os.path.join(args.boundary, boundary_filename)
        textline_path = os.path.join(args.textline, textline_filename)
        # 输出文件名使用原始图像的文件名
        output_path = os.path.join(args.output, input_filename)

        print(f"[{idx}/{len(common_prefixes)}] 处理: 前缀={prefix}  "
              f"(input={input_filename}, boundary={boundary_filename}, textline={textline_filename})")

        try:
            ok = dewarp_single(
                input_path=input_path,
                boundary_path=boundary_path,
                textline_path=textline_path,
                output_path=output_path,
                debug=args.debug,
            )
            if ok:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  [异常] 处理前缀 {prefix} 时出错: {e}")
            fail_count += 1

    print(f"\n批量处理完成：成功 {success_count}，失败 {fail_count}，共 {len(common_prefixes)} 个。")


if __name__ == "__main__":
    main()