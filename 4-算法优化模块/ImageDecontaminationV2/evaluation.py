import argparse
import os
import glob
from typing import Tuple, Optional
import gc

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataLoader import DirtyDocumentsDataset_Test
from model2 import DecontaminationModel


def parse_args():
    parser = argparse.ArgumentParser(description="图像去污模型推理")
    parser.add_argument("--input_dir", type=str, default="./data/test2/input/", help="输入图像目录")
    parser.add_argument("--gt_dir", type=str, default="./data/test2/target/", help="Ground truth图像目录（用于计算PSNR）")
    parser.add_argument("--output_dir", type=str, default="./data/test/GRAY/", help="输出图像目录")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/best_decontamination_model_balanced.pth",
        help="模型权重路径(.pth或checkpoint)",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="批大小")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader工作线程数")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--tile_size", type=int, default=512, help="高分辨率图像分块大小（True默认512）")
    parser.add_argument("--overlap", type=int, default=32, help="分块重叠像素数（默认64）")
    parser.add_argument("--use_tiling", action="store_true", help="是否对高分辨率图像使用分块处理")
    parser.add_argument("--output_grayscale", type=bool, default=True, help="是否启用灰度图像输出模式（只保留Y通道）")
    return parser.parse_args()


def load_model(model_path: str, device: torch.device) -> DecontaminationModel:
    model = DecontaminationModel(in_channels=3, out_channels=3).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"已加载模型: {model_path}")
    return model


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算两张图像之间的PSNR（峰值信噪比）
    
    参数:
        img1: 第一张图像，numpy数组，uint8格式，BGR或灰度
        img2: 第二张图像，numpy数组，uint8格式，BGR或灰度，必须与img1尺寸相同
    
    返回:
        PSNR值（dB）
    """
    if img1.shape != img2.shape:
        raise ValueError(f"图像尺寸不匹配: {img1.shape} vs {img2.shape}")
    
    # 转换为float64以避免溢出
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 计算MSE（均方误差）
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        # 图像完全相同
        return float('inf')
    
    # PSNR计算，MAX=255（8位图像）
    max_pixel = 255.0
    psnr = 10 * np.log10(max_pixel ** 2 / mse)
    
    return psnr


def print_psnr_statistics(psnr_values: list):
    """
    打印PSNR统计信息，包括区间分布
    
    参数:
        psnr_values: PSNR值列表
    """
    if not psnr_values:
        return
    
    psnr_array = np.array(psnr_values)
    
    # 基本统计
    print(f"\n=== PSNR 统计 ===")
    print(f"平均 PSNR: {np.mean(psnr_array):.2f} dB")
    print(f"最小 PSNR: {np.min(psnr_array):.2f} dB")
    print(f"最大 PSNR: {np.max(psnr_array):.2f} dB")
    print(f"标准差: {np.std(psnr_array):.2f} dB")
    print(f"共计算了 {len(psnr_values)} 张图像的PSNR")
    
    # 区间统计
    count_0_10 = np.sum((psnr_array >= 0) & (psnr_array < 10))
    count_10_20 = np.sum((psnr_array >= 10) & (psnr_array < 20))
    count_20_30 = np.sum((psnr_array >= 20) & (psnr_array < 30))
    count_30_plus = np.sum(psnr_array >= 30)
    
    print(f"\n=== PSNR 区间分布 ===")
    print(f"0-10 dB:    {count_0_10:4d} 张 ({count_0_10*100/len(psnr_values):5.1f}%)")
    print(f"10-20 dB:   {count_10_20:4d} 张 ({count_10_20*100/len(psnr_values):5.1f}%)")
    print(f"20-30 dB:   {count_20_30:4d} 张 ({count_20_30*100/len(psnr_values):5.1f}%)")
    print(f"30+ dB:     {count_30_plus:4d} 张 ({count_30_plus*100/len(psnr_values):5.1f}%)")


def tensor_yuv_to_bgr(img_yuv: torch.Tensor, original_size: Tuple[int, int], is_grayscale: bool = False) -> np.ndarray:
    """
    将模型输出的YUV张量裁剪回原始尺寸并转换为BGR uint8或灰度图像
    img_yuv: [3, H, W]，范围[0,1]
    original_size: (h, w)
    is_grayscale: 如果为True，只返回Y通道（灰度图像）
    """
    h, w = original_size
    img_np = img_yuv.clamp(0, 1).cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
    img_np = img_np[:h, :w]  # 去除填充
    img_np = (img_np * 255.0).astype(np.uint8)
    
    if is_grayscale:
        # 只返回Y通道（灰度图像）
        return img_np[:, :, 0]
    else:
        # 转换为BGR
        return cv2.cvtColor(img_np, cv2.COLOR_YUV2BGR)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"输入文件不存在: {image_path}")


def process_high_resolution_image(
    model: DecontaminationModel,
    image_path: str,
    output_dir: str,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 64,
    gt_path: Optional[str] = None,
    output_grayscale: bool = False,
):
    """
    处理高分辨率图像，将其分割为多个块（默认512x512），分别处理后再拼接
    
    参数:
        model: 去污模型
        image_path: 输入图像路径
        output_dir: 输出目录
        device: 设备
        tile_size: 分块大小（默认512，与训练时一致）
        overlap: 块之间的重叠像素数（默认64）
        gt_path: Ground truth图像路径（可选，用于计算PSNR）
        output_grayscale: 是否输出灰度图像（只保留Y通道）
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"输入文件不存在: {image_path}")
    
    if not os.path.isfile(image_path):
        raise ValueError(f"输入路径不是文件: {image_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始图像
    print(f"读取图像: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        # 提供更详细的错误信息
        file_ext = os.path.splitext(image_path)[1].lower()
        supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        if file_ext not in supported_formats:
            raise ValueError(f"不支持的图像格式: {file_ext}。支持的格式: {supported_formats}")
        else:
            raise ValueError(f"无法加载图像: {image_path}。可能是文件损坏或格式不正确。")
    
    # 转换为YUV格式（与训练时一致）
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    else:
        # 灰度图像转换为YUV，使用OpenCV标准转换
        if len(img.shape) == 2:
            img_gray = img
        else:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_yuv = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2YUV)
    
    original_h, original_w = img_yuv.shape[:2]
    print(f"图像尺寸: {original_h}x{original_w}, 开始分块处理...")
    
    # 确保分块大小是8的倍数（UNet结构需要）
    tile_size = max(64, (tile_size // 8) * 8)
    overlap = min(overlap, tile_size // 4)  # 限制重叠区域大小
    
    # 创建输出图像和权重图（3通道YUV格式）
    result_img_y = np.zeros((original_h, original_w), dtype=np.float32)
    result_img_u = np.zeros((original_h, original_w), dtype=np.float32)
    result_img_v = np.zeros((original_h, original_w), dtype=np.float32)
    weight_map = np.zeros((original_h, original_w), dtype=np.float32)
    
    # 计算分块数量
    num_tiles_h = max(1, (original_h - 1) // (tile_size - overlap) + 1)
    num_tiles_w = max(1, (original_w - 1) // (tile_size - overlap) + 1)
    
    print(f"将图像分为 {num_tiles_h}x{num_tiles_w} 块处理，块大小={tile_size}，重叠={overlap}")
    
    # 处理每个分块
    total_tiles = num_tiles_h * num_tiles_w
    processed_tiles = 0
    
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            processed_tiles += 1
            # print(f"处理块 [{i+1},{j+1}]/{total_tiles} ({processed_tiles*100/total_tiles:.1f}%)")
            
            # 计算当前块的坐标
            y_start = max(0, i * (tile_size - overlap))
            y_end = min(original_h, y_start + tile_size)
            x_start = max(0, j * (tile_size - overlap))
            x_end = min(original_w, x_start + tile_size)
            
            # 提取当前块
            tile = img_yuv[y_start:y_end, x_start:x_end]
            tile_h, tile_w = tile.shape[:2]
            
            # 计算需要的填充，确保高度和宽度都能被8整除
            pad_h = 8 - (tile_h % 8) if tile_h % 8 != 0 else 0
            pad_w = 8 - (tile_w % 8) if tile_w % 8 != 0 else 0
            
            # 进行填充（YUV格式，填充值设为255对应Y通道，UV通道为128）
            tile_padded = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 128, 128))
            
            # 归一化并转换为张量 [H, W, 3] -> [3, H, W] -> [1, 3, H, W]
            tile_padded = tile_padded.astype(np.float32) / 255.0
            tile_tensor = torch.from_numpy(tile_padded.transpose((2, 0, 1))).float().unsqueeze(0)
            
            # 清理内存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # 执行推理
            try:
                with torch.no_grad():
                    tile_output = model(tile_tensor.to(device))
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    # 内存不足，清理并尝试减小块大小重新调用
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                    print(f"GPU内存不足，尝试减小分块大小...")
                    
                    # 减小一半分块大小，递归调用
                    new_tile_size = max(256, tile_size // 2)
                    return process_high_resolution_image(
                        model, image_path, output_dir, device,
                        tile_size=new_tile_size,
                        overlap=min(32, overlap),
                        output_grayscale=output_grayscale,
                    )
                else:
                    raise e
            
            # 处理输出 [B, 3, H, W] -> [3, H, W] -> [H, W, 3]
            tile_clean_yuv = tile_output.cpu().squeeze(0).permute(1, 2, 0).numpy()
            tile_clean_yuv = (tile_clean_yuv * 255).astype(np.uint8)
            
            # 去除填充
            if pad_h > 0 or pad_w > 0:
                tile_clean_yuv = tile_clean_yuv[:tile_h, :tile_w]
            
            # 提取YUV三个通道用于后续处理
            tile_clean_y = tile_clean_yuv[:, :, 0].astype(np.float32) / 255.0
            tile_clean_u = tile_clean_yuv[:, :, 1].astype(np.float32) / 255.0
            tile_clean_v = tile_clean_yuv[:, :, 2].astype(np.float32) / 255.0
            
            # 创建权重图 - 使用高斯加权以平滑拼接边缘
            weight = np.ones_like(tile_clean_y)
            
            # 创建边缘淡入淡出效果
            if overlap > 0:
                if i > 0:  # 上边缘
                    overlap_top = min(overlap, tile_h)
                    if overlap_top > 0:
                        for y in range(overlap_top):
                            weight[y, :] *= y / overlap_top
                if j > 0:  # 左边缘
                    overlap_left = min(overlap, tile_w)
                    if overlap_left > 0:
                        for x in range(overlap_left):
                            weight[:, x] *= x / overlap_left
                if i < num_tiles_h - 1 and y_end < original_h:  # 下边缘
                    overlap_bottom = min(overlap, tile_h)
                    if overlap_bottom > 0:
                        for y in range(overlap_bottom):
                            idx = tile_h - overlap_bottom + y
                            if 0 <= idx < tile_h:
                                weight[idx, :] *= (overlap_bottom - y) / overlap_bottom
                if j < num_tiles_w - 1 and x_end < original_w:  # 右边缘
                    overlap_right = min(overlap, tile_w)
                    if overlap_right > 0:
                        for x in range(overlap_right):
                            idx = tile_w - overlap_right + x
                            if 0 <= idx < tile_w:
                                weight[:, idx] *= (overlap_right - x) / overlap_right
            
            # 加权合并到结果图像（分别合并Y、U、V三个通道）
            result_img_y[y_start:y_end, x_start:x_end] += tile_clean_y * weight
            result_img_u[y_start:y_end, x_start:x_end] += tile_clean_u * weight
            result_img_v[y_start:y_end, x_start:x_end] += tile_clean_v * weight
            weight_map[y_start:y_end, x_start:x_end] += weight
            
            # 释放此块的内存
            del tile_tensor, tile_output, tile_clean_y, tile_clean_u, tile_clean_v
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # 归一化结果
    print("处理完所有分块，合并结果...")
    weight_map = np.maximum(weight_map, 1e-6)  # 避免除零
    clean_img_y = result_img_y / weight_map
    clean_img_u = result_img_u / weight_map
    clean_img_v = result_img_v / weight_map
    
    # 转换为8位图像
    clean_img_y = np.clip(clean_img_y * 255.0, 0, 255).astype(np.uint8)
    
    # 保存结果
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"clean_{filename}")
    
    if output_grayscale:
        # 灰度输出模式：只保存Y通道
        cv2.imwrite(output_path, clean_img_y)
        print(f"高分辨率图像处理完成（灰度输出）: {output_path}")
    else:
        # 彩色输出模式：转换为BGR保存
        clean_img_u = np.clip(clean_img_u * 255.0, 0, 255).astype(np.uint8)
        clean_img_v = np.clip(clean_img_v * 255.0, 0, 255).astype(np.uint8)
        clean_img_yuv = np.stack([clean_img_y, clean_img_u, clean_img_v], axis=-1)
        clean_img_bgr = cv2.cvtColor(clean_img_yuv, cv2.COLOR_YUV2BGR)
        cv2.imwrite(output_path, clean_img_bgr)
        print(f"高分辨率图像处理完成（彩色输出）: {output_path}")
    
    # 如果提供了ground truth，计算PSNR
    if gt_path:
        if not os.path.exists(gt_path):
            print(f"警告: Ground truth文件不存在: {gt_path}")
        else:
            try:
                gt_img = cv2.imread(gt_path)
                if gt_img is not None:
                    if output_grayscale:
                        # 灰度输出模式：只计算Y通道的PSNR
                        pred_img = clean_img_y
                        # 如果GT是彩色图像，转换为灰度（只保留Y通道）
                        if len(gt_img.shape) == 3:
                            gt_img_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
                        else:
                            gt_img_gray = gt_img
                        # 确保尺寸一致
                        if pred_img.shape != gt_img_gray.shape:
                            print(f"调整Ground truth尺寸: {gt_img_gray.shape} -> {pred_img.shape}")
                            gt_img_gray = cv2.resize(gt_img_gray, (pred_img.shape[1], pred_img.shape[0]))
                    else:
                        # 彩色输出模式：计算BGR图像的PSNR
                        pred_img = clean_img_bgr
                        # 确保尺寸一致
                        if pred_img.shape != gt_img.shape:
                            print(f"调整Ground truth尺寸: {gt_img.shape} -> {pred_img.shape}")
                            gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]))
                        psnr = calculate_psnr(pred_img, gt_img)
                        print(f"PSNR（彩色BGR）: {psnr:.2f} dB")
                else:
                    print(f"警告: 无法读取Ground truth图像: {gt_path}")
            except Exception as e:
                print(f"计算PSNR时出错: {str(e)}")
    elif gt_path is None:
        print("提示: 未提供Ground truth路径，跳过PSNR计算")
    
    return output_path


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"使用设备: {device}")

    model = load_model(args.model_path, device)

    # 检查输入路径是文件还是目录
    input_path = args.input_dir
    psnr_values = []
    
    if os.path.isfile(input_path):
        # 单个文件
        print(f"处理单个图像: {input_path}")
        
        # 查找对应的ground truth图像
        gt_path = None
        if args.gt_dir:
            if not os.path.exists(args.gt_dir):
                print(f"警告: Ground truth目录不存在: {args.gt_dir}")
            else:
                filename = os.path.basename(input_path)
                base_name = os.path.splitext(filename)[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    potential_gt = os.path.join(args.gt_dir, f"{base_name}{ext}")
                    if os.path.exists(potential_gt):
                        gt_path = potential_gt
                        print(f"找到Ground truth: {gt_path}")
                        break
                if not gt_path:
                    print(f"警告: 未找到对应的Ground truth文件 (输入: {filename})")
        
        output_path = process_high_resolution_image(
            model=model,
            image_path=input_path,
            output_dir=args.output_dir,
            device=device,
            tile_size=args.tile_size,
            overlap=args.overlap,
            gt_path=gt_path,
            output_grayscale=args.output_grayscale,
        )
        
        # 如果提供了ground truth，计算PSNR
        if gt_path and output_path:
            try:
                pred_img = cv2.imread(output_path)
                gt_img = cv2.imread(gt_path)
                if pred_img is not None and gt_img is not None:
                    # 如果输出是灰度图像，GT也需要转换为灰度（只保留Y通道）
                    if args.output_grayscale and len(pred_img.shape) == 2:
                        if len(gt_img.shape) == 3:
                            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
                    if pred_img.shape != gt_img.shape:
                        print(f"调整Ground truth尺寸: {gt_img.shape} -> {pred_img.shape}")
                        gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]))
                    psnr = calculate_psnr(pred_img, gt_img)
                    psnr_values.append(psnr)
                    mode_str = "灰度Y通道" if args.output_grayscale else "彩色BGR"
                    print(f"PSNR（{mode_str}）: {psnr:.2f} dB")
                else:
                    print(f"警告: 无法读取图像 (预测: {output_path}, GT: {gt_path})")
            except Exception as e:
                print(f"计算PSNR时出错: {str(e)}")
        elif not gt_path:
            print("提示: 未找到Ground truth文件，跳过PSNR计算")
                
    elif os.path.isdir(input_path):
        # 目录，处理目录中的所有图像
        image_paths = glob.glob(os.path.join(input_path, '*.png')) + \
                      glob.glob(os.path.join(input_path, '*.jpg')) + \
                      glob.glob(os.path.join(input_path, '*.jpeg'))
        
        if not image_paths:
            print(f"目录中未找到图像文件: {input_path}")
            return
        
        print(f"找到 {len(image_paths)} 张图像，开始处理...")
        for i, image_path in enumerate(image_paths):
            print(f"\n处理图像 [{i+1}/{len(image_paths)}]: {os.path.basename(image_path)}")
            
            # 查找对应的ground truth图像
            gt_path = None
            if args.gt_dir:
                if not os.path.exists(args.gt_dir):
                    if i == 0:  # 只打印一次警告
                        print(f"警告: Ground truth目录不存在: {args.gt_dir}")
                else:
                    filename = os.path.basename(image_path)
                    base_name = os.path.splitext(filename)[0]
                    for ext in ['.png', '.jpg', '.jpeg']:
                        potential_gt = os.path.join(args.gt_dir, f"{base_name}{ext}")
                        if os.path.exists(potential_gt):
                            gt_path = potential_gt
                            break
                    if not gt_path and i == 0:  # 只对第一张图像打印警告
                        print(f"警告: 未找到对应的Ground truth文件 (输入: {filename})")
            
            try:
                output_path = process_high_resolution_image(
                    model=model,
                    image_path=image_path,
                    output_dir=args.output_dir,
                    device=device,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    gt_path=gt_path,
                    output_grayscale=args.output_grayscale,
                )
                
                # 如果提供了ground truth，计算PSNR
                if gt_path and output_path:
                    try:
                        pred_img = cv2.imread(output_path)
                        gt_img = cv2.imread(gt_path)
                        if pred_img is not None and gt_img is not None:
                            # 如果输出是灰度图像，GT也需要转换为灰度（只保留Y通道）
                            if args.output_grayscale and len(pred_img.shape) == 2:
                                if len(gt_img.shape) == 3:
                                    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
                            if pred_img.shape != gt_img.shape:
                                print(f"  调整Ground truth尺寸: {gt_img.shape} -> {pred_img.shape}")
                                gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]))
                            psnr = calculate_psnr(pred_img, gt_img)
                            psnr_values.append(psnr)
                            mode_str = "灰度Y通道" if args.output_grayscale else "彩色BGR"
                            print(f"  PSNR（{mode_str}）: {psnr:.2f} dB")
                        else:
                            print(f"  警告: 无法读取图像 (预测: {output_path}, GT: {gt_path})")
                    except Exception as e:
                        print(f"  计算PSNR时出错: {str(e)}")
                elif not gt_path:
                    if i == 0:  # 只对第一张图像打印提示
                        print(f"  提示: 未找到Ground truth文件，跳过PSNR计算")
                        
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {str(e)}")
            
            # 清理内存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    else:
        raise ValueError(f"输入路径既不是文件也不是目录: {input_path}")
    
    # 打印PSNR统计信息
    if psnr_values:
        print(f"\n{'='*50}")
        print_psnr_statistics(psnr_values)
        print(f"{'='*50}")
    elif args.gt_dir:
        print(f"\n警告: 提供了 --gt_dir ({args.gt_dir})，但未计算任何PSNR值。")
        print(f"请检查:")
        print(f"  1. Ground truth目录是否存在: {os.path.exists(args.gt_dir) if args.gt_dir else 'N/A'}")
        print(f"  2. Ground truth文件名是否与输入文件名匹配（去除扩展名后）")
        print(f"  3. Ground truth文件格式是否为 .png, .jpg 或 .jpeg")


if __name__ == "__main__":
    main()