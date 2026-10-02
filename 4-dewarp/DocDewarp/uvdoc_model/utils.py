import os

import torch
import torch.nn.functional as F

from model import UVDocnet
import cv2
import numpy as np
IMG_SIZE = (488, 712)
GRID_SIZE = (45, 31)


def load_model(ckpt_path):
    """
    Load UVDocnet model.
    """
    model = UVDocnet(num_filter=32, kernel_size=5)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state"])
    return model


def get_version():
    """
    Returns the version of the various packages used for evaluation.
    """
    import pytesseract

    return {
        "tesseract": str(pytesseract.get_tesseract_version()),
        "pyesseract": os.popen("pip list | grep pytesseract").read().split()[-1],
        "Levenshtein": os.popen("pip list | grep Levenshtein").read().split()[-1],
        "jiwer": os.popen("pip list | grep jiwer").read().split()[-1],
        "matlabengineforpython": os.popen("pip list | grep matlab").read().split()[-1],
    }


def bilinear_unwarping(warped_img, point_positions, img_size):
    """
    Utility function that unwarps an image.
    Unwarp warped_img based on the 2D grid point_positions with a size img_size.
    Args:
        warped_img  :       torch.Tensor of shape BxCxHxW (dtype float)
        point_positions:    torch.Tensor of shape Bx2xGhxGw (dtype float)
        img_size:           tuple of int [w, h]
    """
    upsampled_grid = F.interpolate(
        point_positions, size=(img_size[1], img_size[0]), mode="bilinear", align_corners=True
    )
    unwarped_img = F.grid_sample(warped_img, upsampled_grid.transpose(1, 2).transpose(2, 3), align_corners=True)

    return unwarped_img


def bilinear_unwarping_from_numpy(warped_img, point_positions, img_size):
    """
    Utility function that unwarps an image.
    Unwarp warped_img based on the 2D grid point_positions with a size img_size.
    Accept numpy arrays as input.
    """
    warped_img = torch.unsqueeze(torch.from_numpy(warped_img.transpose(2, 0, 1)).float(), dim=0)
    point_positions = torch.unsqueeze(torch.from_numpy(point_positions.transpose(2, 0, 1)).float(), dim=0)

    unwarped_img = bilinear_unwarping(warped_img, point_positions, img_size)

    unwarped_img = unwarped_img[0].numpy().transpose(1, 2, 0)
    return unwarped_img

def create_document_mask(edge_img, kernel_size=5, min_contour_area=1000):
    """
    从边缘图像生成实心文档掩码（文档区域为 255，背景为 0）。

    流程：二值化 → 形态学闭运算 → 轻微膨胀 → 查找轮廓 → 填充最大轮廓及其余大轮廓

    Args:
        edge_img: 原始边缘图像（numpy array，任意尺寸，值域 0~255）
        img_size: 统一 resize 的目标尺寸（正方形边长）
        kernel_size: 形态学操作的核大小，默认 5
        min_contour_area: 次要轮廓的最小面积阈值，用于过滤噪点，默认 1000

    Returns:
        fill_mask: uint8 类型，形状 (img_size, img_size)，文档区域为 255，背景为 0
        fill_mask_float: float32 类型，同上，值域 [0, 1]，可直接用于乘法掩码
    """
    # 1. 严格二值化（消除模糊过渡像素）
    _, edge_binary = cv2.threshold(edge_img, 0, 255, cv2.THRESH_BINARY)

    # 3. 形态学闭运算：桥接断裂的边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    edge_closed = cv2.morphologyEx(edge_binary, cv2.MORPH_CLOSE, kernel)

    # 4. 轻微膨胀，让边缘线更粗、更闭合
    edge_dilated = cv2.dilate(edge_closed, kernel, iterations=1)

    # 5. 查找轮廓并填充
    contours, _ = cv2.findContours(edge_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fill_mask = np.zeros_like(edge_binary)
    if len(contours) > 0:
        # 按面积排序，取最大的轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # 填充最大轮廓（通常是文档主体）
        cv2.drawContours(fill_mask, [contours[0]], -1, 255, thickness=cv2.FILLED)

        # 填充其余较大的闭合区域（如折叠处的内轮廓）
        for cnt in contours[1:]:
            if cv2.contourArea(cnt) > min_contour_area:
                cv2.drawContours(fill_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    fill_mask_float = fill_mask.astype(np.float32) / 255.0

    return fill_mask, fill_mask_float
