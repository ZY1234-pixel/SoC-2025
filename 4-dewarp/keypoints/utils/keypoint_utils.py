# utils/keypoint_utils.py
import torch
import numpy as np
import torch.nn.functional as F


def gaussian_2d(shape, sigma=1.0):
    """生成一个 2D 高斯核 (H, W)"""
    h, w = shape
    xs = torch.arange(0, w, dtype=torch.float32)
    ys = torch.arange(0, h, dtype=torch.float32)
    xs = xs.unsqueeze(0).repeat(h, 1)
    ys = ys.unsqueeze(1).repeat(1, w)
    return xs, ys


def generate_heatmap(heatmap_size, keypoints, sigma=3.0):
    """
    根据关键点坐标生成热力图 (4, H, W)
    keypoints: numpy array (4, 2) 在原图尺寸下的坐标
    heatmap_size: (H, W) 热力图尺寸 (通常与输入尺寸一致)
    sigma: 高斯标准差
    """
    H, W = heatmap_size
    num_keypoints = keypoints.shape[0]
    heatmaps = np.zeros((num_keypoints, H, W), dtype=np.float32)

    for i, (x, y) in enumerate(keypoints):
        # 检查坐标是否有效
        if x < 0 or y < 0 or x >= W or y >= H:
            continue
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        heatmaps[i] = gaussian

    # 可选：限制最大值 1
    heatmaps = np.clip(heatmaps, 0.0, 1.0)
    return heatmaps


def soft_argmax(heatmaps, T=1.0):
    """
    从热力图 (B, C, H, W) 解码浮点坐标，可微
    返回: (B, C, 2) 归一化到 [-1, 1] 的坐标
    """
    B, C, H, W = heatmaps.shape
    prob = F.softmax(heatmaps.view(B, C, -1) / T, dim=-1)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=heatmaps.device),
                            torch.linspace(-1, 1, W, device=heatmaps.device))
    grid = torch.stack([xx, yy], dim=-1).view(-1, 2)
    coords = prob @ grid
    return coords.view(B, C, 2)


def decode_keypoints(heatmaps, img_size, T=1.0, use_soft_argmax=False):
    """
    推理时从热力图解码实际坐标
    heatmaps: (C, H, W) numpy 或 torch
    img_size: (W, H) 输入尺寸（注意是 W,H 顺序）
    返回: (C, 2) numpy [x, y]
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()

    C, H, W = heatmaps.shape
    coords = np.zeros((C, 2), dtype=np.float32)

    for i in range(C):
        hm = heatmaps[i]
        if hm.max() <= 0:
            coords[i, 0] = 0.0
            coords[i, 1] = 0.0
            continue

        # 找到最大值位置
        y, x = np.unravel_index(hm.argmax(), hm.shape)

        # 亚像素精修：用高斯曲面拟合最大值附近区域
        if 1 <= x < W - 1 and 1 <= y < H - 1:
            # 在峰值附近 3×3 邻域用二阶泰勒展开精修
            try:
                dx = 0.5 * (hm[y, x - 1] - hm[y, x + 1]) / (hm[y, x - 1] + hm[y, x + 1] - 2 * hm[y, x] + 1e-8)
                dy = 0.5 * (hm[y - 1, x] - hm[y + 1, x]) / (hm[y - 1, x] + hm[y + 1, x] - 2 * hm[y, x] + 1e-8)
                x += dx
                y += dy
            except:
                pass

        coords[i, 0] = float(x)
        coords[i, 1] = float(y)

    return coords


def edge_aware_loss(keypoint_coords, seg_mask, edge_weight=1.0):
    """
    keypoint_coords: (B, 4, 2) 归一化 [-1, 1]
    seg_mask: (B, 1, H, W) 分割预测 >0.5 的二值 mask (可带梯度)
    edge_weight: 权重
    返回: 边缘贴合损失
    """
    # 这部分较复杂，需要可微边缘提取，先保留接口，后续可添加
    pass