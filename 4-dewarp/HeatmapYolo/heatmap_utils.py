import torch


def generate_heatmaps(keypoints_norm, img_shape=(64, 64), sigma=2.0, device="cuda"):
    """根据归一化坐标 (0~1) 生成高斯热力图。

    Args:
        keypoints_norm: (B, K, 3)，第三维为 [x_norm, y_norm, visible]
        img_shape: 输出特征图大小 (H, W)
        sigma: 高斯核半径
    Returns:
        heatmaps: (B, K, H, W)
    """
    B, K, _ = keypoints_norm.shape
    H, W = img_shape
    heatmaps = torch.zeros((B, K, H, W), dtype=torch.float32, device=device)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    grid_x = grid_x.float()
    grid_y = grid_y.float()

    for b in range(B):
        for k in range(K):
            x_norm = keypoints_norm[b, k, 0].item()
            y_norm = keypoints_norm[b, k, 1].item()
            visible = keypoints_norm[b, k, 2].item()
            if visible < 0.5:
                continue  # 不可见点不生成目标
            center_x = x_norm * (W - 1)
            center_y = y_norm * (H - 1)
            dist_sq = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
            heatmaps[b, k] = torch.exp(-dist_sq / (2 * sigma**2))
    return heatmaps


def decode_heatmap(heatmap):
    """从热力图解码归一化坐标，带抛物线亚像素精修。

    Args:
        heatmap: (B, K, H, W)
    Returns:
        coords_norm: (B, K, 2)，归一化坐标 (0~1)
    """
    B, K, H, W = heatmap.shape
    heatmap_flat = heatmap.view(B, K, -1)
    max_idx = heatmap_flat.argmax(dim=-1)

    y = (max_idx // W).float()
    x = (max_idx % W).float()

    # 亚像素精修：利用峰值附近的一阶/二阶差分拟合抛物线极值点
    for b in range(B):
        for k in range(K):
            px, py = int(x[b, k].item()), int(y[b, k].item())
            if 0 < px < W - 1 and 0 < py < H - 1:
                diff_x = heatmap[b, k, py, px + 1] - heatmap[b, k, py, px - 1]
                hess_x = heatmap[b, k, py, px - 1] - 2 * heatmap[b, k, py, px] + heatmap[b, k, py, px + 1]
                diff_y = heatmap[b, k, py + 1, px] - heatmap[b, k, py - 1, px]
                hess_y = heatmap[b, k, py - 1, px] - 2 * heatmap[b, k, py, px] + heatmap[b, k, py + 1, px]
                shift_x = -0.5 * diff_x / (hess_x + 1e-6)
                shift_y = -0.5 * diff_y / (hess_y + 1e-6)
                if torch.abs(shift_x) < 1.0:
                    x[b, k] += shift_x
                if torch.abs(shift_y) < 1.0:
                    y[b, k] += shift_y

    x_norm = x / (W - 1)
    y_norm = y / (H - 1)
    return torch.stack([x_norm, y_norm], dim=-1)
