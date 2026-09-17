"""Mask2Flow 的损失函数集合

四类监督：
  1. bm        : 迭代加权 L1（越靠后的迭代权重越大）
  2. coord     : 分支 H / 分支 V 的归一化坐标场 L1（GT 直接从 bm 切出，零标注成本）
  3. smooth    : 二阶平滑（弯曲能量），稀疏输入下抑制欠约束区域的抖动
  4. straight  : 自监督直线度/矩形度，用输入掩码本身构造，无需额外标注
"""

import torch
import torch.nn.functional as F


def masked_l1(pred, gt, mask):
    """只在文档内部区域统计 bm 误差。"""
    d = (pred - gt).abs().sum(dim=1, keepdim=True)        # [B,1,H,W]
    return (d * mask).sum() / (mask.sum() * 2 + 1e-6)


def coord_l1(pred, gt, mask):
    return ((pred - gt).abs() * mask).sum() / (mask.sum() + 1e-6)


def smooth_loss(bm, order=2):
    """二阶平滑（弯曲能量），稀疏输入下必须加，抑制欠约束区域的抖动。"""
    dy = bm[:, :, 1:, :] - bm[:, :, :-1, :]
    dx = bm[:, :, :, 1:] - bm[:, :, :, :-1]
    if order == 1:
        return dx.abs().mean() + dy.abs().mean()
    dyy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
    dxx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
    dxy = dy[:, :, :, 1:] - dy[:, :, :, :-1]
    return dxx.abs().mean() + dyy.abs().mean() + 2.0 * dxy.abs().mean()


def warp_mask(mask, bm, H, W):
    """用 backward map 把掩码重采样到展平坐标系。"""
    g = torch.cat([bm[:, 0:1] / (W - 1) * 2 - 1,
                   bm[:, 1:2] / (H - 1) * 2 - 1], dim=1).permute(0, 2, 3, 1)
    return F.grid_sample(mask, g, mode='bilinear',
                         align_corners=True, padding_mode='zeros')


def straightness_loss(masks, bm, H, W):
    """自监督：展平后水平线应水平、竖直线应竖直、边界应成矩形。
    不需要任何额外标注，直接由输入掩码构造。"""
    wh = warp_mask(masks[:, 0:1], bm, H, W)
    wv = warp_mask(masks[:, 1:2], bm, H, W)
    wb = warp_mask(masks[:, 2:3], bm, H, W)

    l_h = (wh[:, :, :, 1:] - wh[:, :, :, :-1]).abs().mean()   # 水平线沿 x 不变
    l_v = (wv[:, :, 1:, :] - wv[:, :, :-1, :]).abs().mean()   # 竖直线沿 y 不变

    gx = (wb[:, :, :, 1:] - wb[:, :, :, :-1]).abs()[:, :, :-1, :]
    gy = (wb[:, :, 1:, :] - wb[:, :, :-1, :]).abs()[:, :, :, :-1]
    l_b = (gx * gy).mean()     # 矩形边：至少一个方向梯度为 0 -> 惩罚乘积
    return l_h + l_v + 2.0 * l_b


def mask2flow_loss(masks, gt_bm, coord_x, coord_y, inside, flows,
                   w_bm=5.0, w_coord=1.0, w_smooth=0.05, w_str=0.5, gamma=0.8,
                   mask_bm_with_inside=False):
    """Args:
        mask_bm_with_inside: 是否用 inside 掩膜 bm / coord 的监督区域。

            默认 False —— 即【全图监督】。原因是域不匹配：
              * inside 定义在【畸变图】上（由 boundary 累积求交得到）
              * bm 和 coord 都定义在【展平图】网格上（bm[i,j] = 展平图像素 (i,j)
                应该去畸变图哪里采样）
            两者不是同一个坐标系，拿 inside 去掩膜 bm 会丢掉约 45% 的监督。
            而且展平帧本身完全落在文档内（GT bm 的像就是文档区域，没有背景），
            所以每一个展平像素都是有效监督点，全图监督才是对的。

            置 True 可切回 inside 掩膜，用于消融对比。
    """
    B, _, H, W = gt_bm.shape
    if mask_bm_with_inside:
        m = F.interpolate(inside, size=(H, W), mode='nearest')
    else:
        m = torch.ones((B, 1, H, W), device=gt_bm.device, dtype=gt_bm.dtype)

    loss_bm = 0.0
    for i, bm_i in enumerate(flows):                    # 迭代加权监督
        w = gamma ** (len(flows) - 1 - i)
        loss_bm = loss_bm + w * masked_l1(bm_i, gt_bm, m)
    loss_bm = loss_bm / sum(gamma ** i for i in range(len(flows)))

    bm = flows[-1]
    loss_cx = coord_l1(coord_x, gt_bm[:, 0:1] / (W - 1), m)
    loss_cy = coord_l1(coord_y, gt_bm[:, 1:2] / (H - 1), m)
    loss_sm = smooth_loss(bm)
    loss_st = straightness_loss(masks, bm, H, W)

    total = (w_bm * loss_bm + w_coord * (loss_cx + loss_cy)
             + w_smooth * loss_sm + w_str * loss_st)
    return total, {'bm': loss_bm, 'coord_x': loss_cx, 'coord_y': loss_cy,
                   'smooth': loss_sm, 'straight': loss_st}
