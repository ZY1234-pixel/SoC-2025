"""Mask2Flow: (水平线掩码, 竖直线掩码, 文档边缘掩码) -> 形变场 bm

设计要点
--------
1. 稀疏二值掩码先经 MaskGeomEncoder 稠密化成几何特征 + 零阶展平坐标先验 (u, v)。
2. 双分支 U-Net：分支 H 预测归一化 **y 坐标场**，分支 V 预测归一化 **x 坐标场**
   （GT 直接从 bm 切出来，零额外标注成本），bottom 处保留 Self-Attention 做长程推理
   并互相注入对方分支的全局信息。
3. CoordAtt 做 HV 交叉融合，再与零阶先验拼接，送入 RAFT 式 ConvGRU 迭代头
   （迭代精修 + 凸上采样）输出绝对坐标 backward map。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from model_utils.utils_model import CAM_Module
from model_utils.dewarp_utils import IterativeDewarpUP
from networks.cross_attn import SelfAttention
from networks.mask_encoding import MaskGeomEncoder, invert_uv_prior
from networks.unet_parts import DoubleConv, Down, Up
from networks.d2dewarp_model import CoordAtt


class GeoBranch(nn.Module):
    """单条几何分支。分支 H 预测归一化 y 坐标场，分支 V 预测归一化 x 坐标场。"""

    def __init__(self, in_ch=64, d_model=256, bilinear=True, n_position=1024):
        super().__init__()
        c1, c2, c3 = 128, 196, 256
        self.down1 = Down(in_ch, c1)      # H/2
        self.down2 = Down(c1, c2)         # H/4
        self.down3 = Down(c2, c3)         # H/8
        self.down4 = Down(c3, d_model)    # H/16
        n_head = 8
        self.attn = SelfAttention(n_layers=2, n_head=n_head,
                                  d_k=d_model // n_head, d_v=d_model // n_head,
                                  d_model=d_model, n_position=n_position,
                                  d_inner=d_model * 2)
        self.up1 = Up(d_model + c3, 196, bilinear)
        self.up2 = Up(196 + c2, 128, bilinear)
        self.up3 = Up(128 + c1, 64, bilinear)
        self.up4 = Up(64 + in_ch, 48, bilinear)
        self.head = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1))

    def bottom_up(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        n, c, h, w = x4.size()
        a = self.attn(x4).transpose(1, 2).contiguous().view(n, c, h, w)
        return (x1, x2, x3), x4 + a

    def top_down(self, skips, x4, x):
        x1, x2, x3 = skips
        y = self.up1(x4, x3); s1 = y
        y = self.up2(y, x2);  s2 = y
        y = self.up3(y, x1);  s3 = y
        y = self.up4(y, x);   s4 = y
        return (s1, s2, s3, s4), torch.sigmoid(self.head(y))


class Mask2FlowModel(nn.Module):
    def __init__(self, img_size=448, in_chans=3, d_model=256,
                 scale=8, iters=4, bilinear=True, grad_ckpt=False):
        super().__init__()
        self.img_size = img_size
        self.d_model = d_model
        self.scale = scale
        self.in_chans = in_chans
        # 两条 U-Net 分支都在 448x448 全分辨率上跑，激活显存很大；
        # 开启后对 encoder + 两条分支做重计算，显存可降约 45%，代价约 30% 速度。
        self.grad_ckpt = grad_ckpt

        self.encoder = MaskGeomEncoder(out_ch=64, dt_ch=4)
        self.proj_h = DoubleConv(64 + 1 + 4, 64)      # 共享特征 + 行序场 + 软距离场
        self.proj_v = DoubleConv(64 + 1 + 4, 64)

        n_pos = (img_size // 16) ** 2
        self.branch_h = GeoBranch(64, d_model, bilinear, n_pos)
        self.branch_v = GeoBranch(64, d_model, bilinear, n_pos)

        # bottom 处双分支交叉
        self.cross_h = nn.Conv2d(d_model, d_model, 1)
        self.cross_v = nn.Conv2d(d_model, d_model, 1)

        self.cam_1, self.cam_2 = CAM_Module(), CAM_Module()
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(196 + 128 + 64 + 48, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model), nn.ReLU(inplace=True))
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(196 + 128 + 64 + 48, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model), nn.ReLU(inplace=True))
        self.fusion_block = CoordAtt(img_size, d_model, scale=scale)

        # 零阶先验嵌入
        self.prior_emb = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True))

        self.dewarp_up = IterativeDewarpUP(
            scale=scale, hidden_dim=d_model * 2 + 64, iters=iters)

    def forward(self, masks, iters=None):
        """
        masks : [B,3,H,W] 值 0/1，顺序 = (h_line, v_line, boundary)
        返回  : coord_x, coord_y, inside, flows
                 flows 为 list[[B,2,H,W]]，取 flows[-1] 作为最终 bm
        """
        B, _, H, W = masks.shape

        def run(fn, *args):
            # 仅在输入张量本身 requires_grad 时才用梯度重计算；
            # 否则（如 encoder 直接吃 masks，masks 是数据张量、不求导）checkpoint
            # 不会真正生效，还会触发 "None of the inputs have requires_grad" 警告。
            needs_ckpt = self.grad_ckpt and self.training and any(
                isinstance(a, torch.Tensor) and a.requires_grad for a in args)
            if needs_ckpt:
                return cp.checkpoint(fn, *args)
            return fn(*args)

        f0, inside, uv_prior, oh, ov, dh, dv = run(self.encoder, masks)

        xh = self.proj_h(torch.cat([f0, oh, dh], dim=1))
        xv = self.proj_v(torch.cat([f0, ov, dv], dim=1))

        sk_h, bh = run(self.branch_h.bottom_up, xh)
        sk_v, bv = run(self.branch_v.bottom_up, xv)
        bh_f = bh + self.cross_v(bv)      # 互相注入对方分支的全局信息
        bv_f = bv + self.cross_h(bh)

        (h1, h2, h3, h4), coord_y = run(self.branch_h.top_down, sk_h, bh_f, xh)
        (v1, v2, v3, v4), coord_x = run(self.branch_v.top_down, sk_v, bv_f, xv)

        hs = H // self.scale

        def cat8(t):
            return torch.cat([F.interpolate(x, size=(hs, hs), mode='bilinear',
                                            align_corners=False) for x in t], dim=1)

        h_map = self.cam_1(self.conv3x3_1(cat8((h1, h2, h3, h4))))
        v_map = self.cam_2(self.conv3x3_2(cat8((v1, v2, v3, v4))))
        h_map, v_map = self.fusion_block(h_map, v_map)

        prior = self.prior_emb(F.interpolate(uv_prior, size=(hs, hs),
                                             mode='bilinear', align_corners=False))
        union_map = torch.cat([h_map, v_map, prior], dim=1)

        # 零阶 backward map：(u,v) 是前向参数化，必须求逆后才能用作 bm 的初始化
        with torch.no_grad():
            bm_init = invert_uv_prior(inside)
        flows = self.dewarp_up(masks, H, W, union_map, bm_init=bm_init, iters=iters)
        return coord_x, coord_y, inside, flows


if __name__ == '__main__':
    m = torch.randint(0, 2, (2, 3, 448, 448)).float().cuda()
    m[:, 2, :, :] = 0.0
    m[:, 2, 40:410, 40] = 1.0
    m[:, 2, 40:410, 408] = 1.0
    m[:, 2, 40, 40:410] = 1.0
    m[:, 2, 408, 40:410] = 1.0

    net = Mask2FlowModel(img_size=448, d_model=256, iters=4).cuda()
    cx, cy, ins, flows = net(m)
    print('coord_x', cx.shape, 'coord_y', cy.shape, 'inside', ins.shape)
    print('bm     ', flows[-1].shape, 'iters', len(flows))
    print('params ', sum(p.numel() for p in net.parameters()) / 1e6, 'M')
