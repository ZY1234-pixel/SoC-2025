# -*- coding: utf-8 -*-
# @Time : 2023/7/29 17:29
# @Author : Heng LI
# @FileName: dewarp_utils.py
# @Software: PyCharm

import torch
from torch import nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.bn(self.conv1(x))))


def coords_grid(batch, ht, wd, gap=1):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    coords = coords[:, ::gap, ::gap]
    return coords[None].repeat(batch, 1, 1, 1)


class UpdateBlock(nn.Module):
    """原始 D2Dewarp 的单步更新块（被 DewarpUP 使用，保持行为不变）。"""

    def __init__(self, hidden_dim=128, scale=8):
        super(UpdateBlock, self).__init__()
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, scale * scale * 9, 1, padding=0))

    def forward(self, imgf, coords1):
        mask = 0.25 * self.mask(imgf)  # scale mask to balence gradients
        dflow = self.flow_head(imgf)
        coords1 = coords1 + dflow

        return mask, coords1


class DewarpUP(nn.Module):
    def __init__(self, scale, hidden_dim=16):
        super(DewarpUP, self).__init__()
        self.scale = scale

        self.update_block = UpdateBlock(hidden_dim=hidden_dim, scale=scale)

    def initialize_flow(self, img, H, W):
        # N, C, H, W = img.shape
        N = img.shape[0]
        coodslar = coords_grid(N, H, W).to(img.device)
        coords0 = coords_grid(N, H // self.scale, W // self.scale).to(img.device)
        coords1 = coords_grid(N, H // self.scale, W // self.scale).to(img.device)

        return coodslar, coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, self.scale, self.scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.scale * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, self.scale * H, self.scale * W)

    def forward(self, img, H, W, feature):
        coodslar, coords0, coords1 = self.initialize_flow(img, H, W)
        coords1 = coords1.detach()

        mask, coords1 = self.update_block(feature, coords1)

        # # 打印
        # with torch.no_grad():
        #     dflow = coords1 - coords0
        #     print("dflow max:", dflow.abs().max().item())
        #     print("dflow mean:", dflow.abs().mean().item())

        flow_up = self.upsample_flow(coords1 - coords0, mask)
        bm_up = coodslar + flow_up
        return bm_up


class ConvGRU(nn.Module):
    """RAFT 的卷积 GRU：把上一次的 hidden state 与本次的 motion feature 融合。"""

    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=p)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=p)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=p)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class UpdateBlockV1(nn.Module):
    """单步迭代更新：ConGRU 更新 hidden -> 预测残差 flow -> 预测凸上采样 mask。"""

    def __init__(self, hidden_dim=128, scale=8, emb_dim=64):
        super().__init__()
        self.flow_emb = nn.Sequential(
            nn.Conv2d(2, emb_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim, emb_dim, 3, padding=1), nn.ReLU(inplace=True))
        self.gru = ConvGRU(hidden_dim, hidden_dim + emb_dim)
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 3, padding=1))
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, scale * scale * 9, 1, padding=0))

    def forward(self, net, inp, coords0, coords1):
        f = self.flow_emb(coords1 - coords0)
        net = self.gru(net, torch.cat([inp, f], dim=1))
        coords1 = coords1 + self.flow_head(net)
        return net, 0.25 * self.mask(net), coords1


class IterativeDewarpUP(nn.Module):
    """RAFT 式迭代精修 + 凸上采样，输出绝对坐标 backward map。"""

    def __init__(self, scale=8, hidden_dim=576, iters=4):
        super().__init__()
        self.scale = scale
        self.iters = iters
        self.context = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(inplace=True))
        self.update_block = UpdateBlockV1(hidden_dim=hidden_dim, scale=scale)

    def initialize_flow(self, img, H, W):
        N = img.shape[0]
        coodslar = coords_grid(N, H, W).to(img.device)
        coords0 = coords_grid(N, H // self.scale, W // self.scale).to(img.device)
        coords1 = coords_grid(N, H // self.scale, W // self.scale).to(img.device)
        return coodslar, coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, self.scale, self.scale, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(self.scale * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, self.scale * H, self.scale * W)

    def forward(self, img, H, W, feature, bm_init=None, iters=None):
        """
        feature : [B, C, H/8, W/8] 融合特征
        bm_init : [B, 2, H, W] 零阶展平坐标先验（高分辨率绝对坐标），可为 None
        返回    : list[[B,2,H,W]]，每个迭代一个 bm，最后一个最准
        """
        iters = iters or self.iters
        coodslar, coords0, coords1 = self.initialize_flow(img, H, W)
        coords0 = coords0.detach()

        if bm_init is not None:   # 用零阶先验初始化，而不是单位网格
            coords1 = F.interpolate(bm_init, size=coords0.shape[2:],
                                    mode='bilinear', align_corners=False) / self.scale
        coords1 = coords1.detach()

        net = torch.relu(self.context(feature))
        inp = torch.relu(feature)

        flows = []
        for _ in range(iters):
            net, mask, coords1 = self.update_block(net, inp, coords0, coords1)
            flows.append(coodslar + self.upsample_flow(coords1 - coords0, mask))
        return flows

