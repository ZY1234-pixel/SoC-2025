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
    def __init__(self, hidden_dim=128, scale=8):
        super(UpdateBlock, self).__init__()
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, scale * scale * 9, 1, padding=0))

    def forward(self, imgf, coords1):
        mask = 0.25 *self.mask(imgf)  # scale mask to balence gradients  # 0.25 *
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
        flow_up = self.upsample_flow(coords1 - coords0, mask)
        bm_up = coodslar + flow_up
        return bm_up

