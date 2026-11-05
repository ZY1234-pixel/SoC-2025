"""并行 CBAM + RCCA 特征增强融合（作用于 UVDocnet 的 out3）。"""

import torch
import torch.nn as nn

from attention.CBAM import CBAMBlock
from attention.RCCA import RCCAModule


class ParallelAttentionFusion(nn.Module):
    """两个注意力分支并行增强特征，再与原始特征相加融合。

        out = x + gamma * fuse( concat([CBAM(x), RCCA(x)]) )

    设计要点：
    - CBAM(x) / RCCA(x) 都是"增强后的特征"，直接 concat 后由 1x1 conv
      学习如何融合；不做 (attn(x) - x) 这类差分，避免把注意力退化成
      残差校正 —— 注意力在这里是特征增强，不是误差修正。
    - 保留 x 的直连相加，使融合结果仍是「原特征 + 增强量」。
    - 本模块参数不在 model/best_model.pkl 中：训练时用 strict=False 加载
      预训练权重，新模块随网络一起微调。
    - gamma 为可学习缩放。默认按常规初始化（不强制恒等），以符合
      「strict=False 加载 + 微调」的用法；若希望加载预训练权重后的初始
      输出保持等价，可传 init_identity=True 把 fuse 层零初始化。
    """

    def __init__(self, channels=128, cbam_reduction=16, cbam_kernel_size=7,
                 rcca_recurrence=2, init_identity=False):
        super().__init__()
        self.cbam = CBAMBlock(channel=channels, reduction=cbam_reduction,
                              kernel_size=cbam_kernel_size)
        self.rcca = RCCAModule(channels, channels)
        self.rcca_recurrence = rcca_recurrence
        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
        if init_identity:
            nn.init.zeros_(self.fuse.weight)
            nn.init.zeros_(self.fuse.bias)

    def forward(self, x):
        a = self.cbam(x)
        b = self.rcca(x, recurrence=self.rcca_recurrence)
        return x + self.fuse(torch.cat([a, b], dim=1))

    def extra_repr(self):
        return (f"channels={self.fuse.in_channels // 2}, "
                f"rcca_recurrence={self.rcca_recurrence}, gamma={float(self.gamma):.3f}")
