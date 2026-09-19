"""成对水印检测网络：以「差异」为主要证据，用图像上下文做门控。"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

try:
    from .network import _Smooth
except ImportError:
    from network import _Smooth


class DifferenceGateMaskNet(nn.Module):
    """差异优先解码器，含精细分支与可选的大块分支。"""

    # 大块分支的融合阈值：越大越保守
    large_gate = 0.7

    def __init__(self, width=64, pretrained=True, large_region=False):
        super().__init__()
        self.large_region = large_region
        # 共享编码器：原图与候选图各跑一次，权重共享
        self.encoder = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None).features
        self.feature_ids = {1, 3, 6, 12, 16}   # 对应 stride 2/4/8/16/32
        channels = (16, 24, 40, 112, 960)
        half = width // 2
        # 归一化放在模型内部，差分在归一化后的特征上计算
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.lateral = nn.ModuleList(nn.Conv2d(c, half, 1) for c in channels)
        # 差异支路、上下文支路、逐像素门控（门控决定该尺度上更信哪一支）
        self.diff = nn.ModuleList(nn.Sequential(nn.Conv2d(half, width, 1, bias=False), nn.BatchNorm2d(width), nn.SiLU(inplace=True), _Smooth(width)) for _ in channels)
        self.context = nn.ModuleList(nn.Sequential(nn.Conv2d(half * 3, width, 1, bias=False), nn.BatchNorm2d(width), nn.SiLU(inplace=True), _Smooth(width)) for _ in channels)
        self.gates = nn.ModuleList(nn.Sequential(nn.Conv2d(half * 3, width, 1), nn.Sigmoid()) for _ in channels)
        # 自顶向下的四层融合，输出 stride 4/8/16/32
        self.smooth = nn.ModuleList(_Smooth(width) for _ in range(4))
        # 细节支路：归一化 RGB 差分 + 边缘差分
        self.detail = nn.Sequential(nn.Conv2d(6, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16), nn.SiLU(inplace=True), _Smooth(16))
        self.fuse = nn.Sequential(nn.Conv2d(width + 16, width, 1, bias=False), nn.BatchNorm2d(width), nn.SiLU(inplace=True), _Smooth(width))
        self.mask_head = nn.Sequential(_Smooth(width), nn.Conv2d(width, 1, 1))
        if large_region:
            self.large_region_head = nn.Sequential(_Smooth(width), nn.Conv2d(width, 1, 1))
        # 外观（仅原图）分支。最初只是最细一层侧向特征上的一个 1x1 卷积（33 个参数），
        # 所以把它的损失权重放大 10 倍也没有任何变化（source_iou 0.0466 -> 0.0468）。
        # 它是唯一对「局部对比度低于 1 的淡水印」有响应的信号（实测一个真实印章相对
        # 其紧邻背景只有 0.63x），所以需要一个真正的解码器：与主 mask 头同构的、
        # 只看原图的 FPN。约 2.5 万参数。
        self.source_lateral = nn.ModuleList(nn.Conv2d(half, width, 1) for _ in channels)
        self.source_smooth = nn.ModuleList(_Smooth(width) for _ in range(4))
        self.source_head = nn.Sequential(_Smooth(width), nn.Conv2d(width, 1, 1))
        self.difference_head = nn.Conv2d(width, 1, 1)

    def _encode(self, image):
        """取 MobileNetV3 的五个中间层输出。"""
        out, value = [], image
        for index, layer in enumerate(self.encoder):
            value = layer(value)
            if index in self.feature_ids:
                out.append(value)
        return out

    def forward(self, source, candidate, return_aux=False):
        if candidate.shape[-2:] != source.shape[-2:]:
            candidate = F.interpolate(candidate, size=source.shape[-2:], mode="bilinear", align_corners=False)
        output_size = source.shape[-2:]
        source = (source - self.image_mean) / self.image_std
        candidate = (candidate - self.image_mean) / self.image_std
        sf, cf = self._encode(source), self._encode(candidate)
        sp = [layer(x) for layer, x in zip(self.lateral, sf)]
        cp = [layer(x) for layer, x in zip(self.lateral, cf)]

        # 逐尺度门控融合：g * 差异支路 + (1-g) * 上下文支路
        fused = []
        for d, ctx, gate, s, c in zip(self.diff, self.context, self.gates, sp, cp):
            features = torch.cat((s, c, (s - c).abs()), dim=1)
            g = gate(features)
            fused.append(g * d((s - c).abs()) + (1 - g) * ctx(features))

        # 自顶向下 FPN，得到 stride 4/8/16/32 四层
        value = fused[-1]
        pyramid = []
        for index in range(3, -1, -1):
            value = F.interpolate(value, size=fused[index].shape[-2:], mode="bilinear", align_corners=False)
            value = self.smooth[index](value + fused[index])
            pyramid.append(value)

        # 细节支路：归一化 RGB 差分 + 水平边缘差分
        rgb_difference = (source - candidate).abs()
        gradient_difference = (source[:, :, :, 1:] - source[:, :, :, :-1]).abs()
        gradient_difference = F.pad(gradient_difference, (0, 1, 0, 0))
        candidate_gradient = (candidate[:, :, :, 1:] - candidate[:, :, :, :-1]).abs()
        candidate_gradient = F.pad(candidate_gradient, (0, 1, 0, 0))
        edge_difference = (gradient_difference - candidate_gradient).abs()
        value = self.fuse(torch.cat((value, self.detail(torch.cat((rgb_difference, edge_difference), dim=1))), dim=1))

        # 主 mask 头（生产默认输出）
        prediction = F.interpolate(self.mask_head(value), size=output_size, mode="bilinear", align_corners=False)

        # 大块分支：在 stride-32 特征上出粗尺度结果
        large_logits = None
        if self.large_region:
            large_logits = F.interpolate(
                self.large_region_head(fused[-1]),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        if not return_aux:
            if large_logits is None:
                return prediction
            # 门控融合：以主 mask 头为主，只在大块分支足够自信处把大块分支
            # logaddexp 进来。logaddexp 是「或」运算、只会加面积，无条件融合会
            # 让整图滑窗精度从 0.920 掉到 0.670；完全不融合又拿不到大块水印的收益。
            return torch.where(
                large_logits.sigmoid() > self.large_gate,
                torch.logaddexp(prediction, large_logits),
                prediction,
            )

        # 外观分支的金字塔：不接触候选图，只看原图
        source_value = self.source_lateral[-1](self.lateral[-1](sf[-1]))
        source_pyramid = []
        for index in range(3, -1, -1):
            source_value = F.interpolate(
                source_value, size=sf[index].shape[-2:], mode="bilinear", align_corners=False
            )
            source_value = self.source_smooth[index](
                source_value + self.source_lateral[index](self.lateral[index](sf[index]))
            )
            source_pyramid.append(source_value)
        source_logits = F.interpolate(
            self.source_head(source_pyramid[0]), size=output_size, mode="bilinear", align_corners=False
        )
        difference_logits = F.interpolate(self.difference_head(fused[0]), size=output_size, mode="bilinear", align_corners=False)
        if large_logits is None:
            return prediction, (source_logits, difference_logits)
        return prediction, (source_logits, difference_logits, large_logits)


if __name__ == "__main__":
    # 形状与参数量自检
    model = DifferenceGateMaskNet(pretrained=False, large_region=True).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 257, 385), torch.randn(1, 3, 257, 385), return_aux=True)
    parameters = sum(p.numel() for p in model.parameters())
    assert output[0].shape == (1, 1, 257, 385) and parameters < 5_000_000
    print(f"shape={tuple(output[0].shape)} parameters={parameters:,}")
