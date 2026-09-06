"""Lightweight paired detector: difference proposals filtered by image context."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

try:
    from .network import _Smooth
except ImportError:
    from network import _Smooth


class DifferenceGateMaskNet(nn.Module):
    """Difference-first decoder with a learned semantic gate at every scale."""

    def __init__(self, width=64, pretrained=True):
        super().__init__()
        self.encoder = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None).features
        self.feature_ids = {1, 3, 6, 12, 16}
        channels = (16, 24, 40, 112, 960)
        half = width // 2
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.lateral = nn.ModuleList(nn.Conv2d(c, half, 1) for c in channels)
        self.diff = nn.ModuleList(nn.Sequential(nn.Conv2d(half, width, 1, bias=False), nn.BatchNorm2d(width), nn.SiLU(inplace=True), _Smooth(width)) for _ in channels)
        self.context = nn.ModuleList(nn.Sequential(nn.Conv2d(half * 3, width, 1, bias=False), nn.BatchNorm2d(width), nn.SiLU(inplace=True), _Smooth(width)) for _ in channels)
        self.gates = nn.ModuleList(nn.Sequential(nn.Conv2d(half * 3, width, 1), nn.Sigmoid()) for _ in channels)
        self.smooth = nn.ModuleList(_Smooth(width) for _ in range(4))
        self.detail = nn.Sequential(nn.Conv2d(6, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16), nn.SiLU(inplace=True), _Smooth(16))
        self.fuse = nn.Sequential(nn.Conv2d(width + 16, width, 1, bias=False), nn.BatchNorm2d(width), nn.SiLU(inplace=True), _Smooth(width))
        self.mask_head = nn.Sequential(_Smooth(width), nn.Conv2d(width, 1, 1))
        self.source_head = nn.Conv2d(half, 1, 1)
        self.difference_head = nn.Conv2d(width, 1, 1)

    def _encode(self, image):
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
        fused = []
        for d, ctx, gate, s, c in zip(self.diff, self.context, self.gates, sp, cp):
            features = torch.cat((s, c, (s - c).abs()), dim=1)
            g = gate(features)
            fused.append(g * d((s - c).abs()) + (1 - g) * ctx(features))
        value = fused[-1]
        pyramid = []
        for index in range(3, -1, -1):
            value = F.interpolate(value, size=fused[index].shape[-2:], mode="bilinear", align_corners=False)
            value = self.smooth[index](value + fused[index])
            pyramid.append(value)
        rgb_difference = (source - candidate).abs()
        gradient_difference = (source[:, :, :, 1:] - source[:, :, :, :-1]).abs()
        gradient_difference = F.pad(gradient_difference, (0, 1, 0, 0))
        candidate_gradient = (candidate[:, :, :, 1:] - candidate[:, :, :, :-1]).abs()
        candidate_gradient = F.pad(candidate_gradient, (0, 1, 0, 0))
        edge_difference = (gradient_difference - candidate_gradient).abs()
        value = self.fuse(torch.cat((value, self.detail(torch.cat((rgb_difference, edge_difference), dim=1))), dim=1))
        prediction = F.interpolate(self.mask_head(value), size=output_size, mode="bilinear", align_corners=False)
        if not return_aux:
            return prediction
        source_logits = F.interpolate(self.source_head(sp[0]), size=output_size, mode="bilinear", align_corners=False)
        difference_logits = F.interpolate(self.difference_head(fused[0]), size=output_size, mode="bilinear", align_corners=False)
        return prediction, (source_logits, difference_logits)


if __name__ == "__main__":
    model = DifferenceGateMaskNet(pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 257, 385), torch.randn(1, 3, 257, 385), return_aux=True)
    parameters = sum(p.numel() for p in model.parameters())
    assert output[0].shape == (1, 1, 257, 385) and parameters < 5_000_000
    print(f"shape={tuple(output[0].shape)} parameters={parameters:,}")
