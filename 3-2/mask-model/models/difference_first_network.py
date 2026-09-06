"""Compact paired mask detector with explicit multi-scale difference features."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

try:
    from .network import _Smooth
except ImportError:
    from network import _Smooth


class DifferenceFirstMaskNet(nn.Module):
    """Siamese MobileNet whose decoder is driven by source/candidate differences."""

    def __init__(self, width: int = 64, pretrained: bool = True):
        super().__init__()
        self.encoder = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        ).features
        self.feature_ids = {1, 3, 6, 12, 16}
        channels = (16, 24, 40, 112, 960)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.lateral = nn.ModuleList(nn.Conv2d(c, width // 2, 1) for c in channels)
        self.diff_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width // 2, width, 1, bias=False),
                nn.BatchNorm2d(width), nn.SiLU(inplace=True), _Smooth(width)
            ) for _ in channels
        )
        self.smooth = nn.ModuleList(_Smooth(width) for _ in range(4))
        self.detail = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.SiLU(inplace=True), _Smooth(16)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(width + 16, width, 1, bias=False),
            nn.BatchNorm2d(width), nn.SiLU(inplace=True), _Smooth(width)
        )
        self.mask_head = nn.Sequential(_Smooth(width), nn.Conv2d(width, 1, 1))
        self.source_head = nn.Conv2d(width // 2, 1, 1)
        self.difference_head = nn.Conv2d(width, 1, 1)

    def _encode(self, image):
        features, value = [], image
        for index, layer in enumerate(self.encoder):
            value = layer(value)
            if index in self.feature_ids:
                features.append(value)
        return features

    def forward(self, source, candidate, return_aux=False):
        if candidate.shape[-2:] != source.shape[-2:]:
            candidate = F.interpolate(candidate, size=source.shape[-2:], mode="bilinear", align_corners=False)
        output_size = source.shape[-2:]
        source = (source - self.image_mean) / self.image_std
        candidate = (candidate - self.image_mean) / self.image_std
        source_features, candidate_features = self._encode(source), self._encode(candidate)
        source_projected = [layer(feature) for layer, feature in zip(self.lateral, source_features)]
        candidate_projected = [layer(feature) for layer, feature in zip(self.lateral, candidate_features)]
        differences = [block((s - c).abs()) for block, s, c in zip(self.diff_blocks, source_projected, candidate_projected)]
        value = differences[-1]
        pyramid = []
        for index in range(3, -1, -1):
            value = F.interpolate(value, size=differences[index].shape[-2:], mode="bilinear", align_corners=False)
            value = self.smooth[index](value + differences[index])
            pyramid.append(value)
        rgb_difference = (source - candidate).abs()
        value = self.fuse(torch.cat((value, self.detail(rgb_difference)), dim=1))
        prediction = F.interpolate(self.mask_head(value), size=output_size, mode="bilinear", align_corners=False)
        if not return_aux:
            return prediction
        source_logits = F.interpolate(self.source_head(source_projected[0]), size=output_size, mode="bilinear", align_corners=False)
        difference_logits = F.interpolate(self.difference_head(differences[0]), size=output_size, mode="bilinear", align_corners=False)
        return prediction, (source_logits, difference_logits)


if __name__ == "__main__":
    model = DifferenceFirstMaskNet(pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 257, 385), torch.randn(1, 3, 257, 385), return_aux=True)
    parameters = sum(p.numel() for p in model.parameters())
    assert output[0].shape == (1, 1, 257, 385) and parameters < 5_000_000
    print(f"shape={tuple(output[0].shape)} parameters={parameters:,}")
