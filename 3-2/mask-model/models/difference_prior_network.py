"""Difference-first paired watermark mask detector."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

try:
    from .network import _Smooth
except ImportError:
    from network import _Smooth


class _PureDifference(nn.Module):
    def __init__(self, channels: int, width: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(channels + 1, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            _Smooth(width),
        )

    def forward(self, source: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        difference = (source - candidate).abs()
        correlation = (F.normalize(source, dim=1) * F.normalize(candidate, dim=1)).sum(
            dim=1, keepdim=True
        )
        return self.project(torch.cat((difference, 1.0 - correlation), dim=1))


class DifferencePriorMaskNet(nn.Module):
    """Predict a mask primarily from paired feature differences.

    The source image only gates the decoded difference features.  It does not
    have a direct source-to-mask path, which prevents printed text from becoming
    a shortcut for watermark detection.
    """

    def __init__(self, width: int = 64, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v3_large(weights=weights).features
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.feature_ids = {1, 3, 6, 12, 16}
        channels = (16, 24, 40, 112, 960)
        half = width // 2
        self.source_lateral = nn.ModuleList(nn.Conv2d(channel, half, 1) for channel in channels)
        self.candidate_lateral = nn.ModuleList(nn.Conv2d(channel, half, 1) for channel in channels)
        self.difference_lateral = nn.ModuleList(_PureDifference(half, width) for _ in channels)
        self.smooth = nn.ModuleList(_Smooth(width) for _ in range(4))
        self.source_context = nn.Sequential(
            nn.Conv2d(half, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )
        self.context_gate = nn.Sequential(nn.Conv2d(width + 16, 1, 1), nn.Sigmoid())
        self.detail = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            _Smooth(16),
        )
        self.final_fuse = nn.Sequential(
            nn.Conv2d(width + 16, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            _Smooth(width),
        )
        self.mask_head = nn.Sequential(_Smooth(width), nn.Conv2d(width, 1, 1))
        self.source_head = nn.Conv2d(16, 1, 1)
        self.difference_head = nn.Conv2d(width, 1, 1)

    def _encode(self, image: torch.Tensor) -> list[torch.Tensor]:
        features, value = [], image
        for index, layer in enumerate(self.encoder):
            value = layer(value)
            if index in self.feature_ids:
                features.append(value)
        return features

    @staticmethod
    def _decode(features: list[torch.Tensor], smooth: nn.ModuleList) -> torch.Tensor:
        value = features[-1]
        for index in range(3, -1, -1):
            value = F.interpolate(value, size=features[index].shape[-2:], mode="bilinear", align_corners=False)
            value = smooth[index](value + features[index])
        return value

    def forward(self, source: torch.Tensor, candidate: torch.Tensor, return_aux: bool = False):
        if candidate.shape[-2:] != source.shape[-2:]:
            candidate = F.interpolate(candidate, size=source.shape[-2:], mode="bilinear", align_corners=False)
        output_size = source.shape[-2:]
        source_normalized = (source - self.image_mean) / self.image_std
        candidate_normalized = (candidate - self.image_mean) / self.image_std
        source_features = self._encode(source_normalized)
        candidate_features = self._encode(candidate_normalized)
        source_projected = [layer(feature) for layer, feature in zip(self.source_lateral, source_features)]
        candidate_projected = [layer(feature) for layer, feature in zip(self.candidate_lateral, candidate_features)]
        difference_features = [
            layer(source_feature, candidate_feature)
            for layer, source_feature, candidate_feature in zip(
                self.difference_lateral, source_projected, candidate_projected
            )
        ]
        decoded = self._decode(difference_features, self.smooth)
        context = self.source_context(source_projected[0])
        context = F.interpolate(context, size=decoded.shape[-2:], mode="bilinear", align_corners=False)
        decoded = decoded * (0.5 + 0.5 * self.context_gate(torch.cat((decoded, context), dim=1)))
        source_highpass = source_normalized - F.avg_pool2d(source_normalized, 5, stride=1, padding=2)
        candidate_highpass = candidate_normalized - F.avg_pool2d(candidate_normalized, 5, stride=1, padding=2)
        detail = self.detail((source_highpass - candidate_highpass).abs())
        decoded = self.final_fuse(torch.cat((decoded, detail), dim=1))
        prediction = F.interpolate(self.mask_head(decoded), size=output_size, mode="bilinear", align_corners=False)
        if not return_aux:
            return prediction
        source_logits = F.interpolate(self.source_head(context), size=output_size, mode="bilinear", align_corners=False)
        difference_logits = F.interpolate(self.difference_head(difference_features[0]), size=output_size, mode="bilinear", align_corners=False)
        return prediction, (source_logits, difference_logits)


if __name__ == "__main__":
    model = DifferencePriorMaskNet(pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 257, 385), torch.randn(1, 3, 257, 385))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert output.shape == (1, 1, 257, 385) and parameters < 5_000_000
    print(f"shape={tuple(output.shape)} parameters={parameters:,} fp32={parameters * 4 / 2**20:.2f} MiB")
