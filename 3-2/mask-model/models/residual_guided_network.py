import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

try:
    from .network import _Smooth
except ImportError:
    from network import _Smooth


class ResidualGuidedWatermarkMaskNet(nn.Module):
    """Compact Difference-first mask detector with a single FPN."""

    def __init__(self, width: int = 64, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v3_large(weights=weights).features
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.raw_alpha = nn.Parameter(torch.tensor(0.4054651))
        self.feature_ids = {1, 3, 6, 12, 16}
        channels = (16, 24, 40, 112, 960)
        source_width = width // 2
        self.source_lateral = nn.ModuleList(nn.Conv2d(channel, source_width, 1) for channel in channels)
        self.difference_lateral = nn.ModuleList(nn.Conv2d(channel, width, 1) for channel in channels)
        self.lateral_fuse = nn.ModuleList(nn.Conv2d(source_width + width, width, 1) for _ in channels)
        self.smooth = nn.ModuleList(_Smooth(width) for _ in range(4))
        self.detail = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.SiLU(inplace=True), _Smooth(16)
        )
        self.final_fuse = nn.Sequential(
            nn.Conv2d(width + 16, width, 1, bias=False),
            nn.BatchNorm2d(width), nn.SiLU(inplace=True), _Smooth(width)
        )
        self.mask_head = nn.Sequential(_Smooth(width), nn.Conv2d(width, 1, 1))
        self.source_head = nn.Conv2d(source_width, 1, 1)
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
        difference_projected = [
            layer((source_feature - candidate_feature).abs())
            for layer, source_feature, candidate_feature in zip(
                self.difference_lateral, source_features, candidate_features
            )
        ]
        fused_features = [
            fuse(torch.cat((source_feature, difference_feature), dim=1))
            for fuse, source_feature, difference_feature in zip(
                self.lateral_fuse, source_projected, difference_projected
            )
        ]
        decoded = self._decode(fused_features, self.smooth)
        rgb_difference = (source_normalized - candidate_normalized).abs()
        decoded = self.final_fuse(torch.cat((decoded, self.detail(rgb_difference)), dim=1))
        prediction = F.interpolate(self.mask_head(decoded), size=output_size, mode="bilinear", align_corners=False)
        if not return_aux:
            return prediction
        source_logits = F.interpolate(self.source_head(source_projected[0]), size=output_size, mode="bilinear", align_corners=False)
        difference_logits = F.interpolate(self.difference_head(difference_projected[0]), size=output_size, mode="bilinear", align_corners=False)
        return prediction, (source_logits, difference_logits)


if __name__ == "__main__":
    model = ResidualGuidedWatermarkMaskNet(pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 257, 385), torch.randn(1, 3, 257, 385))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert output.shape == (1, 1, 257, 385) and parameters < 5_000_000
    print(f"shape={tuple(output.shape)} parameters={parameters:,} fp32={parameters * 4 / 2**20:.2f} MiB")
