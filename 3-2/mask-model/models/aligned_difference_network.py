"""Lightweight alignment-aware paired watermark mask detector."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

try:
    from .network import _Smooth
except ImportError:
    from network import _Smooth


class _Interaction(nn.Module):
    def __init__(self, channels: int, width: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(channels * 4, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            _Smooth(width),
        )

    def forward(self, source: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        difference = (source - candidate).abs()
        # Bounded product keeps the common-content cue numerically stable.
        common = torch.tanh(source) * torch.tanh(candidate)
        return self.project(torch.cat((source, candidate, difference, common), dim=1))


class AlignedDifferenceMaskNet(nn.Module):
    """Siamese multi-scale detector with one low-resolution local alignment block.

    The alignment starts as identity (zero-initialized offset head), so it can
    only learn a correction when the paired candidate is locally misregistered.
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
        self.interactions = nn.ModuleList(_Interaction(half, width) for _ in channels)
        self.smooth = nn.ModuleList(_Smooth(width) for _ in range(4))
        self.offset = nn.Sequential(
            nn.Conv2d(half * 3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        self.detail = nn.Sequential(
            nn.Conv2d(6, 16, 3, stride=2, padding=1, bias=False),
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
        self.source_head = nn.Conv2d(half, 1, 1)
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

    @staticmethod
    def _align(candidate: torch.Tensor, source: torch.Tensor, offset_head: nn.Module) -> torch.Tensor:
        difference = (source - candidate).abs()
        offset = offset_head(torch.cat((source, candidate, difference), dim=1)).tanh() * 4.0
        height, width = candidate.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=candidate.device, dtype=candidate.dtype),
            torch.linspace(-1, 1, width, device=candidate.device, dtype=candidate.dtype),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1).expand(candidate.shape[0], -1, -1, -1).clone()
        grid[..., 0] += offset[:, 0] * 2.0 / max(1, width - 1)
        grid[..., 1] += offset[:, 1] * 2.0 / max(1, height - 1)
        return F.grid_sample(candidate, grid, mode="bilinear", padding_mode="border", align_corners=True)

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
        candidate_projected[2] = self._align(candidate_projected[2], source_projected[2], self.offset)
        fused_features = [
            interaction(source_feature, candidate_feature)
            for interaction, source_feature, candidate_feature in zip(
                self.interactions, source_projected, candidate_projected
            )
        ]
        decoded = self._decode(fused_features, self.smooth)
        rgb_difference = (source_normalized - candidate_normalized).abs()
        decoded = self.final_fuse(torch.cat((decoded, self.detail(torch.cat((source_normalized, rgb_difference), dim=1))), dim=1))
        prediction = F.interpolate(self.mask_head(decoded), size=output_size, mode="bilinear", align_corners=False)
        if not return_aux:
            return prediction
        source_logits = F.interpolate(self.source_head(source_projected[0]), size=output_size, mode="bilinear", align_corners=False)
        difference_logits = F.interpolate(self.difference_head(fused_features[0]), size=output_size, mode="bilinear", align_corners=False)
        return prediction, (source_logits, difference_logits)


if __name__ == "__main__":
    model = AlignedDifferenceMaskNet(pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 257, 385), torch.randn(1, 3, 257, 385))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert output.shape == (1, 1, 257, 385) and parameters < 5_000_000
    print(f"shape={tuple(output.shape)} parameters={parameters:,} fp32={parameters * 4 / 2**20:.2f} MiB")
