"""Improved paired detector: bounded alignment and difference-first heads."""

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .aligned_difference_network import AlignedDifferenceMaskNet
except ImportError:
    from aligned_difference_network import AlignedDifferenceMaskNet


class AlignedDifferenceV2MaskNet(AlignedDifferenceMaskNet):
    """Keep multi-scale interaction while preventing source/RGB shortcuts."""

    def __init__(self, width: int = 64, pretrained: bool = True):
        super().__init__(width=width, pretrained=pretrained)
        self.detail = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            self.detail[3],
        )

    @staticmethod
    def _align(candidate: torch.Tensor, source: torch.Tensor, offset_head: nn.Module) -> torch.Tensor:
        difference = (source - candidate).abs()
        offset = offset_head(torch.cat((source, candidate, difference), dim=1)).tanh() * 1.5
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
        source_highpass = source_normalized - F.avg_pool2d(source_normalized, 5, stride=1, padding=2)
        candidate_highpass = candidate_normalized - F.avg_pool2d(candidate_normalized, 5, stride=1, padding=2)
        detail_difference = (source_highpass - candidate_highpass).abs()
        decoded = self.final_fuse(torch.cat((decoded, self.detail(detail_difference)), dim=1))
        prediction = F.interpolate(self.mask_head(decoded), size=output_size, mode="bilinear", align_corners=False)
        if not return_aux:
            return prediction
        source_logits = F.interpolate(self.source_head(source_projected[0]), size=output_size, mode="bilinear", align_corners=False)
        # The base interaction already projects the finest-scale difference to
        # ``width`` channels, which is exactly what ``difference_head`` expects.
        # Keep the auxiliary target on this shared representation instead of
        # referencing a non-existent lateral layer.
        difference_logits = F.interpolate(self.difference_head(fused_features[0]), size=output_size, mode="bilinear", align_corners=False)
        return prediction, (source_logits, difference_logits)


if __name__ == "__main__":
    model = AlignedDifferenceV2MaskNet(pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 257, 385), torch.randn(1, 3, 257, 385))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert output.shape == (1, 1, 257, 385) and parameters < 5_000_000
    print(f"shape={tuple(output.shape)} parameters={parameters:,}")
