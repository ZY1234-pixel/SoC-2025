import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)


class _Smooth(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )


class WatermarkMaskNet(nn.Module):
    """RGB-only mask model with separate semantic and detail paths."""

    def __init__(self, width: int = 64, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v3_large(weights=weights).features
        self.feature_ids = {1, 3, 6, 12, 16}
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in (16, 24, 40, 112, 960))
        self.smooth = nn.ModuleList(_Smooth(width) for _ in range(4))
        self.detail = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            _Smooth(24),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(width + 24, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            _Smooth(width),
        )
        self.mask_head = nn.Sequential(_Smooth(width), nn.Conv2d(width, 1, 1))
        self.aux_heads = nn.ModuleList(nn.Conv2d(width, 1, 1) for _ in range(2))

    def forward(
        self, image: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        output_size = image.shape[-2:]
        detail = self.detail(image)
        features = []
        x = image
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in self.feature_ids:
                features.append(x)

        x = self.lateral[-1](features[-1])
        pyramid = []
        for index in range(3, -1, -1):
            x = F.interpolate(x, size=features[index].shape[-2:], mode="bilinear", align_corners=False)
            x = self.smooth[index](x + self.lateral[index](features[index]))
            pyramid.append(x)
        x = self.fuse(torch.cat((x, detail), dim=1))
        prediction = self.mask_head(x)
        prediction = F.interpolate(prediction, size=output_size, mode="bilinear", align_corners=False)
        if not return_aux:
            return prediction
        auxiliary = tuple(
            F.interpolate(head(feature), size=output_size, mode="bilinear", align_corners=False)
            for head, feature in zip(self.aux_heads, pyramid[-2:])
        )
        return prediction, auxiliary


if __name__ == "__main__":
    model = WatermarkMaskNet(pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 257, 385))
    parameters = sum(p.numel() for p in model.parameters())
    assert output.shape == (1, 1, 257, 385)
    assert parameters < 5_000_000
    print(f"shape={tuple(output.shape)} parameters={parameters:,} fp32={parameters * 4 / 2**20:.2f} MiB")
