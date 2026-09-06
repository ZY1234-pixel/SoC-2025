"""Training-only teacher head that can consume compact FLUX latent evidence."""

import torch
from torch import nn
from torch.nn import functional as F


class _Block(nn.Sequential):
    def __init__(self, source: int, target: int):
        super().__init__(
            nn.Conv2d(source, target, 3, padding=1, bias=False),
            nn.BatchNorm2d(target),
            nn.SiLU(inplace=True),
            nn.Conv2d(target, target, 3, padding=1, bias=False),
            nn.BatchNorm2d(target),
            nn.SiLU(inplace=True),
        )


class FluxTeacherMaskHead(nn.Module):
    """A discarded-after-distillation, higher-capacity watermark mask teacher."""

    def __init__(self):
        super().__init__()
        self.image_mean = (0.485, 0.456, 0.406)
        self.image_std = (0.229, 0.224, 0.225)
        self.stem = _Block(6, 48)
        self.down1 = nn.Sequential(nn.Conv2d(48, 96, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(96), nn.SiLU(inplace=True), _Block(96, 96))
        self.down2 = nn.Sequential(nn.Conv2d(96, 160, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(160), nn.SiLU(inplace=True), _Block(160, 160))
        self.down3 = nn.Sequential(nn.Conv2d(160, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.SiLU(inplace=True), _Block(256, 256))
        self.down4 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.SiLU(inplace=True), _Block(256, 256))
        self.latent_proj = nn.Sequential(nn.Conv2d(8, 256, 1, bias=False), nn.BatchNorm2d(256), nn.SiLU(inplace=True))
        self.fuse = _Block(512, 256)
        self.up3 = _Block(256 + 256, 160)
        self.up2 = _Block(160 + 160, 96)
        self.up1 = _Block(96 + 96, 48)
        self.output = nn.Sequential(_Block(48 + 48, 32), nn.Conv2d(32, 1, 1))

    def forward(self, source: torch.Tensor, candidate: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if candidate.shape[-2:] != source.shape[-2:]:
            candidate = F.interpolate(candidate, size=source.shape[-2:], mode="bilinear", align_corners=False)
        mean = source.new_tensor(self.image_mean).view(1, 3, 1, 1)
        std = source.new_tensor(self.image_std).view(1, 3, 1, 1)
        source = (source - mean) / std
        candidate = (candidate - mean) / std
        e0 = self.stem(torch.cat((source, candidate), dim=1))
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        latent = F.interpolate(latent, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat((e4, self.latent_proj(latent)), dim=1))
        x = F.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat((x, e3), dim=1))
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat((x, e2), dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat((x, e1), dim=1))
        x = F.interpolate(x, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(self.output(torch.cat((x, e0), dim=1)), size=source.shape[-2:], mode="bilinear", align_corners=False)


if __name__ == "__main__":
    model = FluxTeacherMaskHead().eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 512, 512), torch.randn(1, 3, 512, 512), torch.randn(1, 8, 32, 32))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert output.shape == (1, 1, 512, 512)
    print(f"shape={tuple(output.shape)} parameters={parameters:,} fp32={parameters * 4 / 2**20:.2f} MiB")
