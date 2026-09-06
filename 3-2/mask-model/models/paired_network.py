from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

try:
    from .network import _Smooth
except ImportError:  # Support direct execution for the built-in shape check.
    from network import _Smooth


class PairedWatermarkMaskNet(nn.Module):
    """Predict a watermark mask from a watermarked source and a clean candidate.

    Both images pass through one shared MobileNetV3 encoder. At every encoder
    scale, the decoder receives the source feature, candidate feature, absolute
    difference, and elementwise product. The high-resolution path uses the two
    RGB images and their absolute pixel difference.
    """

    def __init__(self, width: int = 64, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v3_large(weights=weights).features
        self.feature_ids = {1, 3, 6, 12, 16}
        channels = (16, 24, 40, 112, 960)
        self.lateral = nn.ModuleList(nn.Conv2d(4 * channel, width, 1) for channel in channels)
        self.smooth = nn.ModuleList(_Smooth(width) for _ in range(4))
        self.detail = nn.Sequential(
            nn.Conv2d(9, 24, 3, stride=2, padding=1, bias=False),
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

    def _encode(self, image: torch.Tensor) -> list[torch.Tensor]:
        features = []
        feature = image
        for index, layer in enumerate(self.encoder):
            feature = layer(feature)
            if index in self.feature_ids:
                features.append(feature)
        return features

    @staticmethod
    def _paired_feature(source: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (source, candidate, (source - candidate).abs(), source * candidate),
            dim=1,
        )

    def forward(
        self,
        source: torch.Tensor,
        candidate: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if source.ndim != 4 or candidate.ndim != 4:
            raise ValueError("source and candidate must be NCHW tensors")
        if source.shape[:2] != candidate.shape[:2]:
            raise ValueError("source and candidate must have matching batch and channel dimensions")
        if candidate.shape[-2:] != source.shape[-2:]:
            candidate = F.interpolate(
                candidate,
                size=source.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        output_size = source.shape[-2:]
        detail_input = torch.cat((source, candidate, (source - candidate).abs()), dim=1)
        detail = self.detail(detail_input)
        source_features = self._encode(source)
        candidate_features = self._encode(candidate)
        paired_features = [
            lateral(self._paired_feature(source_feature, candidate_feature))
            for lateral, source_feature, candidate_feature in zip(
                self.lateral, source_features, candidate_features
            )
        ]

        feature = paired_features[-1]
        pyramid = []
        for index in range(3, -1, -1):
            feature = F.interpolate(
                feature,
                size=paired_features[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            feature = self.smooth[index](feature + paired_features[index])
            pyramid.append(feature)

        feature = self.fuse(torch.cat((feature, detail), dim=1))
        prediction = self.mask_head(feature)
        prediction = F.interpolate(
            prediction,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        if not return_aux:
            return prediction
        auxiliary = tuple(
            F.interpolate(head(level), size=output_size, mode="bilinear", align_corners=False)
            for head, level in zip(self.aux_heads, pyramid[-2:])
        )
        return prediction, auxiliary

    def initialize_from_single_image_weights(self, checkpoint_path: Path | str) -> dict[str, int]:
        """Transfer compatible tensors from the current RGB-only mask model.

        The source-image slices of the widened paired layers receive the old
        weights. Their candidate/difference/product slices start at zero, so the
        initial paired model reproduces the learned source-image path as closely
        as possible before paired fusion is trained.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        old_state = checkpoint.get("model", checkpoint)
        new_state = self.state_dict()
        compatible = {
            name: value
            for name, value in old_state.items()
            if name in new_state and new_state[name].shape == value.shape
        }
        self.load_state_dict(compatible, strict=False)

        widened = 0
        with torch.no_grad():
            for index, layer in enumerate(self.lateral):
                old_weight = old_state.get(f"lateral.{index}.weight")
                old_bias = old_state.get(f"lateral.{index}.bias")
                if old_weight is None or layer.weight.shape[1] != 4 * old_weight.shape[1]:
                    continue
                layer.weight.zero_()
                layer.weight[:, : old_weight.shape[1]].copy_(old_weight)
                if layer.bias is not None and old_bias is not None:
                    layer.bias.copy_(old_bias)
                widened += 1

            old_detail = old_state.get("detail.0.weight")
            if old_detail is not None and self.detail[0].weight.shape[1] == 3 * old_detail.shape[1]:
                self.detail[0].weight.zero_()
                self.detail[0].weight[:, : old_detail.shape[1]].copy_(old_detail)
                widened += 1

        return {"compatible_tensors": len(compatible), "widened_layers": widened}


if __name__ == "__main__":
    network = PairedWatermarkMaskNet(pretrained=False).eval()
    with torch.inference_mode():
        output = network(torch.randn(1, 3, 257, 385), torch.randn(1, 3, 129, 193))
    parameters = sum(parameter.numel() for parameter in network.parameters())
    assert output.shape == (1, 1, 257, 385)
    assert parameters < 5_000_000
    print(
        f"shape={tuple(output.shape)} parameters={parameters:,} "
        f"fp32={parameters * 4 / 2**20:.2f} MiB"
    )
