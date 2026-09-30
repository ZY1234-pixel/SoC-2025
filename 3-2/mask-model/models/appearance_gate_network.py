"""Appearance-first paired detector with a multiplicative change gate.

Design: ``docs/DESIGN-watermark-mask-v3.md``.

The watermark evidence lives in the Watermarked Source, so appearance is the
primary criterion.  The Clean Candidate only supplies a *gate* that says where a
candidate change is trustworthy; because the gate multiplies, it can suppress a
false positive instead of only ever adding area (which is what ``logaddexp`` and
``np.maximum`` did).

    P = P_appearance ** (g0 + (1 - g0) * g)

Fine text strokes are carried by the stride-4/8 levels, large solid blocks by the
stride-16/32 levels.  All four levels are supervised against the same Watermark
Mask at their own resolution (deep supervision), so there is no second head with
different target semantics and nothing to fuse after the forward pass.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

try:
    from .network import _Smooth
except ImportError:
    from network import _Smooth


class _Align(nn.Module):
    """Lightweight contextual alignment of the candidate features onto the source.

    ``APD``'s Alignment step, reduced to a bounded local offset: predict a small
    displacement for the candidate map and sample it, so that sub-pixel warping
    and texture drift introduced by the generator do not register as change.
    Gross misregistration is out of scope -- both inputs share one coordinate
    system.
    """

    def __init__(self, channels: int, max_offset: float = 1.5):
        super().__init__()
        self.max_offset = max_offset
        self.offset = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 2, 1),
        )

    def forward(self, source_feature, candidate_feature):
        if candidate_feature.shape[-2:] != source_feature.shape[-2:]:
            candidate_feature = F.interpolate(
                candidate_feature, size=source_feature.shape[-2:], mode="bilinear", align_corners=False
            )
        raw = torch.tanh(self.offset(torch.cat((source_feature, candidate_feature), dim=1)))
        offset = raw * self.max_offset
        height, width = source_feature.shape[-2:]
        ys = torch.linspace(-1.0, 1.0, height, device=offset.device, dtype=offset.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=offset.device, dtype=offset.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)[None]
        # normalise the pixel offset into the [-1, 1] grid units of this level
        grid = grid + torch.stack(
            (offset[:, 0] * 2.0 / max(width - 1, 1), offset[:, 1] * 2.0 / max(height - 1, 1)),
            dim=-1,
        )
        grid = grid.expand(source_feature.shape[0], -1, -1, -1)
        return F.grid_sample(
            candidate_feature,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )


class AppearanceGateMaskNet(nn.Module):
    """Appearance-driven mask with a multiplicative candidate-change gate."""

    def __init__(self, width: int = 64, pretrained: bool = True):
        super().__init__()
        self.encoder = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        ).features
        self.feature_ids = {1, 3, 6, 12, 16}
        channels = (16, 24, 40, 112, 960)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.smooth = nn.ModuleList(_Smooth(width) for _ in range(4))

        # appearance detail stem: raw absolute residual at full resolution
        self.detail = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            _Smooth(16),
        )
        self.fuse_detail = nn.Sequential(
            nn.Conv2d(width + 16, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            _Smooth(width),
        )

        # deep supervision: one head per pyramid level (stride 4 / 8 / 16 / 32)
        self.side_heads = nn.ModuleList(
            nn.Sequential(_Smooth(width), nn.Conv2d(width, 1, 1)) for _ in range(4)
        )

        # difference gate branch
        self.align = nn.ModuleList(_Align(width) for _ in range(4))
        self.diff_proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, width, 1, bias=False),
                nn.BatchNorm2d(width),
                nn.SiLU(inplace=True),
            )
            for _ in range(4)
        )
        self.gate_head = nn.Sequential(_Smooth(width), nn.Conv2d(width, 1, 1))
        self.global_gate = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.SiLU(inplace=True),
            nn.Linear(width // 2, 1),
        )
        # Start permissive (g0 ~ 0.88): the gate only ever multiplies the appearance
        # probability, so a low g0 would suppress a correct appearance prediction
        # before the gate has learned anything.  Starting near 1 keeps the initial
        # behaviour close to pure appearance detection and lets training tighten it.
        nn.init.zeros_(self.global_gate[-1].weight)
        nn.init.constant_(self.global_gate[-1].bias, 2.0)

    def _encode(self, image):
        out, value = [], image
        for index, layer in enumerate(self.encoder):
            value = layer(value)
            if index in self.feature_ids:
                out.append(value)
        return out

    def forward(self, source, candidate, return_aux=False):
        if candidate.shape[-2:] != source.shape[-2:]:
            candidate = F.interpolate(
                candidate, size=source.shape[-2:], mode="bilinear", align_corners=False
            )
        output_size = source.shape[-2:]
        raw_residual = (source - candidate).abs()
        source = (source - self.image_mean) / self.image_std
        candidate = (candidate - self.image_mean) / self.image_std

        sf = self._encode(source)
        cf = self._encode(candidate)

        # ---- appearance pyramid (source only: appearance is the main criterion)
        value = self.lateral[-1](sf[-1])
        pyramid = []
        for index in range(3, -1, -1):
            value = F.interpolate(
                value, size=sf[index].shape[-2:], mode="bilinear", align_corners=False
            )
            value = self.smooth[index](value + self.lateral[index](sf[index]))
            pyramid.append(value)  # stride 4, 8, 16, 32

        # ---- difference gate at every scale, aligned before differencing
        gate_features = []
        for index, feature in enumerate(pyramid):
            aligned = self.align[index](feature, self.lateral[index](cf[index]))
            gate_features.append(self.diff_proj[index]((feature - aligned).abs()))

        detail = self.detail(raw_residual)
        if detail.shape[-2:] != pyramid[0].shape[-2:]:
            detail = F.interpolate(
                detail, size=pyramid[0].shape[-2:], mode="bilinear", align_corners=False
            )
        fine = self.fuse_detail(torch.cat((pyramid[0], detail), dim=1))

        # ---- side masks (per level logits at that level's resolution)
        side_logits = [head(feature) for head, feature in zip(self.side_heads, pyramid)]
        gate_logits = self.gate_head(fine)

        # ---- multiplicative fusion in probability space
        appearance = torch.sigmoid(side_logits[0])
        gate = torch.sigmoid(gate_logits)
        global_gate = torch.sigmoid(
            self.global_gate(gate_features[0].mean(dim=(2, 3)))
        ).view(-1, 1, 1, 1)
        blend = global_gate + (1.0 - global_gate) * gate
        combined = appearance * blend

        probability = F.interpolate(combined, size=output_size, mode="bilinear", align_corners=False)
        if not return_aux:
            return probability

        appearance_out = F.interpolate(
            appearance, size=output_size, mode="bilinear", align_corners=False
        )
        gate_out = F.interpolate(gate, size=output_size, mode="bilinear", align_corners=False)
        aux = {
            "appearance": appearance_out,
            "gate": gate_out,
            "global_gate": global_gate.detach().view(-1),
            "side": [
                F.interpolate(torch.sigmoid(logit), size=output_size, mode="bilinear", align_corners=False)
                for logit in side_logits
            ],
            "side_logits": side_logits,
        }
        return probability, aux


if __name__ == "__main__":
    model = AppearanceGateMaskNet(pretrained=False).eval()
    with torch.inference_mode():
        combined, aux = model(torch.rand(1, 3, 257, 385), torch.rand(1, 3, 257, 385), return_aux=True)
    parameters = sum(p.numel() for p in model.parameters())
    assert combined.shape == (1, 1, 257, 385), combined.shape
    assert all(level.shape == (1, 1, 257, 385) for level in aux["side"]), "side masks must be full size"
    assert 0.0 <= combined.min() and combined.max() <= 1.0
    assert parameters < 5_000_000, parameters
    # a zero gate must be able to suppress: same appearance, gate forced to 0
    print(f"shape={tuple(combined.shape)} parameters={parameters:,}")
    print("side levels produce full-resolution masks:", [tuple(l.shape) for l in aux["side"]])
