import torch.nn as nn


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super().__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)
        self.conv2 = nn.Conv2d(
            expand_size,
            expand_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expand_size,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)
        self.skip = self._make_skip(in_size, out_size, stride)

    def _make_skip(self, in_size, out_size, stride):
        if stride == 1 and in_size == out_size:
            return nn.Identity()
        if stride == 1:
            return nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size),
            )
        if in_size == out_size:
            return nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_size),
            )
        return nn.Sequential(
            nn.Conv2d(in_size, in_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_size),
            nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_size),
        )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        return self.act3(out + self.skip(x))


class MobileNetV3_Large(nn.Module):
    def __init__(self, act=nn.Hardswish, downsample_factor=8):
        super().__init__()
        if downsample_factor not in (8, 16):
            raise ValueError("MobileNetV3 supports downsample_factor 8 or 16.")
        # Keep convolution and batch-normalization in Sequential containers.
        # This matches the keys used by the shipped MobileNetV3 checkpoint
        # (for example, ``backbone.conv1.0.weight``).
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            act(inplace=True),
        )
        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, False, 1),
            Block(3, 16, 64, 24, nn.ReLU, False, 2),
            Block(3, 24, 72, 24, nn.ReLU, False, 1),
            Block(5, 24, 72, 40, nn.ReLU, True, 2),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(3, 40, 240, 80, act, False, 2),
            Block(3, 80, 200, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 480, 112, act, True, 1),
            Block(3, 112, 672, 112, act, True, 1),
            Block(5, 112, 672, 160, act, True, 2),
            Block(5, 160, 672, 160, act, True, 1),
            Block(5, 160, 960, 160, act, True, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(960),
            act(inplace=True),
        )

        # Match Cloud_side_train/nets/deeplabv3_plus.py:MobileNetV3.
        # Stride/dilation are not in a state_dict: loading the checkpoint into
        # an unmodified classification backbone silently leaves output stride 32.
        if downsample_factor == 8:
            for index in range(6, 12):
                self.bneck[index].apply(lambda module: self._nostride_dilate(module, 2))
            for index in range(12, len(self.bneck)):
                self.bneck[index].apply(lambda module: self._nostride_dilate(module, 4))
        else:
            for index in range(12, len(self.bneck)):
                self.bneck[index].apply(lambda module: self._nostride_dilate(module, 2))

    @staticmethod
    def _nostride_dilate(module, dilate):
        if not isinstance(module, nn.Conv2d) or module.kernel_size == (1, 1):
            return
        # Apply to both the depthwise convolution and the residual projection,
        # including 5x5 kernels, so the two paths retain identical spatial sizes.
        if module.stride == (2, 2):
            module.stride = (1, 1)
            dilate = max(1, dilate // 2)
        module.dilation = (dilate, dilate)
        module.padding = tuple((size // 2) * dilate for size in module.kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        for index, block in enumerate(self.bneck):
            x = block(x)
            # Use the end of the 1/4-resolution, 24-channel stage.  There are
            # two 24-channel bottlenecks (indices 1 and 2); the decoder was
            # trained on the latter, stage-complete feature.
            if index == 2:
                low_level_features = x
        x = self.conv2(x)
        return low_level_features, x
