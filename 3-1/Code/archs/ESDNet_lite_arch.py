# -----------------------------------------------------------------------
# ESDNet-Lite: Lightweight ESDNet for Android NCNN deployment
#
# Target: ≥20× FLOPs reduction vs ESDNet (13.52G → <0.676G)
#
# Key changes vs ESDNet:
#   1. Channel reduction: en=16, inter=8, de=20, de_inter=8
#   2. SAM → LightSAM: single-scale DB + SE channel attention
#      (removes multi-scale branches → eliminates dynamic interpolate)
#   3. Dense block simplification: d_list=(1,2,1) for LightSAM,
#      (1,1) for RDB; max dilation=2 (ARM NEON friendly)
#   4. Depthwise separable conv in dense blocks for extra savings
#   5. NCNN-friendly ops: replicate padding, no dynamic-size interpolate
#   6. Single output (out_1 only) for inference; multi-output for training
# -----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Conv(nn.Module):
    """Plain convolution."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, bias=True)

    def forward(self, x):
        return self.conv(x)


class DWConvReLU(nn.Module):
    """Depthwise-separable convolution + ReLU.

    DW 3×3 (groups=in_ch, dilation) → PW 1×1 (in_ch → out_ch) → ReLU.
    When in_ch is small and differs from out_ch significantly, falls back
    to standard conv (depthwise not beneficial for very small groups).
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, padding=1):
        super().__init__()
        # Use depthwise-separable only when in_ch >= 8 (worthwhile savings)
        if in_ch >= 8 and kernel_size == 3:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding,
                          dilation=dilation, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, bias=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding,
                          dilation=dilation, bias=True),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class ConvReLU(nn.Module):
    """Standard convolution + ReLU."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Lightweight Dense Block (LDB) — uses DW-Sep conv, fewer layers
# ---------------------------------------------------------------------------

class LDB(nn.Module):
    """Lightweight Dense Block with depthwise-separable convolutions.

    Compared to original DB: fewer dilated layers, DW-Sep conv, max dilation=2.
    """
    def __init__(self, in_ch, d_list, inter_num):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        c = in_ch
        for d in d_list:
            self.conv_layers.append(
                DWConvReLU(c, inter_num, kernel_size=3, dilation=d, padding=d)
            )
            c += inter_num
        self.conv_post = Conv(c, in_ch, kernel_size=1)

    def forward(self, x):
        t = x
        for layer in self.conv_layers:
            t = torch.cat([layer(t), t], dim=1)
        return self.conv_post(t)


class LRDB(nn.Module):
    """Lightweight Residual Dense Block."""
    def __init__(self, in_ch, d_list, inter_num):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        c = in_ch
        for d in d_list:
            self.conv_layers.append(
                DWConvReLU(c, inter_num, kernel_size=3, dilation=d, padding=d)
            )
            c += inter_num
        self.conv_post = Conv(c, in_ch, kernel_size=1)

    def forward(self, x):
        t = x
        for layer in self.conv_layers:
            t = torch.cat([layer(t), t], dim=1)
        return self.conv_post(t) + x


# ---------------------------------------------------------------------------
# Lightweight Scale-Aware Module (LightSAM)
# Single-scale DB + SE channel attention (no multi-scale interpolation)
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class LightSAM(nn.Module):
    """Lightweight Scale-Aware Module.

    Replaces the original 3-branch multi-scale SAM with a single-scale
    dense block + SE attention.  This eliminates:
      - 2/3 of DB computation (0.5× and 0.25× branches removed)
      - All dynamic-size F.interpolate calls (NCNN-unfriendly)
      - CSAF module (replaced by simpler SE)
    """
    def __init__(self, in_ch, d_list, inter_num):
        super().__init__()
        self.block = LDB(in_ch, d_list, inter_num)
        self.se = SEBlock(in_ch)

    def forward(self, x):
        return x + self.se(self.block(x))


# ---------------------------------------------------------------------------
# Encoder / Decoder levels
# ---------------------------------------------------------------------------

class EncoderLevel(nn.Module):
    def __init__(self, feat_ch, inter_num, level, sam_number):
        super().__init__()
        self.rdb = LRDB(feat_ch, d_list=(1, 1), inter_num=inter_num)
        self.sam_blocks = nn.ModuleList(
            [LightSAM(feat_ch, d_list=(1, 2, 1), inter_num=inter_num)
             for _ in range(sam_number)]
        )
        self.level = level
        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feat_ch, 2 * feat_ch, 3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        out = self.rdb(x)
        for sam in self.sam_blocks:
            out = sam(out)
        if self.level < 3:
            return out, self.down(out)
        return out


class DecoderLevel(nn.Module):
    def __init__(self, feat_ch, inter_num, sam_number):
        super().__init__()
        self.rdb = LRDB(feat_ch, d_list=(1, 1), inter_num=inter_num)
        self.sam_blocks = nn.ModuleList(
            [LightSAM(feat_ch, d_list=(1, 2, 1), inter_num=inter_num)
             for _ in range(sam_number)]
        )
        self.conv = Conv(feat_ch, 12, kernel_size=3, padding=1)

    def forward(self, x, feat=True):
        x = self.rdb(x)
        for sam in self.sam_blocks:
            x = sam(x)
        out = F.pixel_shuffle(self.conv(x), 2)
        if feat:
            feature = F.interpolate(x, scale_factor=2, mode='bilinear',
                                    align_corners=False)
            return out, feature
        return out


class Encoder(nn.Module):
    def __init__(self, feat_ch, inter_num, sam_number):
        super().__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feat_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.enc1 = EncoderLevel(feat_ch, inter_num, level=1, sam_number=sam_number)
        self.enc2 = EncoderLevel(2 * feat_ch, inter_num, level=2, sam_number=sam_number)
        self.enc3 = EncoderLevel(4 * feat_ch, inter_num, level=3, sam_number=sam_number)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)  # [B,3,H,W] → [B,12,H/2,W/2]
        x = self.conv_first(x)
        f1, d1 = self.enc1(x)
        f2, d2 = self.enc2(d1)
        f3 = self.enc3(d2)
        return f1, f2, f3


class Decoder(nn.Module):
    def __init__(self, en_ch, feat_ch, inter_num, sam_number):
        super().__init__()
        self.preconv_3 = ConvReLU(4 * en_ch, feat_ch, 3, padding=1)
        self.decoder_3 = DecoderLevel(feat_ch, inter_num, sam_number)

        self.preconv_2 = ConvReLU(2 * en_ch + feat_ch, feat_ch, 3, padding=1)
        self.decoder_2 = DecoderLevel(feat_ch, inter_num, sam_number)

        self.preconv_1 = ConvReLU(en_ch + feat_ch, feat_ch, 3, padding=1)
        self.decoder_1 = DecoderLevel(feat_ch, inter_num, sam_number)

    def forward(self, y1, y2, y3):
        x3 = self.preconv_3(y3)
        out_3, feat_3 = self.decoder_3(x3)

        x2 = self.preconv_2(torch.cat([y2, feat_3], dim=1))
        out_2, feat_2 = self.decoder_2(x2)

        x1 = self.preconv_1(torch.cat([y1, feat_2], dim=1))
        out_1 = self.decoder_1(x1, feat=False)

        return out_1, out_2, out_3


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class ESDNetLite(nn.Module):
    """
    ESDNet-Lite: Lightweight variant targeting ≥20× FLOPs reduction for
    Android NCNN deployment.

    Default params (Lite):
        en_feature_num=16, en_inter_num=8,
        de_feature_num=20, de_inter_num=8, sam_number=1

    vs. original ESDNet:
        en_feature_num=48, en_inter_num=32,
        de_feature_num=64, de_inter_num=32, sam_number=1
    """
    padder_size = 8  # pixel_unshuffle(2) + 2× stride-2

    def __init__(self,
                 en_feature_num=16,
                 en_inter_num=8,
                 de_feature_num=20,
                 de_inter_num=8,
                 sam_number=1):
        super().__init__()
        self.encoder = Encoder(en_feature_num, en_inter_num, sam_number)
        self.decoder = Decoder(en_feature_num, de_feature_num, de_inter_num,
                               sam_number)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.check_image_size(x)
        f1, f2, f3 = self.encoder(x)
        out_1, out_2, out_3 = self.decoder(f1, f2, f3)
        return out_1[:, :, :H, :W], out_2, out_3

    def check_image_size(self, x):
        _, _, h, w = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        return x


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    net = ESDNetLite()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        out1, out2, out3 = net(x)
    print('out_1:', out1.shape)
    print('out_2:', out2.shape)
    print('out_3:', out3.shape)

    # Parameter count
    total = sum(p.numel() for p in net.parameters())
    print(f'Parameters: {total / 1e6:.3f} M')

    # FLOPs (if ptflops available)
    try:
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(
            net, (3, 256, 256), as_strings=True,
            print_per_layer_stat=False, verbose=False)
        print(f'MACs (256×256): {macs}')
        print(f'Params: {params}')
    except ImportError:
        print('Install ptflops for FLOPs count: pip install ptflops')
