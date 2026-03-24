# -----------------------------------------------------------------------
# ESDNet for image demoireing — migrated from UHDM project
# Original paper: "Towards Efficient and Scale-Robust Ultra-High-Definition
#   Image Demoireing" (ECCV 2022)
#
# Modifications vs. original nets.py:
#   1. Added padder_size=8 to my_model (pixel_unshuffle×2 + 2× stride-2
#      = factor-8 total downsampling).
#   2. CSAF uses three independent squeeze layers (squeeze_0/2/4) instead
#      of one shared layer, fixing the original shared-pool issue.
#   3. F.sigmoid → torch.sigmoid (deprecation fix).
#   4. SAM and Decoder_Level use size= instead of scale_factor= for
#      upsampling to avoid shape mismatch on odd input sizes.
# -----------------------------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks (unchanged from original nets.py)
# ---------------------------------------------------------------------------

class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x):
        return self.conv(x)


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DB(nn.Module):
    """Dense Block with dilated convolutions."""
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for d in d_list:
            self.conv_layers.append(
                conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3,
                          dilation_rate=d, padding=d)
            )
            c += inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for layer in self.conv_layers:
            t = torch.cat([layer(t), t], dim=1)
        return self.conv_post(t)


class RDB(nn.Module):
    """Residual Dense Block."""
    def __init__(self, in_channel, d_list, inter_num):
        super(RDB, self).__init__()
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for d in d_list:
            self.conv_layers.append(
                conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3,
                          dilation_rate=d, padding=d)
            )
            c += inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for layer in self.conv_layers:
            t = torch.cat([layer(t), t], dim=1)
        return self.conv_post(t) + x


class CSAF(nn.Module):
    """Cross-Scale Attention Fusion.

    NOTE: Three SEPARATE squeeze layers (squeeze_0/2/4) instead of one shared
    layer to avoid potential cross-tile statistics inconsistency when the same
    pool layer is applied to tensors of different spatial sizes (x0/x2/x4).
    """
    def __init__(self, in_chnls, ratio=4):
        super(CSAF, self).__init__()
        # Three independent pooling ops — one per input scale
        self.squeeze_0  = nn.AdaptiveAvgPool2d(1)   # for x0 (full scale)
        self.squeeze_2  = nn.AdaptiveAvgPool2d(1)   # for x2 (1/2 scale)
        self.squeeze_4  = nn.AdaptiveAvgPool2d(1)   # for x4 (1/4 scale)
        self.compress1  = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2  = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        out0 = self.squeeze_0(x0)
        out2 = self.squeeze_2(x2)
        out4 = self.squeeze_4(x4)
        out  = torch.cat([out0, out2, out4], dim=1)
        out  = F.relu(self.compress1(out))
        out  = F.relu(self.compress2(out))
        out  = torch.sigmoid(self.excitation(out))
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        return x0 * w0 + x2 * w2 + x4 * w4


class SAM(nn.Module):
    """Scale-Aware Module — multi-scale feature fusion via CSAF."""
    def __init__(self, in_channel, d_list, inter_num):
        super(SAM, self).__init__()
        self.basic_block   = DB(in_channel, d_list, inter_num)
        self.basic_block_2 = DB(in_channel, d_list, inter_num)
        self.basic_block_4 = DB(in_channel, d_list, inter_num)
        self.fusion = CSAF(3 * in_channel)

    def forward(self, x):
        x_0 = x
        H, W = x.shape[2], x.shape[3]
        x_2 = F.interpolate(x,   scale_factor=0.5,  mode='bilinear', align_corners=False)
        x_4 = F.interpolate(x,   scale_factor=0.25, mode='bilinear', align_corners=False)
        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)
        # 用 size 而非 scale_factor 上采样，避免奇数尺寸时 floor 导致尺寸不匹配
        y_2 = F.interpolate(y_2, size=(H, W), mode='bilinear', align_corners=False)
        y_4 = F.interpolate(y_4, size=(H, W), mode='bilinear', align_corners=False)
        return x + self.fusion(y_0, y_2, y_4)


class Encoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number):
        super(Encoder_Level, self).__init__()
        self.rdb = RDB(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
        self.sam_blocks = nn.ModuleList(
            [SAM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
             for _ in range(sam_number)]
        )
        self.level = level
        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        out = self.rdb(x)
        for sam in self.sam_blocks:
            out = sam(out)
        if self.level < 3:
            return out, self.down(out)
        return out


class Decoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Decoder_Level, self).__init__()
        self.rdb = RDB(feature_num, (1, 2, 1), inter_num)
        self.sam_blocks = nn.ModuleList(
            [SAM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
             for _ in range(sam_number)]
        )
        self.conv = conv(in_channel=feature_num, out_channel=12, kernel_size=3, padding=1)

    def forward(self, x, feat=True):
        x = self.rdb(x)
        for sam in self.sam_blocks:
            x = sam(x)
        out = F.pixel_shuffle(self.conv(x), 2)
        if feat:
            H, W = out.shape[2], out.shape[3]
            feature = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            return out, feature
        return out


class Encoder(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Encoder, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.encoder_1 = Encoder_Level(feature_num,     inter_num, level=1, sam_number=sam_number)
        self.encoder_2 = Encoder_Level(2 * feature_num, inter_num, level=2, sam_number=sam_number)
        self.encoder_3 = Encoder_Level(4 * feature_num, inter_num, level=3, sam_number=sam_number)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)           # [B,3,H,W] → [B,12,H/2,W/2]
        x = self.conv_first(x)
        f1, d1 = self.encoder_1(x)
        f2, d2 = self.encoder_2(d1)
        f3      = self.encoder_3(d2)
        return f1, f2, f3


class Decoder(nn.Module):
    def __init__(self, en_num, feature_num, inter_num, sam_number):
        super(Decoder, self).__init__()
        self.preconv_3 = conv_relu(4 * en_num,             feature_num, 3, padding=1)
        self.decoder_3 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_2 = conv_relu(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_1 = conv_relu(en_num + feature_num,   feature_num, 3, padding=1)
        self.decoder_1 = Decoder_Level(feature_num, inter_num, sam_number)

    def forward(self, y_1, y_2, y_3):
        x3 = self.preconv_3(y_3)
        out_3, feat_3 = self.decoder_3(x3)

        x2 = self.preconv_2(torch.cat([y_2, feat_3], dim=1))
        out_2, feat_2 = self.decoder_2(x2)

        x1 = self.preconv_1(torch.cat([y_1, feat_2], dim=1))
        out_1 = self.decoder_1(x1, feat=False)

        return out_1, out_2, out_3


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class ESDNet(nn.Module):
    """
    ESDNet (my_model in original UHDM repo) with added check_image_size.

    Args:
        en_feature_num (int): Initial channel count for encoder dense blocks. Default: 48
        en_inter_num   (int): Growth rate for encoder dense blocks. Default: 32
        de_feature_num (int): Initial channel count for decoder dense blocks. Default: 64
        de_inter_num   (int): Growth rate for decoder dense blocks. Default: 32
        sam_number     (int): SAM blocks per encoder/decoder level (1=ESDNet, 2=ESDNet-L). Default: 1
    """
    # factor-8 total downsampling (pixel_unshuffle×2 + 2× stride-2 conv)
    padder_size = 8

    def __init__(self,
                 en_feature_num=48,
                 en_inter_num=32,
                 de_feature_num=64,
                 de_inter_num=32,
                 sam_number=1):
        super(ESDNet, self).__init__()
        self.encoder = Encoder(en_feature_num, en_inter_num, sam_number)
        self.decoder = Decoder(en_feature_num, de_feature_num, de_inter_num, sam_number)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.check_image_size(x)
        f1, f2, f3        = self.encoder(x)
        out_1, out_2, out_3 = self.decoder(f1, f2, f3)
        # crop back if padded
        return out_1[:, :, :H, :W], out_2, out_3

    def check_image_size(self, x):
        _, _, h, w = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        # 始终 pad（pad=0 时为无操作），避免 trace 时 if 分支导致 TracerWarning
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        return x
        return x


if __name__ == '__main__':
    # Quick sanity check
    net = ESDNet()
    x   = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        out1, out2, out3 = net(x)
    print('out_1:', out1.shape)   # [1, 3, 256, 256]
    print('out_2:', out2.shape)
    print('out_3:', out3.shape)
