import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn

from thop import profile, clever_format
from model_utils.dewarp_utils import DewarpUP
from model_utils.utils_model import CAM_Module
from networks.cross_attn import SelfAttention
from networks.unet_model import UNetLineSeg


class Attention(nn.Module):
    def __init__(self, img_size, d_model, scale=8):
        super(Attention, self).__init__()
        self.self_atten = SelfAttention(n_layers=4,
                                        n_head=8,
                                        d_k=d_model // 8,
                                        d_v=d_model // 8,
                                        d_model=d_model,
                                        n_position=(img_size // scale) * (img_size // scale),
                                        d_inner=d_model * 4)

    def forward(self, x):
        n, c, h, w = x.shape
        x = self.self_atten(x)
        x = x.transpose(1, 2).contiguous().view(n, c, h, w)
        return x


class CoordAtt(nn.Module):
    def __init__(self, img_size, d_model, scale=8):
        super(CoordAtt, self).__init__()
        self.pool_h_x = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_h_y = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_x = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w_y = nn.AdaptiveAvgPool2d((1, None))

        self.self_atten1 = Attention(img_size=img_size, d_model=d_model, scale=scale)
        self.self_atten2 = Attention(img_size=img_size, d_model=d_model, scale=scale)
        self.self_atten3 = Attention(img_size=img_size, d_model=d_model, scale=scale)
        self.self_atten4 = Attention(img_size=img_size, d_model=d_model, scale=scale)

        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)

        self.conv5 = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.conv7 = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.conv8 = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):
        identity_x = x
        identity_y = y
        n, c, h, w = x.size()
        # pool
        x_h = self.pool_h_x(x)
        x_w = self.pool_w_x(x).permute(0, 1, 3, 2)

        y_h = self.pool_h_y(y)
        y_w = self.pool_w_y(y).permute(0, 1, 3, 2)

        # cat + conv + split

        x = torch.cat([x_h, y_w], dim=2)  # x_h->[B, C, H, 1] y_w->[B, C, 1, W] -> [B, C, W, 1]
        x = self.self_atten1(x)
        x_h, y_w = torch.split(x, [h, w], dim=2)

        y_w = y_w.permute(0, 1, 3, 2)  # [B, C, 1, W]
        y = torch.cat([y_h, x_w], dim=2)
        y = self.self_atten2(y)
        y_h, x_w = torch.split(y, [h, w], dim=2)

        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv1(x_h)  # .sigmoid()
        x_w = self.conv2(x_w)  # .sigmoid()
        y_h = self.conv3(y_h)  # .sigmoid()
        y_w = self.conv4(y_w)  # .sigmoid()

        x = self.self_atten3(torch.cat([x_h, y_h], dim=2))  # x_h->[B, C, H, 1] y_h->[B, C, H, 1]
        x_h, y_h = torch.split(x, [h, h], dim=2)

        y = self.self_atten4(torch.cat([x_w, y_w], dim=3))
        x_w, y_w = torch.split(y, [w, w], dim=3)

        x_h = self.conv5(x_h).sigmoid()
        x_w = self.conv6(x_w).sigmoid()
        y_h = self.conv7(y_h).sigmoid()
        y_w = self.conv8(y_w).sigmoid()

        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y_h = y_h.expand(-1, -1, h, w)
        y_w = y_w.expand(-1, -1, h, w)

        x = identity_x * x_w * x_h
        y = identity_y * y_w * y_h

        return x, y


class D2DewarpModel(nn.Module):
    """ D2DewarpModel
    """

    def __init__(self, img_size=448, in_chans=4, hv_out_chans=1, d_model=256):
        super().__init__()
        self.img_size = img_size
        self.d_model = d_model
        self.scale = 8
        self.unet_2decoder = UNetLineSeg(n_channels=in_chans, n_classes=hv_out_chans, d_model=d_model)
        self.fusion_block = CoordAtt(d_model, d_model, scale=self.scale)

        self.cam_1 = CAM_Module()
        self.cam_2 = CAM_Module()

        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels=196 + 128 + 64 + 48, out_channels=self.d_model,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True)
        )

        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels=196 + 128 + 64 + 48, out_channels=self.d_model,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True)
        )

        self.dewarp_up = DewarpUP(scale=self.scale, hidden_dim=d_model * 2)

    def forward(self, x):
        pred_h_lst, pred_v_lst, h_map, v_map = self.unet_2decoder(x)

        h_map = self.cam_1(self.conv3x3_1(h_map))
        v_map = self.cam_2(self.conv3x3_2(v_map))
        n, c, h, w = h_map.shape
        h_map, v_map = self.fusion_block(v_map, h_map)

        union_map = torch.cat((h_map, v_map), dim=1)

        bm_up = self.dewarp_up(x, x.shape[2], x.shape[3], union_map)
        return pred_h_lst, pred_v_lst, bm_up


if __name__ == '__main__':
    x = torch.randn((2, 4, 448, 448)).cuda()
    y = torch.randn((2, 320, 56, 56)).cuda()
    net = D2DewarpModel(img_size=448, in_chans=4, hv_out_chans=1, d_model=448).cuda()

    print(net)

    pred_h_lst, pred_v_lst, bm_up = net(x)
    n_p = sum(x.numel() for x in net.parameters())  # number parameters
    n_g = sum(x.numel() for x in net.parameters() if x.requires_grad)  # number gradients
    print('Model Summary: %g parameters, %g gradients\n' % (n_p, n_g))

    flops, params = profile(net, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" % (flops))
    print("params: %s" % (params))

    print("pred_h_lst[-1]: ", pred_h_lst[-1].shape)
    print("pred_v_lst[-1]: ", pred_v_lst[-1].shape)
    print("bm_up: ", bm_up.shape)
