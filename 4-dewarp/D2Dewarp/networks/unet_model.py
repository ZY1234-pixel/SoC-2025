from networks.cross_attn import SelfAttention
from networks.unet_parts import *


class UNetLineSeg(nn.Module):
    def __init__(self, n_channels, n_classes, d_model, bilinear=True):
        super(UNetLineSeg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.d_model = d_model
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 196)
        # factor = 2 if bilinear else 1
        self.down4 = Down(196, self.d_model)

        n_head = 8
        d_k = self.d_model // n_head
        d_v = self.d_model // n_head

        self.self_attention = SelfAttention(n_layers=4,
                                            n_head=n_head,
                                            d_k=d_k,
                                            d_v=d_v,
                                            d_model=self.d_model,
                                            n_position=896,
                                            d_inner=d_model * 4)

        self.up1 = Up(self.d_model + 196, 196, bilinear)
        self.up2 = Up(324, 128, bilinear)
        self.up3 = Up(192, 64, bilinear)
        self.up4 = Up(96, 48, bilinear)
        self.outc1 = OutConv(196, n_classes)
        self.outc2 = OutConv(128, n_classes)
        self.outc3 = OutConv(64, n_classes)
        self.outc4 = OutConv(48, n_classes)
        self.outc = OutConv(4, n_classes)

        self.up11 = Up(self.d_model + 196, 196, bilinear)
        self.up22 = Up(324, 128, bilinear)
        self.up33 = Up(192, 64, bilinear)
        self.up44 = Up(96, 48, bilinear)
        self.outc11 = OutConv(196, n_classes)
        self.outc22 = OutConv(128, n_classes)
        self.outc33 = OutConv(64, n_classes)
        self.outc44 = OutConv(48, n_classes)
        self.outcc = OutConv(4, n_classes)

        # self.htan = nn.Hardtanh(0., 1.0)

        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        n, c, H, W = x.shape
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        n, c, h, w = x5.size()
        x5 = self.self_attention(x5)
        x5 = x5.transpose(1, 2).contiguous().view(n, c, h, w)

        x = self.up1(x5, x4)
        side_1 = F.interpolate(x, size=(H // 8, W // 8), mode='bilinear', align_corners=False)
        side_out1_ = self.outc1(x)
        side_out1 = F.interpolate(side_out1_, size=(H, W), mode='bilinear', align_corners=False)

        x = self.up2(x, x3)
        side_2 = F.interpolate(x, size=(H // 8, W // 8), mode='bilinear', align_corners=False)
        side_out2_ = self.outc2(x)
        side_out2 = F.interpolate(side_out2_, size=(H, W), mode='bilinear', align_corners=False)

        x = self.up3(x, x2)
        side_3 = F.interpolate(x, size=(H // 8, W // 8), mode='bilinear', align_corners=False)
        side_out3_ = self.outc3(x)
        side_out3 = F.interpolate(side_out3_, size=(H, W), mode='bilinear', align_corners=False)

        x = self.up4(x, x1)
        side_4 = F.interpolate(x, size=(H // 8, W // 8), mode='bilinear', align_corners=False)
        side_out4_ = self.outc4(x)
        side_out4 = F.interpolate(side_out4_, size=(H, W), mode='bilinear', align_corners=False)

        # ---
        logits1 = self.outc(torch.cat((side_out1, side_out2, side_out3, side_out4), dim=1))

        x_ = self.up11(x5, x4)
        side_11 = F.interpolate(x_, size=(H // 8, W // 8), mode='bilinear', align_corners=False)
        side_out11_ = self.outc11(x_)
        side_out11 = F.interpolate(side_out11_, size=(H, W), mode='bilinear', align_corners=False)

        x_ = self.up22(x_, x3)
        side_22 = F.interpolate(x_, size=(H // 8, W // 8), mode='bilinear', align_corners=False)
        side_out22_ = self.outc22(x_)
        side_out22 = F.interpolate(side_out22_, size=(H, W), mode='bilinear', align_corners=False)

        x_ = self.up33(x_, x2)
        side_33 = F.interpolate(x_, size=(H // 8, W // 8), mode='bilinear', align_corners=False)
        side_out33_ = self.outc33(x_)
        side_out33 = F.interpolate(side_out33_, size=(H, W), mode='bilinear', align_corners=False)

        x_ = self.up44(x_, x1)
        side_44 = F.interpolate(x_, size=(H // 8, W // 8), mode='bilinear', align_corners=False)
        side_out44_ = self.outc44(x_)
        side_out44 = F.interpolate(side_out44_, size=(H, W), mode='bilinear', align_corners=False)

        logits2 = self.outcc(torch.cat((side_out11, side_out22, side_out33, side_out44), dim=1))

        h_map = torch.cat((side_1, side_2, side_3, side_4), dim=1)
        v_map = torch.cat((side_11, side_22, side_33, side_44), dim=1)
        #
        # h_map = self.cam_1(self.conv3x3_1(h_map))
        # v_map = self.cam_2(self.conv3x3_2(v_map))

        return (side_out1, side_out2, side_out3, side_out4, logits1), \
               (side_out11, side_out22, side_out33, side_out44, logits2), h_map, v_map


if __name__ == '__main__':
    x1 = torch.rand((2, 3, 448, 448)).cuda()
    net = UNetLineSeg(n_channels=3, n_classes=1, d_model=448).cuda()
    print(net)
    logits1, logits2 = net(x1)
    n_p = sum(x.numel() for x in net.parameters())  # number parameters
    n_g = sum(x.numel() for x in net.parameters() if x.requires_grad)  # number gradients
    print('Model Summary: %g parameters, %g gradients\n' % (n_p, n_g))

    print("logits1: ", logits1[-1].shape)
    print("logits2: ", logits2[-1].shape)
