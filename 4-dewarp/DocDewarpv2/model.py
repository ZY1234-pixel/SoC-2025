import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.fusion import ParallelAttentionFusion


def conv3x3(in_channels, out_channels, kernel_size, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def dilated_conv_bn_act(in_channels, out_channels, act_fn, BatchNorm, dilation):
    model = nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            bias=False,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        ),
        BatchNorm(out_channels),
        act_fn,
    )
    return model


def dilated_conv(in_channels, out_channels, kernel_size, dilation, stride=1):
    model = nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
        )
    )
    return model


class ResidualBlockWithDilation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        BatchNorm,
        kernel_size,
        stride=1,
        downsample=None,
        is_activation=True,
        is_top=False,
    ):
        super(ResidualBlockWithDilation, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.is_activation = is_activation
        self.is_top = is_top
        if self.stride != 1 or self.is_top:
            self.conv1 = conv3x3(in_channels, out_channels, kernel_size, self.stride)
            self.conv2 = conv3x3(out_channels, out_channels, kernel_size)
        else:
            self.conv1 = dilated_conv(in_channels, out_channels, kernel_size, dilation=3)
            self.conv2 = dilated_conv(out_channels, out_channels, kernel_size, dilation=3)

        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = BatchNorm(out_channels)

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))

        out2 += residual
        out = self.relu(out2)
        return out


class ResnetStraight(nn.Module):
    def __init__(
        self,
        num_filter,
        map_num,
        BatchNorm,
        block_nums=[3, 4, 6, 3],
        block=ResidualBlockWithDilation,
        kernel_size=5,
        stride=[1, 1, 2, 2],
    ):
        super(ResnetStraight, self).__init__()
        self.in_channels = num_filter * map_num[0]
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        self.block_nums = block_nums
        self.kernel_size = kernel_size

        self.layer1 = self.blocklayer(
            block,
            num_filter * map_num[0],
            self.block_nums[0],
            BatchNorm,
            kernel_size=self.kernel_size,
            stride=self.stride[0],
        )
        self.layer2 = self.blocklayer(
            block,
            num_filter * map_num[1],
            self.block_nums[1],
            BatchNorm,
            kernel_size=self.kernel_size,
            stride=self.stride[1],
        )
        self.layer3 = self.blocklayer(
            block,
            num_filter * map_num[2],
            self.block_nums[2],
            BatchNorm,
            kernel_size=self.kernel_size,
            stride=self.stride[2],
        )

    def blocklayer(self, block, out_channels, block_nums, BatchNorm, kernel_size, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(
                    self.in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                BatchNorm(out_channels),
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                BatchNorm,
                kernel_size,
                stride,
                downsample,
                is_top=True,
            )
        )
        self.in_channels = out_channels
        for i in range(1, block_nums):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    BatchNorm,
                    kernel_size,
                    is_activation=True,
                    is_top=False,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return out1, out2, out3


class UVDocnet(nn.Module):
    def __init__(self, num_filter, kernel_size=5):
        super(UVDocnet, self).__init__()
        self.num_filter = num_filter
        self.in_channels = 3
        self.kernel_size = kernel_size
        self.stride = [1, 2, 2, 2]

        BatchNorm = nn.BatchNorm2d
        act_fn = nn.ReLU(inplace=True)
        map_num = [1, 2, 4, 8, 16]

        self.resnet_head = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.num_filter * map_num[0],
                bias=False,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.kernel_size // 2,
            ),
            BatchNorm(self.num_filter * map_num[0]),
            act_fn,
            nn.Conv2d(
                self.num_filter * map_num[0],
                self.num_filter * map_num[0],
                bias=False,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.kernel_size // 2,
            ),
            BatchNorm(self.num_filter * map_num[0]),
            act_fn,
        )

        self.resnet_down = ResnetStraight(
            self.num_filter,
            map_num,
            BatchNorm,
            block_nums=[3, 4, 6, 3],
            block=ResidualBlockWithDilation,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )

        map_num_i = 2
        self.bridge_1 = nn.Sequential(
            dilated_conv_bn_act(
                self.num_filter * map_num[map_num_i],
                self.num_filter * map_num[map_num_i],
                act_fn,
                BatchNorm,
                dilation=1,
            )
        )

        self.bridge_2 = nn.Sequential(
            dilated_conv_bn_act(
                self.num_filter * map_num[map_num_i],
                self.num_filter * map_num[map_num_i],
                act_fn,
                BatchNorm,
                dilation=2,
            )
        )

        self.bridge_3 = nn.Sequential(
            dilated_conv_bn_act(
                self.num_filter * map_num[map_num_i],
                self.num_filter * map_num[map_num_i],
                act_fn,
                BatchNorm,
                dilation=5,
            )
        )

        self.bridge_4 = nn.Sequential(
            *[
                dilated_conv_bn_act(
                    self.num_filter * map_num[map_num_i],
                    self.num_filter * map_num[map_num_i],
                    act_fn,
                    BatchNorm,
                    dilation=d,
                )
                for d in [8, 3, 2]
            ]
        )

        self.bridge_5 = nn.Sequential(
            *[
                dilated_conv_bn_act(
                    self.num_filter * map_num[map_num_i],
                    self.num_filter * map_num[map_num_i],
                    act_fn,
                    BatchNorm,
                    dilation=d,
                )
                for d in [12, 7, 4]
            ]
        )

        self.bridge_6 = nn.Sequential(
            *[
                dilated_conv_bn_act(
                    self.num_filter * map_num[map_num_i],
                    self.num_filter * map_num[map_num_i],
                    act_fn,
                    BatchNorm,
                    dilation=d,
                )
                for d in [18, 12, 6]
            ]
        )

        self.bridge_concat = nn.Sequential(
            nn.Conv2d(
                self.num_filter * map_num[map_num_i] * 6,
                self.num_filter * map_num[2],
                bias=False,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BatchNorm(self.num_filter * map_num[2]),
            act_fn,
        )

        # 并行 CBAM + RCCA：作用在 resnet_down 的 out3 (B,128,45,31) 上。
        # 新模块参数不在 model/best_model.pkl 中，训练时 strict=False 加载
        # 预训练权重后随网络一起微调。
        self.parallel_attention = ParallelAttentionFusion(
            channels=self.num_filter * map_num[2],
            cbam_reduction=16,
            cbam_kernel_size=7,
            rcca_recurrence=2,
        )

        self.out_point_positions2D = nn.Sequential(
            nn.Conv2d(
                self.num_filter * map_num[2],
                self.num_filter * map_num[0],
                bias=False,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
            ),
            BatchNorm(self.num_filter * map_num[0]),
            nn.PReLU(),
            nn.Conv2d(
                self.num_filter * map_num[0],
                2,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
            ),
        )

        self.out_point_positions3D = nn.Sequential(
            nn.Conv2d(
                self.num_filter * map_num[2],
                self.num_filter * map_num[0],
                bias=False,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
            ),
            BatchNorm(self.num_filter * map_num[0]),
            nn.PReLU(),
            nn.Conv2d(
                self.num_filter * map_num[0],
                3,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
            ),
        )

        # Multi-scale auxiliary 2D-grid heads: each predicts a (B,2,h,w) grid from a
        # different decoder/down-sampling stage, so the cross-consistency loss can be
        # enforced at several resolutions (see train_add_crossloss.multi_scale_cross_loss).
        # out1 (after resnet_down.layer1) is at 1/4 input resolution,
        # out2 (after resnet_down.layer2) is at 1/8 input resolution.
        self.aux_head_s4 = nn.Sequential(
            nn.Conv2d(
                self.num_filter * map_num[0],
                2,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
                bias=False,
            ),
        )
        self.aux_head_s8 = nn.Sequential(
            nn.Conv2d(
                self.num_filter * map_num[1],
                2,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
                bias=False,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.2)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                nn.init.xavier_normal_(m.weight, gain=0.2)

    def forward(self, x, return_aux=False):
        resnet_head = self.resnet_head(x)
        out1, out2, out3 = self.resnet_down(resnet_head)
        # 并行 CBAM + RCCA：作用在 encoder 的 out3 上，增强后的特征再送入
        # 6 条空洞金字塔分支，而不是在 bridge_concat 之后。
        out3 = self.parallel_attention(out3)
        bridge_1 = self.bridge_1(out3)
        bridge_2 = self.bridge_2(out3)
        bridge_3 = self.bridge_3(out3)
        bridge_4 = self.bridge_4(out3)
        bridge_5 = self.bridge_5(out3)
        bridge_6 = self.bridge_6(out3)
        bridge_concat = torch.cat([bridge_1, bridge_2, bridge_3, bridge_4, bridge_5, bridge_6], dim=1)
        bridge = self.bridge_concat(bridge_concat)

        out_point_positions2D = self.out_point_positions2D(bridge)
        out_point_positions3D = self.out_point_positions3D(bridge)

        if return_aux:
            # Multi-scale 2D-grid predictions from intermediate decoder stages.
            # aux_s4: 1/4 input resolution, aux_s8: 1/8, the main head: 1/16.
            # These are the real network outputs at different down-sampling scales,
            # used to apply the cross-consistency loss separately at each scale.
            aux_2d_s4 = self.aux_head_s4(out1)
            aux_2d_s8 = self.aux_head_s8(out2)
            aux2d_list = [aux_2d_s4, aux_2d_s8, out_point_positions2D]
            return out_point_positions2D, out_point_positions3D, aux2d_list

        return out_point_positions2D, out_point_positions3D


# -----------------------------------------------------------------------------#
#   自适应迭代细化模块（参考论文/专利 ArbDR：F-Net + 轴对齐线熵自适应停止）
#
#   设计要点（与论文/专利一致）：
#   - base_net (C-Net)：原 UVDocnet，预测初始后向形变场 D_0（输出空间 -> 输入图像）。
#   - refine_net (F-Net)：与 base_net 同构，接收"当前已矫正图像" O_m，预测残差后向
#     形变场 G_{m+1}（下一输出空间 -> O_m 空间）。
#   - 复合算子 compose(D_m, G) = grid_sample(D_m, G)：以 G 的坐标为索引在 D_m 上做双线性
#     采样，把多次非线性映射以链式方式累积到"输出 -> 输入"的后向映射上。最终只对原始
#     输入图做一次重采样（grid_sample），最大限度减少插值模糊，保证高保真。
#   - 训练时用可微的 axis_alignment_loss（见 utils）作为正则项；推理时用非可微的
#     axis_aligned_line_entropy（见 utils）做自适应停止，迭代次数无需人工设定。
# -----------------------------------------------------------------------------#
ITER_IMG_SIZE = (488, 712)  # (W, H)，与 utils.IMG_SIZE 保持一致


def _unwarp(warped_img, point_positions, img_size):
    """模型内部使用的双线性反变形（等价于 utils.bilinear_unwarping，避免循环依赖）。"""
    upsampled_grid = F.interpolate(
        point_positions, size=(img_size[1], img_size[0]), mode="bilinear", align_corners=True
    )
    unwarped_img = F.grid_sample(
        warped_img, upsampled_grid.transpose(1, 2).transpose(2, 3), align_corners=True
    )
    return unwarped_img


class UVDocIterative(nn.Module):
    """UVDoc + 自适应迭代细化（ArbDR 风格）。"""

    def __init__(self, num_filter=32, kernel_size=5):
        super(UVDocIterative, self).__init__()
        self.base_net = UVDocnet(num_filter=num_filter, kernel_size=kernel_size)  # C-Net
        self.refine_net = UVDocnet(num_filter=num_filter, kernel_size=kernel_size)  # F-Net

    def initial_grid(self, x):
        """C-Net：输入原图，输出初始 2D/3D 网格（D_0）。"""
        g2d, g3d = self.base_net(x)
        return g2d, g3d

    def refine_grid(self, O):
        """F-Net：输入当前矫正图 O_m，输出残差后向形变场 G_{m+1}。"""
        g2d, _ = self.refine_net(O)
        return g2d

    @staticmethod
    def compose(D_m, G):
        """基于重采样的复合算子 ◁：D_{m+1} = D_m ◁ G_{m+1}。
        D_m, G 均为 Bx2xGHxGW 的后向映射（值域 [-1,1]）。
        以 G 的坐标为索引、在 D_m 上做双线性采样，链式累积非线性映射。
        """
        grid = G.permute(0, 2, 3, 1)  # BxGHxGWx2，grid_sample 所需格式
        return F.grid_sample(D_m, grid, align_corners=True, mode="bilinear")

    def forward(self, x, max_iter=2, img_size=ITER_IMG_SIZE):
        """训练用：展开 max_iter 步细化，返回各步后向映射与矫正图。
        Returns:
            base_g2d, base_g3d: 初始 2D/3D 网格（来自 C-Net）
            grids: list[D_0, D_1, ..., D_{max_iter}]，长度 max_iter+1
            imgs:  list[O_0, O_1, ..., O_{max_iter}]，对应各步矫正图
        """
        base_g2d, base_g3d = self.initial_grid(x)
        d = base_g2d
        grids = [d]
        imgs = []
        cur = _unwarp(x, d, img_size)
        imgs.append(cur)
        for _ in range(max_iter):
            g = self.refine_grid(cur)
            d = self.compose(d, g)
            grids.append(d)
            cur = _unwarp(x, d, img_size)
            imgs.append(cur)
        return base_g2d, base_g3d, grids, imgs

    @torch.no_grad()
    def adaptive_infer(self, x, max_iter=5, tau=0.99, theta_thresh=5.0, img_size=ITER_IMG_SIZE):
        """推理用：自适应迭代停止（论文 Alg.1 / 专利）。

        严格按论文/专利实现：在粗矫正图 O_1 上**检测一次**线段（FilterAlignedLines），
        之后每个迭代步用形变场 G_{n+1} 经 RBF 插值把同一组参考线**变换**到 O_{n+1}
        （WarpLines），而非重新检测；当线熵改善率 < tau 时停止，返回最佳后向映射。
        迭代次数无需人工设定。

        Returns:
            best_D: 最终后向映射（Bx2xGHxGW）
            n_opt : 实际采用的迭代次数（0 表示仅粗矫正）
        """
        # 局部导入（line_utils 不依赖 model/utils，无任何循环依赖）
        from line_utils import (
            detect_aligned_segments,
            detect_all_segments,
            aligned_line_entropy_from_segments,
            segment_ang_axis,
            warp_segments,
        )

        base_g2d, _ = self.base_net(x)
        d = base_g2d
        o = _unwarp(x, d, img_size)

        # 1) 在 O_1 上检测一次、并筛选轴对齐参考线（Lref）
        segs, angs = detect_aligned_segments(o, theta_thresh=theta_thresh)
        if len(segs) == 0:
            # 兜底：放宽到全部线段，避免无参考线而直接停止
            segs, angs = detect_all_segments(o)
        if len(segs) == 0:
            return d, 0  # 确实无线段 -> 返回粗矫正结果

        # 2) 初始线熵 S_best（论文/专利 Sbest = CalculateLineEntropy(Lref)）
        s_best = aligned_line_entropy_from_segments(segs, angs)
        n_opt = 0
        best_d = d
        cur = o

        for m in range(1, max_iter + 1):
            g = self.refine_net(cur)[0]            # G_{m+1}: O_{m+1} -> O_m 坐标形变场
            d = self.compose(d, g)                  # D_{m+1} = D_m ◁ G_{m+1}
            o = _unwarp(x, d, img_size)             # O_{m+1}（供下一步 F-Net 输入）
            # 把参考线从 O_m 像素坐标系经 RBF 逆映射变换到 O_{n+1}（WarpLines）
            warped_segs = warp_segments(segs, g, img_size)
            warped_angs = segment_ang_axis(warped_segs)
            s_n = aligned_line_entropy_from_segments(warped_segs, warped_angs)
            if s_n < tau * s_best:
                s_best, n_opt, segs, angs, best_d = s_n, m, warped_segs, warped_angs, d
                cur = o
            else:
                break
        return best_d, n_opt
