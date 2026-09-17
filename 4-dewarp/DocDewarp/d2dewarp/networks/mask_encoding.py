"""二值掩码 -> 稠密几何编码

三个输入通道（h_line, v_line, boundary）都是极稀疏的二值图（线条仅占 1~3% 像素），
直接用普通卷积做感受野增长太慢，因此这里先把稀疏线索"稠密化"成：
    - 线序场  (row order / column order)  ：双向 cumsum 归一化，单调且对断裂鲁棒
    - 软距离场 (soft distance field)      ：多尺度均值金字塔，让线条快速扩散到全图
    - 内部区域 + 零阶展平坐标 (u, v)       ：由 boundary 推 inside，再推归一化矩形坐标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.unet_parts import DoubleConv


def cdf_position(mask, dim, eps=1e-6):
    """双向累积归一化：把稀疏二值线扩散成稠密的单调位置场，取值 [0,1]。

    对断裂、不连通的线也鲁棒（不依赖连通性）。
    """
    c_fwd = torch.cumsum(mask, dim=dim)
    c_bwd = torch.flip(torch.cumsum(torch.flip(mask, dims=[dim]), dim=dim), dims=[dim])
    return c_fwd / (c_fwd + c_bwd + eps)


class SoftDistanceField(nn.Module):
    """多尺度均值金字塔 -> 可微的软距离场，让稀疏线条快速扩散到全图。"""

    def __init__(self, scales=(1, 2, 4, 8, 16, 32), out_ch=4):
        super().__init__()
        self.scales = scales
        self.fuse = nn.Sequential(
            nn.Conv2d(len(scales), 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_ch, 1),
        )

    def forward(self, x):  # [B,1,H,W]
        pyr = []
        for s in self.scales:
            pyr.append(x if s == 1
                       else F.avg_pool2d(x, 2 * s + 1, stride=1, padding=s))
        return self.fuse(torch.cat(pyr, dim=1))


def _first_last(mask2d):
    """mask2d: [B,N,M] 二值，沿最后一维求每行第一个 / 最后一个 1 的下标 -> [B,N]。"""
    m = mask2d.float()
    first = m.argmax(dim=-1)
    last = m.shape[-1] - 1 - torch.flip(m, [-1]).float().argmax(dim=-1)
    return first, last


def _interp_rows(prof, t, lo, hi):
    """prof: [B,H]（按行号索引）；t: [B,H,W] 连续行号 -> [B,H,W]。"""
    B, N = prof.shape
    lo = lo.view(B, 1, 1)
    hi = hi.view(B, 1, 1)
    t = t.clamp(lo, (hi - 1.001).clamp(min=lo))
    i0 = t.floor().long().clamp(0, N - 2)
    f = t - i0.float()
    p = prof.unsqueeze(2).expand(B, N, t.shape[2])
    return torch.gather(p, 1, i0) * (1 - f) + torch.gather(p, 1, i0 + 1) * f


def _interp_cols(prof, t, lo, hi):
    """prof: [B,W]（按列号索引）；t: [B,H,W] 连续列号 -> [B,H,W]。"""
    B, N = prof.shape
    lo = lo.view(B, 1, 1)
    hi = hi.view(B, 1, 1)
    t = t.clamp(lo, (hi - 1.001).clamp(min=lo))
    i0 = t.floor().long().clamp(0, N - 2)
    f = t - i0.float()
    p = prof.unsqueeze(1).expand(B, t.shape[1], N)
    return torch.gather(p, 2, i0) * (1 - f) + torch.gather(p, 2, i0 + 1) * f


def invert_uv_prior(inside, iters=4):
    """把 (u,v) 这个【前向】参数化求逆，得到零阶【后向】map。

    重要：由 inside 的双向 cumsum 得到的 (u,v) 是
        "畸变图上的点 (x,y) 对应展平图上的 (u,v)"  —— 前向（distorted -> flat）
    而网络要输出的 bm 是 backward map：
        "展平图上的点 (i,j) 应该去畸变图的哪里采样" —— 后向（flat -> distorted）
    两者互为逆映射，直接拿 (u,v)*(S-1) 当 bm 初始化会显著变差（实测 55px vs 21px）。

    连续极限下前向模型可写成（L/R/T/B 为每行/每列的文档边界）：
        u(x,y) = (x - L(y)) / (R(y) - L(y))
        v(x,y) = (y - T(x)) / (B(x) - T(x))
    求逆即解不动点：
        x = L(y) + a * (R(y) - L(y))
        y = T(x) + b * (B(x) - T(x))
    其中 (a,b) 是展平网格的归一化坐标。quad 接近凸四边形时 3~4 次迭代即收敛。

    inside : [B,1,H,W]
    返回   : [B,2,H,W] 绝对像素坐标的零阶 backward map
    """
    B, _, H, W = inside.shape
    m = inside.squeeze(1)                                   # [B,H,W]
    dev = m.device

    L, R = _first_last(m)                                   # [B,H] 每行左右边界
    T, Bo = _first_last(m.transpose(1, 2))                  # [B,W] 每列上下边界

    # 有效行 / 列范围（把插值索引夹进合法区间，避免空行污染边界插值）
    ar = torch.arange(H, device=dev).view(1, H).expand(B, H)
    ac = torch.arange(W, device=dev).view(1, W).expand(B, W)
    row_valid, col_valid = m.sum(dim=2) > 0, m.sum(dim=1) > 0
    y0 = torch.where(row_valid, ar, torch.full_like(ar, H)).min(dim=1).values
    y1 = torch.where(row_valid, ar, torch.full_like(ar, -1)).max(dim=1).values + 1
    x0 = torch.where(col_valid, ac, torch.full_like(ac, W)).min(dim=1).values
    x1 = torch.where(col_valid, ac, torch.full_like(ac, -1)).max(dim=1).values + 1
    y0 = y0.clamp(0, H - 1).float()
    y1 = y1.clamp(1, H).float()
    x0 = x0.clamp(0, W - 1).float()
    x1 = x1.clamp(1, W).float()

    a = torch.linspace(0, 1, W, device=dev).view(1, 1, W).expand(B, H, W)
    b = torch.linspace(0, 1, H, device=dev).view(1, H, 1).expand(B, H, W)

    y = y0.view(B, 1, 1) + b * (y1 - y0).view(B, 1, 1)
    x = a * (W - 1)
    for _ in range(iters):
        xl = _interp_rows(L.float(), y, y0, y1)
        xr = _interp_rows(R.float(), y, y0, y1)
        x = xl + a * (xr - xl)
        yt = _interp_cols(T.float(), x, x0, x1)
        yb = _interp_cols(Bo.float(), x, x0, x1)
        y = yt + b * (yb - yt)
    xl = _interp_rows(L.float(), y, y0, y1)
    xr = _interp_rows(R.float(), y, y0, y1)
    x = xl + a * (xr - xl)
    return torch.stack([x, y], dim=1)


class MaskGeomEncoder(nn.Module):
    """输入 [B,3,H,W] 二值掩码 (h_line, v_line, boundary) -> 稠密几何特征 + 先验。

    返回: feat[out_ch], inside[1], uv_prior[2], oh[1], ov[1], dh[dt_ch], dv[dt_ch]
      inside   : boundary 围出的文档区域
      uv_prior : inside 的归一化矩形坐标 = 展平坐标的零阶估计
      oh / ov  : 水平线的“行序场” / 竖直线的“列序场”
      dh / dv  : 水平线 / 竖直线的软距离场
    """

    def __init__(self, out_ch=64, dt_ch=4):
        super().__init__()
        self.dt_h = SoftDistanceField(out_ch=dt_ch)
        self.dt_v = SoftDistanceField(out_ch=dt_ch)
        self.dt_b = SoftDistanceField(out_ch=dt_ch)
        in_ch = 3 + 3 * dt_ch + 1 + 2 + 2 + 2  # raw+dt+inside+(oh,ov)+(u,v)+(x,y) = 22
        self.stem = DoubleConv(in_ch, out_ch)
        self.dt_ch = dt_ch
        self.in_ch = in_ch

    def forward(self, masks):
        mh, mv, mb = masks[:, 0:1], masks[:, 1:2], masks[:, 2:3]
        B, _, H, W = masks.shape

        # 1) boundary -> inside（四个方向的累积求交）
        cl = torch.cumsum(mb, dim=3)
        cr = torch.flip(torch.cumsum(torch.flip(mb, [3]), 3), [3])
        ct = torch.cumsum(mb, dim=2)
        cb = torch.flip(torch.cumsum(torch.flip(mb, [2]), 2), [2])
        inside = ((cl > 0) & (cr > 0) & (ct > 0) & (cb > 0)).float()

        # 兜底：文档贴边 / 边界被裁切时 inside 会退化为空，此时退回全图，
        # 让零阶先验退化成单位网格（等价于标准 RAFT 初始化），避免出现全 0 的 NaN。
        area = inside.flatten(1).sum(1).view(B, 1, 1, 1)
        empty = (area < 0.01 * H * W).float()
        inside = inside * (1.0 - empty) + empty

        # 2) inside -> 归一化矩形坐标（零阶展平坐标先验，最关键的稠密信号）
        sl = torch.cumsum(inside, dim=3)
        sr = torch.flip(torch.cumsum(torch.flip(inside, [3]), 3), [3])
        st = torch.cumsum(inside, dim=2)
        sb = torch.flip(torch.cumsum(torch.flip(inside, [2]), 2), [2])
        u = sl / (sl + sr + 1e-6)   # 归一化列坐标 -> 约束 bm 的 x
        v = st / (st + sb + 1e-6)   # 归一化行坐标 -> 约束 bm 的 y

        # 3) 线序场：水平线沿 y 累积，竖直线沿 x 累积
        oh = cdf_position(mh * inside, dim=2)
        ov = cdf_position(mv * inside, dim=3)

        # 4) 绝对坐标（CoordConv，让网络知道像素位置）
        yy = torch.linspace(0, 1, H, device=masks.device).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(0, 1, W, device=masks.device).view(1, 1, 1, W).expand(B, 1, H, W)

        dh, dv, db = self.dt_h(mh), self.dt_v(mv), self.dt_b(mb)
        feat = self.stem(torch.cat(
            [mh, mv, mb, dh, dv, db, inside, oh, ov, u, v, xx, yy], dim=1))
        return feat, inside, torch.cat([u, v], dim=1), oh, ov, dh, dv


if __name__ == '__main__':
    _m = torch.randint(0, 2, (2, 3, 448, 448)).float().cuda()
    # 造一个矩形边界环
    _m[:, 2, :, :] = 0.0
    _m[:, 2, 40:410, 40] = 1.0
    _m[:, 2, 40:410, 408] = 1.0
    _m[:, 2, 40, 40:410] = 1.0
    _m[:, 2, 408, 40:410] = 1.0

    _net = MaskGeomEncoder().cuda()
    _f, _ins, _uv, _oh, _ov, _dh, _dv = _net(_m)
    print('feat', _f.shape, 'inside', _ins.shape, _ins.min().item(), _ins.max().item())
    print('uv   ', _uv.shape, _uv.min().item(), _uv.max().item())
    print('oh/ov', _oh.shape, _ov.shape, 'dh/dv', _dh.shape, _dv.shape)
