import os

import torch
import torch.nn.functional as F

from model import UVDocnet
import cv2
import numpy as np
IMG_SIZE = (488, 712)
GRID_SIZE = (45, 31)


def load_model(ckpt_path):
    """
    Load UVDocnet model.
    """
    model = UVDocnet(num_filter=32, kernel_size=5)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state"])
    return model


def get_version():
    """
    Returns the version of the various packages used for evaluation.
    """
    import pytesseract

    return {
        "tesseract": str(pytesseract.get_tesseract_version()),
        "pyesseract": os.popen("pip list | grep pytesseract").read().split()[-1],
        "Levenshtein": os.popen("pip list | grep Levenshtein").read().split()[-1],
        "jiwer": os.popen("pip list | grep jiwer").read().split()[-1],
        "matlabengineforpython": os.popen("pip list | grep matlab").read().split()[-1],
    }


def bilinear_unwarping(warped_img, point_positions, img_size):
    """
    Utility function that unwarps an image.
    Unwarp warped_img based on the 2D grid point_positions with a size img_size.
    Args:
        warped_img  :       torch.Tensor of shape BxCxHxW (dtype float)
        point_positions:    torch.Tensor of shape Bx2xGhxGw (dtype float)
        img_size:           tuple of int [w, h]
    """
    upsampled_grid = F.interpolate(
        point_positions, size=(img_size[1], img_size[0]), mode="bilinear", align_corners=True
    )
    unwarped_img = F.grid_sample(warped_img, upsampled_grid.transpose(1, 2).transpose(2, 3), align_corners=True)

    return unwarped_img


def bilinear_unwarping_from_numpy(warped_img, point_positions, img_size):
    """
    Utility function that unwarps an image.
    Unwarp warped_img based on the 2D grid point_positions with a size img_size.
    Accept numpy arrays as input.
    """
    warped_img = torch.unsqueeze(torch.from_numpy(warped_img.transpose(2, 0, 1)).float(), dim=0)
    point_positions = torch.unsqueeze(torch.from_numpy(point_positions.transpose(2, 0, 1)).float(), dim=0)

    unwarped_img = bilinear_unwarping(warped_img, point_positions, img_size)

    unwarped_img = unwarped_img[0].numpy().transpose(1, 2, 0)
    return unwarped_img

def create_document_mask(edge_img, kernel_size=5, min_contour_area=1000):
    """
    从边缘图像生成实心文档掩码（文档区域为 255，背景为 0）。

    流程：二值化 → 形态学闭运算 → 轻微膨胀 → 查找轮廓 → 填充最大轮廓及其余大轮廓

    Args:
        edge_img: 原始边缘图像（numpy array，任意尺寸，值域 0~255）
        img_size: 统一 resize 的目标尺寸（正方形边长）
        kernel_size: 形态学操作的核大小，默认 5
        min_contour_area: 次要轮廓的最小面积阈值，用于过滤噪点，默认 1000

    Returns:
        fill_mask: uint8 类型，形状 (img_size, img_size)，文档区域为 255，背景为 0
        fill_mask_float: float32 类型，同上，值域 [0, 1]，可直接用于乘法掩码
    """
    # 1. 严格二值化（消除模糊过渡像素）
    _, edge_binary = cv2.threshold(edge_img, 0, 255, cv2.THRESH_BINARY)

    # 3. 形态学闭运算：桥接断裂的边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    edge_closed = cv2.morphologyEx(edge_binary, cv2.MORPH_CLOSE, kernel)

    # 4. 轻微膨胀，让边缘线更粗、更闭合
    edge_dilated = cv2.dilate(edge_closed, kernel, iterations=1)

    # 5. 查找轮廓并填充
    contours, _ = cv2.findContours(edge_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fill_mask = np.zeros_like(edge_binary)
    if len(contours) > 0:
        # 按面积排序，取最大的轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # 填充最大轮廓（通常是文档主体）
        cv2.drawContours(fill_mask, [contours[0]], -1, 255, thickness=cv2.FILLED)

        # 填充其余较大的闭合区域（如折叠处的内轮廓）
        for cnt in contours[1:]:
            if cv2.contourArea(cnt) > min_contour_area:
                cv2.drawContours(fill_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    fill_mask_float = fill_mask.astype(np.float32) / 255.0

    return fill_mask, fill_mask_float


# -----------------------------------------------------------------------------#
#   轴对齐相关：可微训练损失 (L_AL) + 推理用自适应停止指标 (H_align)
#
#   L_AL 按 AADD《Axis-aligned document dewarping》第 3.1 节 Eq.(3)-(7) 实现：
#     - Eq.(3)/(4)：GT 网格在 UV 空间是均匀网格 Q^GT，
#       f(p) = Interpolate(Q^GT ; P^GT, P) 由 GT 2D 网格定义（专利 S2 的 BM->UV）；
#     - Eq.(5)-(7)：把预测 2D 后向网格映射到 UV 空间，
#       L_hor = Σ_行 Var(v)、L_ver = Σ_列 Var(u)、L_AL = L_hor + L_ver。
#   数值实现（方案 A+C）：
#     - f 由 GT 2D 网格散布得到，Q^GT 为均匀网格 => L_AL(P^GT)=0 且梯度为 0；
#     - UV 图空洞用邻域有效值扩散填充，避免填 0 在边界处产生尖峰梯度；
#     - 方差按行/列数归一化到与 L2D 同量级（论文 gamma=0.2，L_AL 仅作正则项）；
#     - 有效性使用连续软 mask，grid_sample 使用 border padding。
#   推理用的自适应停止指标仍是非可微的 axis_aligned_line_entropy。
# -----------------------------------------------------------------------------#
def _inpaint_holes(uv, valid, iters=32):
    """把 UV 图中的空洞（valid=0）用邻域有效值的均值迭代扩散填充。

    uv: (B,2,Hc,Wc)，valid: (B,1,Hc,Wc)。每次迭代把有效区外扩 1 像素，
    全程由可微算子构成，避免"空洞填 0"在 grid_sample 边界处产生尖峰梯度。
    """
    k = 3
    for _ in range(iters):
        num = F.avg_pool2d(uv * valid, k, stride=1, padding=k // 2) * (k * k)
        den = F.avg_pool2d(valid, k, stride=1, padding=k // 2) * (k * k)
        filled = num / den.clamp_min(1e-8)
        grow = (den > 0).float()
        uv = torch.where(valid.bool(), uv, filled)
        valid = torch.maximum(valid, grow)
    return uv


def _build_uv_map(grid2D_gt, canvas):
    """由 GT 的 2D 后向网格（输出->源，[-1,1]）构建 UV 空间映射图。

    对应专利 S2 的 BM->UV：把每个 GT 节点 (i,j) 的"节点坐标"按其在源空间的位置
    双线性散布(splat)到 canvas 上，得到一张以源坐标为索引、存储对应 UV(节点)坐标的图。
    随后对任意预测源点用 F.grid_sample 取回其 UV 坐标，即实现
    f(.) = (源坐标 -> 节点坐标) 的 GT 逆映射（全程可微）。

    Args:
        grid2D_gt: 真值 2D 后向网格 Bx2xHxW（[-1,1]）。
        canvas:    UV 映射图分辨率 (Hc, Wc)。
    Returns:
        uv_map: Bx2xHcxWc，通道 0=u(随行)，通道 1=v(随列)，值域归一化 [-1,1]。
    """
    B, _, h, w = grid2D_gt.shape
    Hc, Wc = canvas
    dev = grid2D_gt.device

    # 节点坐标（归一化 UV 空间 [-1,1]）：u 随"列索引 j"（水平方向，常量沿行），
    # v 随"行索引 i"（垂直方向，常量沿列）。这样 L_ver=Var_跨行(u)→0、L_hor=Var_跨列(v)→0。
    # （与真值 uvmap 的通道约定一致：uvmap[0]=u=水平、uvmap[1]=v=垂直。）
    ii = torch.arange(h, device=dev).view(1, h, 1).expand(B, h, w).reshape(-1)
    jj = torch.arange(w, device=dev).view(1, 1, w).expand(B, h, w).reshape(-1)
    vals_u = (2.0 * jj / max(1, w - 1) - 1.0)
    vals_v = (2.0 * ii / max(1, h - 1) - 1.0)

    # GT 节点在源空间的位置 -> canvas 像素坐标（与 grid_sample align_corners=True 一致）。
    sx = (grid2D_gt[:, 0].reshape(-1) + 1.0) / 2.0 * (Wc - 1)  # 列方向
    sy = (grid2D_gt[:, 1].reshape(-1) + 1.0) / 2.0 * (Hc - 1)  # 行方向
    x0 = torch.floor(sx).long().clamp(0, Wc - 1)
    y0 = torch.floor(sy).long().clamp(0, Hc - 1)
    x1 = (x0 + 1).clamp(0, Wc - 1)
    y1 = (y0 + 1).clamp(0, Hc - 1)
    wx1 = (sx - x0.float()).clamp(0.0, 1.0)
    wy1 = (sy - y0.float()).clamp(0.0, 1.0)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    uv_accum = torch.zeros(B, 2, Hc, Wc, device=dev)
    w_accum = torch.zeros(B, 1, Hc, Wc, device=dev)
    # 必须 view(B, 1, 1)：view(-1, 1) 是二维 (B,1)，expand 到三维 (B,h,w) 时会被
    # 右对齐成 (1,B,1)，导致 "expanded size (h) must match (B)" 的运行时错误。
    bidx = torch.arange(B, device=dev).view(B, 1, 1).expand(B, h, w).reshape(-1)
    z0 = torch.zeros_like(bidx)
    z1 = torch.ones_like(bidx)

    def _splat(yy, xx, ww):
        uv_accum.index_put_((bidx, z0, yy, xx), ww * vals_u, accumulate=True)
        uv_accum.index_put_((bidx, z1, yy, xx), ww * vals_v, accumulate=True)
        w_accum.index_put_((bidx, z0, yy, xx), ww, accumulate=True)

    _splat(y0, x0, wy0 * wx0)
    _splat(y0, x1, wy0 * wx1)
    _splat(y1, x0, wy1 * wx0)
    _splat(y1, x1, wy1 * wx1)

    valid = (w_accum > 0).float()
    uv_map = uv_accum / w_accum.clamp_min(1e-8)
    # GT 网格只散布到 canvas 的稀疏位置，其余是空洞；若保持为 0，grid_sample 在
    # 空洞边缘会产生 u/v 突变和尖峰梯度。用邻域有效值扩散填充，得到平滑 UV 图。
    uv_map = _inpaint_holes(uv_map, valid)
    return uv_map


def axis_alignment_loss(grid2D, grid2D_gt=None, uv_map_gt=None, canvas=None):
    """可微的轴对齐损失（Axis-Aligned Geometric Constraint Loss, L_AL）。

    AADD Eq.(3)-(7)：
      1) f 由 GT 定义（默认路径，方案 A）：用 GT 2D 网格 _build_uv_map 散布出均匀
         UV 网格 Q^GT，对应 f(p)=Interpolate(Q^GT; P^GT, P) 与专利 S2 的 BM->UV。
         该路径保证 L_AL(P^GT)=0 且梯度为 0（论文 Eq.(3) 的均匀网格性质）。
         若显式传入 uv_map_gt，仍可作为 f 使用，但 NaN 区域会平滑填充、软 mask 参与统计。
      2) 预测 2D 后向网格 grid2D 经 f 映射到 UV 空间得到 q=(u,v)；
      3) L_hor = Σ_行 Var_跨列(v)，L_ver = Σ_列 Var_跨行(u)，L_AL = 二者之和。

    数值处理：方差按论文 Eq.(6) 求和（恢复求和量级，不再按行/列数取均值）、
    连续软 mask（不二值化、不 detach）、去掉 clamp(var,0) 的折点、
    grid_sample 使用 border padding、UV 图空洞用邻域有效值扩散填充。
    L_AL 仍是论文 gamma=0.2 意义下的正则项；求和使其量级约为逐点平均 L2D 的
    数十倍，需与 delta_w、cross loss 的量级一起权衡。

    Args:
        grid2D:     预测 2D 后向网格 Bx2xHxW（[-1,1]）。
        grid2D_gt:  真值 2D 后向网格 Bx2xHxW（[-1,1]），默认路径必需。
        uv_map_gt:  可选，UVDoc 真值 uvmap Bx2xHcxWc（NaN 为文档外）。
                   为与论文 Eq.(3) 的均匀 Q^GT 一致，训练时不传此参数。
        canvas:     UV 映射图分辨率 (Hc, Wc)，默认 4*H x 4*W。
    """
    if grid2D is None:
        return torch.zeros((), device="cpu")

    if uv_map_gt is not None:
        uv_map = uv_map_gt.to(device=grid2D.device, dtype=grid2D.dtype)
        valid = (~torch.isnan(uv_map))[:, :1, :, :].float()
        uv_filled = torch.where(valid.bool(), uv_map, torch.zeros_like(uv_map))
        uv_filled = _inpaint_holes(uv_filled, valid)
        mask_src = valid
    else:
        if grid2D_gt is None:
            return torch.zeros((), device=grid2D.device)
        if canvas is None:
            canvas = (4 * grid2D.shape[2], 4 * grid2D.shape[3])
        uv_filled = _build_uv_map(grid2D_gt, canvas)   # B x 2 x Hc x Wc
        mask_src = None                                 # GT 网格完整覆盖，无需 mask

    grid = grid2D.permute(0, 2, 3, 1)
    q = F.grid_sample(
        uv_filled, grid, align_corners=True, mode="bilinear", padding_mode="border"
    )
    u, v = q[:, 0], q[:, 1]

    if mask_src is None:
        mask = torch.ones_like(u)
    else:
        m = F.grid_sample(
            mask_src, grid, align_corners=True, mode="bilinear", padding_mode="border"
        )
        mask = m.squeeze(1).clamp(0.0, 1.0)   # 连续软 mask，保留梯度

    # q[:, 0/1] 为 (B, h=行, w=列)：列方向=dim2，行方向=dim1。
    # AADD Eq.(6)：L_hor = Σ_行 Var(v)、L_ver = Σ_列 Var(u)，按论文取求和。
    l_hor = _masked_var(v, mask, 2).sum(dim=1)   # B
    l_ver = _masked_var(u, mask, 1).sum(dim=1)   # B
    return (l_hor + l_ver).mean()


def _masked_var(x, mask, dim):
    """沿 dim 计算（软）加权方差：var = E[x^2] - E[x]^2。

    mask 为连续权重（论文默认路径恒为 1），不再二值化/detach/clamp，
    保证在 L_AL 极小点附近梯度平滑。
    """
    n = mask.sum(dim).clamp(min=1.0)
    s1 = (x * mask).sum(dim)
    s2 = (x * x * mask).sum(dim)
    return s2 / n - (s1 / n) ** 2


def axis_aligned_line_entropy(img_tensor, theta_thresh=5.0):
    """轴对齐线熵 H_align（推理用、非可微）：基于 LSD 线段检测，筛选与坐标轴夹角
    < theta_thresh 的参考线，返回其"长度加权平均轴向偏离角"（度，越小越好）。

    用于迭代细化的自适应停止：当改善率低于阈值（论文 tau=0.99）时终止迭代。
    img_tensor: [1,C,H,W] 张量（值域任意，内部归一化到 0~255）。

    Returns:
        float，角度（度）。无可参考线段时返回较大熵(45.0)，避免误判为已对齐而过早停止。
    """
    from line_utils import detect_aligned_segments, aligned_line_entropy_from_segments

    seg, angles = detect_aligned_segments(img_tensor, theta_thresh)
    return aligned_line_entropy_from_segments(seg, angles)


def load_iterative_model(ckpt_path, num_filter=32, kernel_size=5):
    """加载 UVDocIterative 模型，兼容旧版 UVDocnet 权重。

    - 若 ckpt 含 base_net.* / refine_net.* 键：直接加载（完整迭代模型）。
    - 若 ckpt 为旧版 UVDocnet（无前缀键）：把权重同时映射给 base_net 与 refine_net，
      使 F-Net 从 C-Net 初始化（论文中 F-Net 由收敛后的 C-Net 初始化）。
    """
    from model import UVDocIterative

    model = UVDocIterative(num_filter=num_filter, kernel_size=kernel_size)
    ckpt = torch.load(ckpt_path)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    if any(k.startswith("base_net.") for k in state.keys()):
        model.load_state_dict(state, strict=True)
    else:
        new_state = {}
        for k, v in state.items():
            new_state["base_net." + k] = v
            new_state["refine_net." + k] = v
        model.load_state_dict(new_state, strict=False)
    return model
