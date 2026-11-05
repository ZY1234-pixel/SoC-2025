"""线段检测 / 变换（RBF 逆向映射）/ 轴对齐线熵 的辅助函数。

与文档矫正自适应迭代（论文 Alg.1 / 专利）相关：
- 在粗矫正图上用 LSD 检测一次线段，并筛选轴对齐参考线（FilterAlignedLines）；
- 之后每个迭代步用形变场把同一组参考线经 RBF 插值变换到下一步（WarpLines）；
- 计算长度加权平均的轴向偏离角（轴对齐线熵 H_align）。

本模块不依赖 model / utils，避免循环导入。
"""
import cv2
import numpy as np


def _to_gray(img_tensor):
    """把 [1,C,H,W] 张量转为灰度 uint8 图（C 可为 3 或 1）。"""
    img = img_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    if img.ndim == 3 and img.shape[2] == 3:
        img = np.clip(img, 0.0, 1.0)
        gray = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = np.clip(img, 0.0, 255.0).astype(np.uint8)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    return gray


def _lsd_detect(gray):
    """LSD 线段检测，返回 Nx4 (x1,y1,x2,y2) 线段数组；无线段时返回空数组。"""
    try:
        lsd = cv2.createLineSegmentDetector()
        detected = lsd.detect(gray)
        lines = detected[0] if detected is not None else None
    except Exception:
        lines = None
    if lines is None or len(lines) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


def segment_ang_axis(segments):
    """每条线段与"最近坐标轴"的夹角（度，[0,45]）；线段数组为空时返回空数组。"""
    if len(segments) == 0:
        return np.zeros((0,), dtype=np.float32)
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    ang = np.abs(np.degrees(np.arctan2(dy, dx))) % 180.0
    ang = np.minimum(ang, 180.0 - ang)  # [0, 90]
    ang_axis = np.minimum(ang, 90.0 - ang)  # [0, 45]
    return ang_axis.astype(np.float32)


def detect_all_segments(img_tensor):
    """检测全部线段（不做角度过滤），返回 (segments[N,4], angles[N])。"""
    gray = _to_gray(img_tensor)
    seg = _lsd_detect(gray)
    return seg, segment_ang_axis(seg)


def detect_aligned_segments(img_tensor, theta_thresh=5.0):
    """参考论文/专利 FilterAlignedLines：仅保留与坐标轴夹角 < theta_thresh 的参考线。
    返回 (segments[N,4], angles[N])；若无任何对齐线段返回空数组（调用方应做兜底）。"""
    seg, angles = detect_all_segments(img_tensor)
    mask = angles < theta_thresh
    return seg[mask], angles[mask]


def aligned_line_entropy_from_segments(segments, angles):
    """长度加权平均的轴向偏离角（度，越小越好）；无线段时返回 45.0。"""
    if len(segments) == 0:
        return 45.0
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    length = np.hypot(dx, dy)
    total = length.sum()
    if total <= 0:
        return 45.0
    return float((length * angles).sum() / total)


def _rbf_inverse_map(src, val, query):
    """薄板样条 (TPS) 径向基插值，求解逆向映射：给定控制点 src(形变前位置) 与
    val(对应形变后位置)，对任意 query(形变前位置) 估计其形变后位置。
    对应专利"优选采用径向基函数 RBF 插值映射以获得精确位置"。"""
    Ns = src.shape[0]
    if Ns == 0:
        return query.copy()
    diff = src[:, None, :] - src[None, :, :]
    r2 = np.sum(diff * diff, axis=2)
    r2c = np.clip(r2, 0.0, None)
    phi = r2c * np.log(np.clip(r2c, 1e-12, None) + 1e-12)  # ~ r^2 log(r^2)
    np.fill_diagonal(phi, 0.0)
    phi += 1e-8 * np.eye(Ns)
    P = np.hstack([np.ones((Ns, 1)), src])  # (Ns,3)
    Z = np.zeros((3, 3))
    L = np.vstack([np.hstack([phi, P]), np.hstack([P.T, Z])])  # (Ns+3, Ns+3)
    rhs = np.zeros((Ns + 3, 2))
    rhs[:Ns, :] = val
    sol = np.linalg.solve(L, rhs)  # (Ns+3, 2)
    w = sol[:Ns]
    lin = sol[Ns:]  # (3,2)
    dq = query[:, None, :] - src[None, :, :]
    r2q = np.sum(dq * dq, axis=2)
    phiq = r2q * np.log(np.clip(r2q, 1e-12, None) + 1e-12)
    Pq = np.hstack([np.ones((query.shape[0], 1)), query])
    out = phiq @ w + Pq @ lin
    return out


def warp_segments(segments, g, img_size):
    """把 O_n 像素坐标系下的线段（Nx4）经形变场 g 逆向映射到 O_{n+1} 像素坐标系。
    g: Bx2xGHxGW 的归一化坐标形变场（O_{n+1} 输出像素 -> O_n 输出像素）。
    对应论文/专利 WarpLines：随累积变形更新参考线位置（RBF 插值）。"""
    if len(segments) == 0:
        return segments
    W, H = img_size
    gnp = g[0].detach().cpu().numpy().astype(np.float64)  # (2, GH, GW)
    GH, GW = gnp.shape[1], gnp.shape[2]
    # 控制点：形变前(O_n)像素位置 = g 反归一化
    gx = (gnp[0] + 1.0) / 2.0 * (W - 1)
    gy = (gnp[1] + 1.0) / 2.0 * (H - 1)
    # 控制点对应：形变后(O_{n+1})像素位置 = 网格节点索引反归一化
    jj = np.arange(GW, dtype=np.float64)
    ii = np.arange(GH, dtype=np.float64)
    JJ, II = np.meshgrid(jj, ii)
    ux = JJ * (W - 1.0) / (GW - 1)
    uy = II * (H - 1.0) / (GH - 1)
    src = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (Ns,2) O_n 像素
    val = np.stack([ux.ravel(), uy.ravel()], axis=1)  # (Ns,2) O_{n+1} 像素
    pts = segments.reshape(-1, 2).astype(np.float64)  # (2N,2) 端点
    warped = _rbf_inverse_map(src, val, pts)
    return warped.reshape(-1, 4).astype(np.float32)
