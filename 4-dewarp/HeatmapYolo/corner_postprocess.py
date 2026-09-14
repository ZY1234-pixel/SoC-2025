# -*- coding: utf-8 -*-
"""角点后处理：置信度门控 + 几何一致性修复

针对三类失败：
1. 角点热力图弱响应/贴边 -> 解码点飞出画面
2. 强旋转下弱通道峰值被邻近通道"吸走" -> 两个角点扎堆
3. 排序错乱导致连线交叉（漏斗形）
"""
import numpy as np


def order_corners(pts):
    """把 4 个点按几何角色重排为 TL, TR, BR, BL

    用质心极角得到凸四边形的环形顺序（保证连线不交叉），
    再以 TL(x+y 最小) 为起点，用邻点中 x-y 更大者为 TR 确定绕行方向。
    """
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    cyc = np.argsort(ang)
    i_tl = int(np.argmin(pts[:, 0] + pts[:, 1]))
    pos = int(np.where(cyc == i_tl)[0][0])
    nxt, prv = cyc[(pos + 1) % 4], cyc[(pos - 1) % 4]
    if pts[nxt, 0] - pts[nxt, 1] > pts[prv, 0] - pts[prv, 1]:
        order = [cyc[(pos + k) % 4] for k in range(4)]
    else:
        order = [cyc[(pos - k) % 4] for k in range(4)]
    return pts[order], order


def _convex_score(quad):
    """凸性 + 矩形度打分：非凸 -> -inf；凸 -> -平均内角偏差（越小越接近矩形）"""
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    q = quad[np.argsort(ang)]
    signs = []
    for i in range(4):
        a = q[(i + 1) % 4] - q[i]
        b = q[(i + 2) % 4] - q[(i + 1) % 4]
        signs.append(a[0] * b[1] - a[1] * b[0])
    if not (np.all(np.array(signs) >= 0) or np.all(np.array(signs) <= 0)):
        return -float("inf")
    dev = 0.0
    for i in range(4):
        a = q[(i + 1) % 4] - q[i]
        b = q[(i - 1) % 4] - q[i]
        cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        dev += abs(np.degrees(np.arccos(np.clip(cosang, -1, 1))) - 90.0)
    return -dev


def _best_completion(p3):
    """给定 3 个保留角点，返回使四边形最接近矩形的第 4 个点"""
    best, best_score = None, -float("inf")
    for b in range(3):
        a, c = p3[(b + 1) % 3], p3[(b + 2) % 3]
        d = a + c - p3[b]
        score = _convex_score(np.array([a, p3[b], c, d]))
        if score > best_score:
            best_score, best = score, d
    return best


def _min_pair(pts):
    dmin, ii, jj = float("inf"), -1, -1
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if d < dmin:
                dmin, ii, jj = d, i, j
    return dmin, ii, jj


def postprocess_corners(pred, conf, img_w, img_h, dup_ratio=0.05, out_ratio=0.02, conf_thr=0.8):
    """修复角点并保证顺序 TL,TR,BR,BL

    Args:
        pred: (4,2) 原图像素坐标
        conf: (4,) 各通道热力图峰值（0~1），作为置信度
        img_w, img_h: 原图宽高
        dup_ratio: 两个角点距离小于该比例(相对短边)视为扎堆
        out_ratio: 超出画面该比例即视为飞出；另外贴边且低置信也会触发几何修复
        conf_thr: 低置信度阈值
    Returns:
        (4,2) 修复后的角点（顺序 TL,TR,BR,BL），已夹取到画面内
    """
    W, H = float(img_w), float(img_h)
    pts = np.asarray(pred, dtype=float).copy()
    conf = np.asarray(conf, dtype=float).copy()
    thr = dup_ratio * min(W, H)

    # 1) 扎堆去重：最近两点 < thr 时保留置信度高者，低者用平行四边形补全
    for _ in range(3):
        dmin, ii, jj = _min_pair(pts)
        if dmin >= thr:
            break
        keep, drop = (ii, jj) if conf[ii] >= conf[jj] else (jj, ii)
        p3 = np.array([pts[i] for i in range(4) if i != drop])
        est = _best_completion(p3)
        if est is None:
            break
        pts = np.vstack([p3, est])
        conf = np.array([conf[i] for i in range(4) if i != drop] + [0.0])

    # 2) 排序
    q, _ = order_corners(pts)

    # 3) 飞出画面 / 贴边且低置信度：用平行四边形替换
    mx, my = out_ratio * W, out_ratio * H
    edge_thr = 0.01
    for i in range(4):
        p = q[i]
        out = p[0] < -mx or p[0] > W + mx or p[1] < -my or p[1] > H + my
        near_edge = (
            conf[i] < conf_thr
            and (p[0] < edge_thr * W or p[0] > (1 - edge_thr) * W or p[1] < edge_thr * H or p[1] > (1 - edge_thr) * H)
        )
        if (out or near_edge) and conf[i] < conf_thr:
            q[i] = q[(i - 1) % 4] + q[(i + 1) % 4] - q[(i + 2) % 4]
    q, _ = order_corners(q)

    # 4) 最终夹取到画面内
    q[:, 0] = np.clip(q[:, 0], 0, W - 1)
    q[:, 1] = np.clip(q[:, 1], 0, H - 1)
    return q
