# -*- coding: utf-8 -*-
"""边拟合精修(snap): 用图像梯度把预测四边形的每条边重新拟合, 再求交点

解决的问题: 缺边时模型把画面内的真实角点朝目标内部拉(边没延伸到画面边框)。
每条边在图像里通常是长而清晰的直线, 用梯度峰值拟合比直接回归角点稳得多。

含位移守卫: 精修后的角点相对原预测位移超过 max_move_ratio * 目标短边时退回原值,
           防止贴合到背景里错误的边缘(实测不加守卫会出现 +140px 的灾难样本)。
"""
import itertools
import numpy as np
import cv2


def bilinear(img, x, y):
    H, W = img.shape[:2]
    if x < 0 or y < 0 or x > W - 1.001 or y > H - 1.001:
        return 0.0
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    return float((img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
                  + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy))


def find_edge_offset(gmag, px, py, nx, ny, band, nstep=31):
    ds = np.linspace(-band, band, nstep)
    vals = np.array([bilinear(gmag, px + nx * d, py + ny * d) for d in ds])
    i = int(np.argmax(vals))
    if i == 0 or i == len(ds) - 1:
        return float(ds[i]), float(vals[i])
    y0, y1, y2 = vals[i - 1], vals[i], vals[i + 1]
    denom = (y0 - 2 * y1 + y2)
    off = ds[i] + 0.5 * (y0 - y2) / denom * (ds[1] - ds[0]) if abs(denom) > 1e-9 else ds[i]
    return float(off), float(vals[i])


def fit_line_ransac(pts, res=2.5, iters=120, seed=0):
    if len(pts) < 8:
        return None, 0
    rng = np.random.default_rng(seed)
    n = len(pts)
    best_in, best_line = None, None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        p, q = pts[i], pts[j]
        d = q - p
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        d = d / L
        nv = np.array([-d[1], d[0]])
        inl = np.abs((pts - p) @ nv) < res
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best_line = inl, (p, d)
    if best_in is None or best_in.sum() < 8:
        return None, 0
    P = pts[best_in]
    c = P.mean(0)
    _u, _s, vt = np.linalg.svd(P - c)
    d = vt[0] / (np.linalg.norm(vt[0]) + 1e-12)
    return (c, d), int(best_in.sum())


def clip_poly(poly, W, H, tol=0.5):
    """Sutherland-Hodgman 裁剪到 [0,W]x[0,H]

    tol: 把距离边框 0.5px 内的点吸附到边框上, 避免 -0.0 之类浮点误差
         把"贴在边框上"的角点判成画外, 凭空多出一个顶点。
    """
    pts = []
    for p in poly:
        q = np.array(p, float).copy()
        if abs(q[0]) <= tol:
            q[0] = 0.0
        if abs(q[0] - W) <= tol:
            q[0] = W
        if abs(q[1]) <= tol:
            q[1] = 0.0
        if abs(q[1] - H) <= tol:
            q[1] = H
        pts.append(q)
    for axis, lim, sign in ((0, 0.0, 1), (0, W, -1), (1, 0.0, 1), (1, H, -1)):
        if len(pts) < 3:
            return np.zeros((0, 2))
        out = []
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            ia = sign * (a[axis] - lim) >= 0
            ib = sign * (b[axis] - lim) >= 0
            if ia != ib:
                t = (lim - a[axis]) / (b[axis] - a[axis] + 1e-12)
                out.append(a + t * (b - a))
            if ib:
                out.append(b)
        pts = out
    if len(pts) < 3:
        return np.zeros((0, 2))
    ded = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - ded[-1]) > 1e-6:
            ded.append(p)
    if len(ded) > 1 and np.linalg.norm(ded[0] - ded[-1]) < 1e-6:
        ded.pop()
    return np.array(ded)


def poly_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def largest_quad(poly):
    """从凸多边形顶点里取面积最大的 4 个(保序)"""
    n = len(poly)
    if n <= 4:
        return poly
    best, best_a = None, -1.0
    for idx in itertools.combinations(range(n), 4):
        q = poly[list(idx)]
        a = poly_area(q)
        if a > best_a:
            best, best_a = q, a
    return best


def visible_quad(quad, W, H):
    """真实/预测四边形 -> 可见区域四边形。返回 (4x2 角点, 是否裁掉了角)"""
    vis = clip_poly(quad, W, H)
    if len(vis) == 4:
        return vis, False
    if len(vis) > 4:
        return largest_quad(vis), True
    # 交集只剩 3 个顶点(目标几乎整块在画面外): 退回原始四边形并标记
    return np.asarray(quad, float), True



def refine_quad(img, quad, band_ratio=0.015, n_samples=80, res=2.5, max_move_ratio=0.05):
    """img: BGR 原图; quad: 4x2 预测角点。返回 (精修四角, 边, 每条边是否成功)"""
    H, W = img.shape[:2]
    quad = np.asarray(quad, float)
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    max_move = max_move_ratio * max(1.0, min(w, h))
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)
    lines, ok = [], []
    for k in range(4):
        p, q = quad[k], quad[(k + 1) % 4]
        L = np.linalg.norm(q - p)
        if L < 20:
            lines.append(None); ok.append(False); continue
        d = (q - p) / L
        nv = np.array([-d[1], d[0]])
        band = max(6.0, band_ratio * L)
        collect = []
        for t in np.linspace(0.05, 0.95, n_samples):
            s = p + (q - p) * t
            if not (0 <= s[0] <= W - 1 and 0 <= s[1] <= H - 1):
                continue
            off, mag = find_edge_offset(gmag, s[0], s[1], nv[0], nv[1], band)
            if mag < 8.0:
                continue
            collect.append(s + nv * off)
        line, ninl = fit_line_ransac(np.array(collect)) if collect else (None, 0)
        lines.append(line)
        ok.append(line is not None and ninl >= 25)

    def inter(l1, l2):
        (p1, d1), (p2, d2) = l1, l2
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-9:
            return None
        t = np.linalg.solve(A, p2 - p1)
        return p1 + d1 * t[0]

    out = []
    for k in range(4):
        i1, i2 = (k - 1) % 4, k
        l1, l2 = lines[i1], lines[i2]
        # 门控: 相邻两条边必须都"拟合可靠"(内点数够), 否则这个角点不动。
        # 依据: 真实书本照片上 (文字多/书页弯曲) 中位只有 1/4 条边能可靠拟合,
        #       此时精修会把角点拉偏(实测 41→50px); 干净场景中位 4/4 条, 精修大幅变好(15→4px)。
        if l1 is None or l2 is None or not ok[i1] or not ok[i2]:
            out.append(quad[k]); continue
        pt = inter(l1, l2)
        out.append(quad[k] if (pt is None or np.linalg.norm(pt - quad[k]) > max_move) else pt)
    return np.array(out), lines, ok
