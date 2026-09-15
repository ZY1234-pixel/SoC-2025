# -*- coding: utf-8 -*-
"""角点规范重排(与标注脚本 extract_corners_rounded.order_corners 完全一致)

规则:
1. 按质心极角升序得到环绕环;
2. 起点 = x+y 最小者(并列时取 y 最小/最靠上);
3. 沿极角升序依次标 TL,TR,BR,BL。
"""
import numpy as np


def order_corners(pts):
    """返回 4 个索引, 使 pts[order] 为 TL,TR,BR,BL"""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError(f"pts shape must be (4,2), got {pts.shape}")
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    cyc = np.argsort(ang)  # 极角升序
    s_cyc = pts[cyc, 0] + pts[cyc, 1]
    smin = s_cyc.min()
    cand = np.where(s_cyc <= smin + 1e-9)[0]
    start = int(cand[np.argmin(pts[cyc[cand], 1])])  # 并列取最靠上
    return np.array([cyc[(start + k) % 4] for k in range(4)])


def reorder_corners(pts):
    """把任意顺序的 4 点重排为 TL,TR,BR,BL"""
    pts = np.asarray(pts, dtype=np.float64)
    return pts[order_corners(pts)]
