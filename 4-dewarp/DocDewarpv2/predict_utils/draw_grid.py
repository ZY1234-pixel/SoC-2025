import cv2
import numpy as np
def sample_grid_from_bm(bm0, bm1, grid_size):
    """把矫正图上的均匀网格顶点，通过后向形变场映射回原图坐标。

    bm0 / bm1: (H, W) 的 float32 后向映射（x / y 方向），取值归一化到 [-1, 1]
    返回: (gx, gy)，形状均为 (grid_size + 1, grid_size + 1)，为网格顶点在原图上的像素坐标
    """
    h, w = bm0.shape[:2]
    u = np.linspace(-1., 1., grid_size + 1, dtype=np.float32)
    v = np.linspace(-1., 1., grid_size + 1, dtype=np.float32)
    cols, rows = np.meshgrid(u, v)                      # 矫正图（规则网格）上的归一化位置
    mx = (cols + 1.) * .5 * (w - 1)                     # 后向映射图的采样坐标（像素索引）
    my = (rows + 1.) * .5 * (h - 1)
    sx = cv2.remap(bm0, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    sy = cv2.remap(bm1, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gx = (sx + 1.) * .5 * (w - 1)                       # 归一化坐标 -> 原图像素坐标
    gy = (sy + 1.) * .5 * (h - 1)
    return gx, gy


def draw_deformation_grid(img, bm0, bm1, grid_size=20, thickness=2,
                          line_color=(0, 255, 255), point_color=(255, 0, 0), point_radius=3):
    """把形变场以网格形式叠加到原图上（img 为 RGB uint8 的 numpy 数组）。

    画的是「矫正图上的规则网格」被后向映射拉回原图后的曲线网格：
    若形变场正确，每个网格单元都会贴合文档中对应的一小块区域，且网格会随纸张弯曲而弯曲。
    """
    gx, gy = sample_grid_from_bm(bm0, bm1, grid_size)
    pts = np.stack([np.round(gx), np.round(gy)], axis=-1).astype(np.int32)  # (n, n, 2)

    canvas = img.copy()
    n = grid_size + 1
    rows = [np.ascontiguousarray(pts[i]) for i in range(n)]      # 第 i 条横线
    cols = [np.ascontiguousarray(pts[:, i]) for i in range(n)]    # 第 i 条竖线
    # 先画一圈黑色描边，保证网格在任何底纹上都看得清
    for line in rows + cols:
        cv2.polylines(canvas, [line], False, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    for line in rows + cols:
        cv2.polylines(canvas, [line], False, line_color, thickness, cv2.LINE_AA)
    if point_radius > 0:
        for p in pts.reshape(-1, 2):
            cv2.circle(canvas, (int(p[0]), int(p[1])), point_radius, point_color, -1, cv2.LINE_AA)
    return canvas
