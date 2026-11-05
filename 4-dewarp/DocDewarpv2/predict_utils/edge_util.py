import numpy as np
import cv2
import torch
def create_document_mask(edge_img, img_size, kernel_size=5, min_contour_area=1000):
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


def or_with_edge(mask_img, edge_img, threshold=127):
    """
    将 mask_img 与 edge_img 逐像素逻辑或，得到合并后的有效掩码。

    Args:
        mask_img: 被文档掩码过滤后的图像，(H, W)，uint8
        edge_img: 原始边缘图像，(H, W)，uint8
        threshold: 二值化阈值，默认 127

    Returns:
        合并后的有效掩码，(H, W)
    """
    _, mask_bin = cv2.threshold(mask_img, threshold, 255, cv2.THRESH_BINARY)
    _, edge_bin = cv2.threshold(edge_img, threshold, 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_or(mask_bin, edge_bin)
    return combined

def filter_by_document_mask(img, doc_mask):
    """
    根据文档掩码过滤图像：
    - 文档区域内（doc_mask > 0）：像素保持不变
    - 文档区域外（doc_mask == 0）：像素值置为 0

    Args:
        img: 输入图像，形状 (H, W) 或 (H, W, C)，uint8 类型
        doc_mask: 文档掩码，形状 (H, W)，uint8 类型（0 或 255）

    Returns:
        过滤后的图像，uint8 类型，与输入同形状
    """
    result = img.copy()
    # doc_mask 中 255 表示文档区域，0 表示背景
    doc_region = doc_mask > 0
    if result.ndim == 2:
        result[~doc_region] = 0
    else:
        result[~doc_region] = 0
    return result


def fit_document_quad(edge_img, threshold=127, use_contour=True):
    """由文档边缘掩码拟合文档的四个角点（顺序 TL, TR, BR, BL）。

    直接用 minAreaRect 得到的是「包住文档的旋转矩形」，它的角点通常落在文档之外：
    用它做单应矫正后，文档真实边界会落到图像内部，四周反而留下黑色三角区。
    因此这里先用 minAreaRect 定出四个角的大致方位，再把每个角点吸附到文档边界上
    「沿该对角方向投影最远」的那个边缘像素（极值点法）——得到的就是文档真正的四个
    角点，用它做单应矫正才能让文档四边精确贴合图像四边。

    Args:
        edge_img: 文档边缘掩码，(H, W) 或 (H, W, C)，边缘像素为 255
        threshold: 二值化阈值，默认 127
        use_contour: True 时只取文档实心掩码的最大外轮廓作为候选边缘像素，
                     以抑制内部折痕线与噪点的干扰

    Returns:
        quad: float32 (4, 2)，顺序为 TL, TR, BR, BL；边缘信息不足时返回 None
    """
    edge = np.asarray(edge_img)
    if edge.ndim == 3:
        edge = edge[..., 0]
    ys, xs = np.nonzero(edge > threshold)
    if xs.size < 4:
        return None

    pts = np.stack([xs, ys], 1).astype(np.float32)
    if use_contour:
        doc_mask, _ = create_document_mask(edge, max(edge.shape[:2]))
        contours, _ = cv2.findContours(doc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) >= 4:
                pts = cnt.reshape(-1, 2).astype(np.float32)

    # minAreaRect 先给出四个角的方位（TL, TR, BR, BL），后续按其一一对齐地吸附
    quad0 = _order_quad_tl_tr_br_bl(_fit_min_quad(pts))
    center = pts.mean(axis=0)
    quad = []
    for corner in quad0:
        d = corner - center
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            quad.append(corner)
            continue
        proj = pts.dot(d / n)            # 边缘像素在该对角方向上的投影
        quad.append(pts[int(np.argmax(proj))])
    return np.asarray(quad, dtype=np.float32)


def _quad_output_size(quad, fallback_w, fallback_h):
    """按文档四边形两条边的长度推算输出宽高（保持文档长宽比，总面积与原图相当）。

    Args:
        quad: (4, 2)，顺序 TL, TR, BR, BL
        fallback_w, fallback_h: 推算失败时使用的尺寸

    Returns:
        (out_w, out_h)
    """
    w = float(np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) * 0.5
    h = float(np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) * 0.5
    if not np.isfinite(w) or not np.isfinite(h) or w < 1.0 or h < 1.0:
        return int(fallback_w), int(fallback_h)
    # 面积保持与原图相当，避免输出分辨率与原图相差过大
    scale = float(np.sqrt(float(fallback_w) * float(fallback_h) / max(w * h, 1e-6)))
    out_w = max(16, int(round(w * scale)))
    out_h = max(16, int(round(h * scale)))
    return out_w, out_h


def document_homography(quad, out_w, out_h):
    """求把文档四边形映射到输出矩形的单应矩阵。

    Args:
        quad: (4, 2)，顺序 TL, TR, BR, BL
        out_w, out_h: 输出图像宽 / 高

    Returns:
        H: float64 (3, 3) 单应矩阵
    """
    dst_quad = np.array(
        [[0.0, 0.0], [out_w - 1.0, 0.0], [out_w - 1.0, out_h - 1.0], [0.0, out_h - 1.0]],
        dtype=np.float32)
    return cv2.getPerspectiveTransform(np.asarray(quad, dtype=np.float32), dst_quad)


def homography_rectify_document(img, edge_img, preserve_ar=False, threshold=127,
                                min_quad_area=100.0, edge_thickness=2):
    """用文档边缘掩码对输入图像做第一次矫正（单应性变换）。

    文档的倾斜与透视属于纯射影形变，一次单应就能把它精确摆正（文档四条边贴合
    输出图像的四条边）；剩下的纸张弯曲才是需要网络学习的曲面形变。因此这里先用
    单应把「透视」这一层去掉，再把矫正结果交给后续流程，网络只需专注 dewarp。

    流程：边缘像素集合 -> minAreaRect 拟合文档四边形 -> 与输出矩形求单应 H ->
    warpPerspective 矫正图像，并对文档边缘掩码施加同一个 H（_edge 与图像严格对齐，
    可继续作为后续流程的边缘条件）。

    边缘掩码的同步变换不走 warpPerspective，而是「变换轮廓点 + 重画」：
    原始边缘是 2~4 px 的细线，warpPerspective 是后向映射 + 最近邻取整，细线会被
    四舍五入打散成断续的虚线；同时拟合出的四边形是直边（弦），真实纸边是弯的，
    凸到弦外侧的那部分会被映射出画布（x<0 / x>out_w-1）而直接裁掉——实测会出现
    「左右两条边整条消失」甚至四条边全丢。直接对轮廓点做 perspectiveTransform 是
    纯几何变换、没有重采样；再把越界的点裁剪回画布边界后重画，就能保证四条边都
    落在画布最外圈。

    Args:
        img: uint8 图像，(H, W, 3) 或 (H, W)
        edge_img: 与 img 同尺寸的文档边缘掩码 (H, W)，边缘像素为 255，其余为 0
        preserve_ar: True 时按文档四边形两条边的长度推算输出宽高（保持文档长宽比，
                     面积与原图相当）；False 时输出尺寸与原图一致（默认）
        threshold: 二值化阈值，默认 127
        min_quad_area: 四边形面积阈值，小于该值认为拟合不可靠，放弃矫正
        edge_thickness: 重画矫正后边缘线的线宽（像素），默认 2

    Returns:
        rect_img, rect_edge, H, quad：矫正后的图像 / 同步变换后的边缘掩码 / 单应矩阵 /
        拟合得到的文档四边形。无法执行矫正时返回 None, 原 edge_img, None, None
    """
    img = np.asarray(img)
    quad = fit_document_quad(edge_img, threshold)
    if quad is None or _quad_area(quad) < float(min_quad_area):
        return None, edge_img, None, None

    h0, w0 = img.shape[:2]
    if preserve_ar:
        out_w, out_h = _quad_output_size(quad, w0, h0)
    else:
        out_w, out_h = int(w0), int(h0)

    H = document_homography(quad, out_w, out_h)
    rect_img = cv2.warpPerspective(img, H, (out_w, out_h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    rect_edge = _warp_edge_by_contour(edge_img, H, out_w, out_h,
                                      threshold=threshold,
                                      thickness=edge_thickness)

    return rect_img, rect_edge, H, quad


def _warp_edge_by_contour(edge_img, H, out_w, out_h, threshold=127, thickness=2):
    """用「轮廓点透视变换 + 裁剪到画布 + 重画」的方式同步变换文档边缘掩码。

    相比直接 warpPerspective 边缘细线（会被最近邻取整打散、且越界部分被裁掉），
    这里对轮廓点做纯几何变换，没有任何重采样损失。

    Args:
        edge_img: 文档边缘掩码 (H, W)，边缘像素为 255
        H: 3x3 单应矩阵（与原图 -> 矫正图的 image warp 用同一个）
        out_w, out_h: 输出画布宽 / 高
        threshold: 二值化阈值，默认 127
        thickness: 重画边缘线的线宽，默认 2

    Returns:
        uint8 (out_h, out_w) 边缘掩码，边缘为 255，其余为 0
    """
    edge = np.asarray(edge_img)
    if edge.ndim == 3:
        edge = edge[..., 0]
    # 边缘线可能是抗锯齿的灰度图，先按阈值二值化，保证轮廓提取只认真正的边缘
    _, edge = cv2.threshold(edge, threshold, 255, cv2.THRESH_BINARY)

    # 1. 由边缘线得到文档实心掩码，再取其最大外轮廓的全部边界点
    doc_mask, _ = create_document_mask(edge, max(out_w, out_h))
    contours, _ = cv2.findContours(doc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((out_h, out_w), dtype=np.uint8)
    cnt = max(contours, key=cv2.contourArea).astype(np.float32).reshape(-1, 1, 2)

    # 2. 纯几何变换：轮廓点直接过单应，无重采样
    cnt_t = cv2.perspectiveTransform(cnt, H.astype(np.float32)).reshape(-1, 2)

    # 3. 落出画布的点裁剪回边界：纸张弯曲会让部分纸边凸到四边形外侧，
    #    不裁剪会被整段丢掉（表现为左右/上下某两条边消失），裁剪后它们被压到
    #    画布最外圈，正好是矫正后文档边缘应有的位置
    cnt_t[:, 0] = np.clip(cnt_t[:, 0], 0.0, float(out_w - 1))
    cnt_t[:, 1] = np.clip(cnt_t[:, 1], 0.0, float(out_h - 1))

    # 4. 重画成二值边缘掩码
    rect_edge = np.zeros((out_h, out_w), dtype=np.uint8)
    cv2.drawContours(rect_edge, [np.round(cnt_t).astype(np.int32)], -1,
                    255, max(1, int(thickness)))
    return rect_edge


def _nearest_edge_index_map(edge_bin):
    """
    为每个像素查找"最近的文档边缘像素"，返回其坐标索引图（最近邻特征变换）。

    Args:
        edge_bin: bool 数组 (H, W)，True 表示文档边缘像素（边缘那一圈）

    Returns:
        ny, nx: int32 数组 (H, W)，(ny[i,j], nx[i,j]) 为距 (i,j) 最近的边缘像素坐标
    """
    # 优先使用 scipy 的精确欧氏距离变换（return_indices 直接给出最近特征点坐标）
    try:
        from scipy import ndimage as ndi
        # distance_transform_edt 计算"非零像素到最近零像素"的距离，
        # 因此传入 ~edge_bin，使背景（零）恰好是边缘像素
        _, (ny, nx) = ndi.distance_transform_edt(~edge_bin, return_indices=True)
        return ny.astype(np.int32), nx.astype(np.int32)
    except ImportError:
        pass

    # 回退实现：OpenCV 带标签的距离变换（DIST_L2 + mask 5，欧氏距离足够精确）
    dist, labels = cv2.distanceTransformWithLabels(
        (~edge_bin).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(edge_bin)
    # 每个边缘像素拥有独立标签，建立 标签 -> 坐标 的查找表
    lut = np.zeros((int(labels.max()) + 1, 2), dtype=np.int32)
    lut[labels[ys, xs], 0] = ys
    lut[labels[ys, xs], 1] = xs
    return lut[labels, 0], lut[labels, 1]


def _fit_min_quad(points):
    """点集的最小外接矩形，4 个角点 (float32, shape (4, 2))"""
    return cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32))).astype(np.float32)


def _order_quad_tl_tr_br_bl(box):
    """将 4 个角点按 TL, TR, BR, BL 顺序排好，规则：以质心为原点的极角排序，
    从最接近左上方向 (-3π/4) 的角点开始。保证两个外接矩形使用同一规则后角点一一对应。"""
    c = box.mean(axis=0)
    ang = np.arctan2(box[:, 1] - c[1], box[:, 0] - c[0])
    d = np.abs(((ang - (-3 * np.pi / 4) + np.pi) % (2 * np.pi)) - np.pi)
    start = int(np.argmin(d))
    order = np.argsort((ang - ang[start]) % (2 * np.pi))
    return box[order]


def _quad_area(box):
    """4 个角点围成的四边形面积（用 shoelace）"""
    x = box[:, 0]; y = box[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def snap_lbl_to_document_edge(lbl, edge_mask, doc_mask=None, threshold=127, inward_offset=2.0):
    """
    用文档边缘掩码矫正 backward map（lbl），使矫正后的文档边缘与图像边缘贴合。

    核心问题：网络预测的 lbl 把矫正图的矩形采样框映射到了原图里一个"轴对齐矩形"
    区域，而真实的文档是一个"旋转四边形"——两者的角会超出文档，出现大块黑边
    或被错误地填成边缘像素的涂抹带。简单地把越界采样点投到最近边缘像素只能消除
    黑边，但会在四周产生一条涂抹的边带（"矫正后文档边缘无法贴合图像边缘"）。

    本函数的做法：
    1. 用 minAreaRect 拟合文档边缘得到文档的四边形四个角点（doc_quad）。
    2. 取 lbl 在矫正图四条边上的采样点对应的原图坐标，拟合"网络当前映射框"的
       四边形（map_quad）。
    3. 计算单应变换 H，使得 map_quad 精确对齐到 doc_quad。
    4. 对全部 lbl 应用该单应变换，把"轴对齐矩形"矫正成"旋转四边形"——矫正图
       的边界精确映射到文档边界，文档边缘与图像边缘贴合。
    5. 对极少数仍然越界的点（单应变换后残留的几何误差）执行"投到最近边缘 + 向
       内偏移"兜底，避免黑边。带 inward_offset 是为了不让采样点压在线上被双线性
       插值掺到外侧被置零的背景像素上。

    Args:
        lbl: torch.Tensor，形状 (1, H, W, 2)，值域 [-1, 1] 的采样网格（align_corners=True）
        edge_mask: 文档边缘掩码，numpy array (h, w) 或 (H, W)，边缘像素为 255，其余为 0
        doc_mask: 文档实心掩码（文档区域内为 255）。为 None 时由 edge_mask 自动填充生成
        threshold: 二值化阈值，默认 127
        inward_offset: 兜底重定向后再往文档内部偏移的像素数（避免双线性插值与
                       背景混合），默认 2.0

    Returns:
        lbl_snapped: 与 lbl 同形状、同 dtype、同 device 的采样网格
        outside_mask: bool numpy array (H, W)，单应+兜底之后仍然落在文档之外的点
                      （正常应全 False，存图即全黑）
        redirect_mask: bool numpy array (H, W)，本次被兜底重定向过的点
    """
    grid = lbl[0].detach().cpu().numpy().astype(np.float32)  # (H, W, 2)
    H, W = grid.shape[:2]
    sx_max = float(max(W - 1, 1))
    sy_max = float(max(H - 1, 1))

    def _prepare(mask):
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        return mask

    edge_mask = _prepare(edge_mask)
    if doc_mask is None:
        doc_mask, _ = create_document_mask(edge_mask, max(H, W))
    else:
        doc_mask = _prepare(doc_mask)

    edge_bin = edge_mask > threshold
    doc_bin = doc_mask > threshold

    def _outside(px_, py_):
        xi_ = np.clip(np.rint(px_), 0, W - 1).astype(np.int32)
        yi_ = np.clip(np.rint(py_), 0, H - 1).astype(np.int32)
        return ~doc_bin[yi_, xi_]

    # 没有边缘信息时保持原样
    if not edge_bin.any():
        z = np.zeros((H, W), dtype=bool)
        return lbl, z, z

    # 归一化网格坐标 -> 原图像素坐标（align_corners=True）
    px = (grid[..., 0] + 1.0) * 0.5 * sx_max
    py = (grid[..., 1] + 1.0) * 0.5 * sy_max

    # ====================== 单应矫正：在网络 lbl 基础上把"映射框"对齐到"文档四边形" ======================
    # 关键：本函数是在【网络预测的 lbl】之上做修正，而不是替换它——网络学到的曲面
    # 去扭曲被完整保留，单应只负责把输出图四边重新对齐到真实文档边界，使文档边缘
    # 贴合图像边界。纯"重定向到最近边缘"只会把越界点压成一条边缘填充带，无法让
    # 边缘贴边界；只有整张网格做一次几何对齐（单应）才能实现目标。
    pre_outside = _outside(px, py)
    eys, exs = np.where(edge_bin)
    doc_quad = _order_quad_tl_tr_br_bl(_fit_min_quad(np.stack([exs, eys], 1)))

    # 取矫正图四条边的 lbl 采样点作为"网络当前映射框"的代表，拟合其外接矩形
    bx = np.concatenate([px[0], px[-1], px[:, 0], px[:, -1]])
    by = np.concatenate([py[0], py[-1], py[:, 0], py[:, -1]])
    map_quad = _order_quad_tl_tr_br_bl(_fit_min_quad(np.stack([bx, by], 1)))

    # 退化保护：四边形面积过小（拟合失效）就直接跳过单应，保留网络原始 lbl
    if _quad_area(map_quad) > 100 and _quad_area(doc_quad) > 100:
        # 单应 G：把网络映射框(map_quad) 对齐到真实文档四边形(doc_quad)，
        # 再作用到全部 (px,py) 上 = 在网络的 lbl 之上叠加这次几何修正。
        G = cv2.getPerspectiveTransform(map_quad.astype(np.float32),
                                         doc_quad.astype(np.float32))
        pts = np.stack([px, py], -1).reshape(-1, 1, 2).astype(np.float32)
        t = cv2.perspectiveTransform(pts, G).reshape(H, W, 2)
        hx, hy = t[..., 0], t[..., 1]
        post_outside = _outside(hx, hy)

        # 采纳米则：本函数真正关心的是"输出图四边是否对到了文档边界"，而不是整图
        # 越界比例。用边框越界比例的改善作为主采纳条件，并加"整体不显著恶化"的
        # 保护，避免因为 doc_quad（minAreaRect）略大于 fill_mask_ 而误拒一个本来
        # 正确的对齐（误拒会让单应失效、只剩边缘填充带 -> 拼接缝）。
        border = np.zeros((H, W), dtype=bool)
        border[0, :] = border[-1, :] = True
        border[:, 0] = border[:, -1] = True
        pre_border_out = (pre_outside & border).mean()
        post_border_out = (post_outside & border).mean()

        if post_border_out <= pre_border_out + 1e-9 and \
           post_outside.mean() <= pre_outside.mean() + 0.02:
            px, py = hx, hy

    # ====================== 兜底：把仍然越界的点投到最近边缘像素 ======================
    redirect_mask = _outside(px, py)
    if redirect_mask.any():
        ny, nx = _nearest_edge_index_map(edge_bin)
        xi = np.clip(np.rint(px), 0, W - 1).astype(np.int32)
        yi = np.clip(np.rint(py), 0, H - 1).astype(np.int32)
        ex = nx[yi, xi].astype(np.float32)
        ey = ny[yi, xi].astype(np.float32)
        if inward_offset > 0:
            dx = ex - px; dy = ey - py
            norm = np.sqrt(dx * dx + dy * dy) + 1e-6
            ox = ex + inward_offset * dx / norm
            oy = ey + inward_offset * dy / norm
            bad = redirect_mask & _outside(ox, oy)
            ex = np.where(bad, ex, ox)
            ey = np.where(bad, ey, oy)
        px = np.where(redirect_mask, ex, px)
        py = np.where(redirect_mask, ey, py)

    # 原图像素坐标 -> 归一化网格坐标
    grid[..., 0] = px / sx_max * 2.0 - 1.0
    grid[..., 1] = py / sy_max * 2.0 - 1.0

    lbl_snapped = torch.from_numpy(grid).unsqueeze(0).to(device=lbl.device, dtype=lbl.dtype)
    outside_mask = _outside(px, py)
    return lbl_snapped, outside_mask, redirect_mask


def _largest_valid_component(valid):
    """保留最大连通域，剔除孤立的噪声有效点。

    Args:
        valid: bool 数组 (H, W)

    Returns:
        bool 数组 (H, W)，仅保留最大的连通区域
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        valid.astype(np.uint8), connectivity=8)
    if num <= 1:
        return valid
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _row_spans(valid):
    """逐行求有效区间 [最左列, 最右列]；整行都无效时记为 (-1, -1)。

    Args:
        valid: bool 数组 (H, W)

    Returns:
        left, right: int32 数组 (H,)
    """
    H, W = valid.shape[:2]
    left = np.full(H, -1, dtype=np.int32)
    right = np.full(H, -1, dtype=np.int32)
    rows = np.flatnonzero(valid.any(axis=1))
    if rows.size:
        sub = valid[rows]
        left[rows] = np.argmax(sub, axis=1)
        right[rows] = (W - 1) - np.argmax(sub[:, ::-1], axis=1)
    return left, right


def _interp_and_smooth_span(span, length, smooth):
    """把无效项（值为 -1）用相邻有效值线性插值补齐，再做滑动平均平滑。

    Args:
        span: int 数组，值为 -1 的位置表示无效
        length: 数组长度，用于限制平滑窗口
        smooth: 平滑窗口（会被调整为奇数）

    Returns:
        float32 数组，或 None（全部无效时）
    """
    v = span.astype(np.float32)
    ok = v >= 0
    if not ok.any():
        return None
    x = np.arange(v.size, dtype=np.float32)
    v = np.interp(x, x[ok], v[ok]).astype(np.float32)
    k = int(smooth) | 1
    k = max(1, min(k, max(1, (length // 4) | 1)))
    if k > 1:
        kernel = np.ones(k, dtype=np.float32) / float(k)
        v = np.convolve(np.pad(v, k // 2, mode='edge'), kernel, mode='valid')
    return v.astype(np.float32)


def prune_samples_outside_document(bm0, bm1, edge_mask, doc_mask=None, threshold=127,
                                   smooth=9, inward_offset=1, min_valid_ratio=0.2,
                                   return_debug=False):
    """
    删除落在文档边缘之外的采样点，并把剩下的有效采样区域重新参数化（拉伸）到整幅输出图，
    使矫正后的文档边缘与图像边缘完全贴合。

    动机：网络预测的后向形变场 (bm0, bm1) 把输出图（规则矩形）逐像素映射回原图，映射得到
    的区域 R 与真实文档区域 Q（由边缘掩码围成的四边形）通常并不重合：R 超出 Q 的那些采样
    点取到的是被置零的背景，这就是矫正结果里"文档边缘贴不到图像边缘"的黑边来源。
    理想情况下矫正图的边界就是文档的边界，因此：

    1. 【删除】逐像素判断采样点是否落在文档内，落在文档边缘之外的采样点一律标记为无效，
       不再参与采样（越界点被整片删除，而不是像重定向那样被压到边缘上形成涂抹带）。
    2. 【重参数化】用剩下有效点的边界把整张网格重新参数化：
          · 水平方向：第 i 行只保留 [最左有效列, 最右有效列]，再线性拉伸到输出列 [0, W-1]；
          · 垂直方向：在该坐标系下逐输出列求 [最上有效行, 最下有效行]，再拉伸到 [0, H-1]。
       由于有效区域的边界 F^{-1}(∂Q) 正好映射回文档边缘，输出图的第一/最后一列、第一/最后
       一行采到的就是文档左/右/上/下边缘上的点，即"文档边缘 == 图像边界"。
    3. 若有效区域本来就是整张网格（网络预测已贴合），两次拉伸退化为恒等映射，不做任何改动。

    Args:
        bm0, bm1: (H, W) float32 后向形变场（x / y 方向），取值归一化到 [-1, 1]，
                  分辨率与原图一致（H = img_h, W = img_w）
        edge_mask: 文档边缘掩码 (H, W) uint8，边缘像素为 255，其余为 0
        doc_mask: 文档实心掩码（文档区域内为 255）。为 None 时由 edge_mask 自动填充生成
        threshold: 二值化阈值，默认 127
        smooth: 行/列边界的平滑窗口（会被调整为奇数），保证重参数化后的网格连续、不产生锯齿
        inward_offset: 判定时把文档掩码向内腐蚀的像素数（默认 1）。因为双线性插值会取到
                       邻近像素，取 1 可让输出图最外圈采样点落在文档边缘内侧 1 像素处，
                       既保证贴合又不会掺入已被置零的背景；设为 0 则严格以文档边缘为界
        min_valid_ratio: 有效采样点比例低于该值时认为边缘掩码不可靠，直接原样返回
        return_debug: 为 True 时额外返回调试用掩码

    Returns:
        bm0_new, bm1_new: 与输入同形状、同 dtype 的形变场
        valid:      bool (H, W)，未被删除的有效采样点（仅 return_debug=True 时返回）
        outside:    bool (H, W)，处理后仍落在文档之外的采样点，
                    正常情况下应接近全 False（仅 return_debug=True 时返回）
    """
    bm0 = np.asarray(bm0)
    bm1 = np.asarray(bm1)
    H, W = bm0.shape[:2]
    sx_max = float(max(W - 1, 1))
    sy_max = float(max(H - 1, 1))
    zeros = np.zeros((H, W), dtype=bool)

    def _prepare(mask):
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        return mask

    edge_mask = _prepare(edge_mask)
    if doc_mask is None:
        doc_mask, _ = create_document_mask(edge_mask, max(H, W))
    else:
        doc_mask = _prepare(doc_mask)

    edge_bin = edge_mask > threshold
    doc_bin = doc_mask > threshold
    # 判定用的掩码：向内腐蚀，避免最外圈采样点因双线性插值掺到文档外侧被置零的背景
    doc_test = doc_bin
    if inward_offset and int(inward_offset) > 0:
        k = 2 * int(inward_offset) + 1
        doc_test = cv2.erode(doc_bin.astype(np.uint8), np.ones((k, k), np.uint8),
                             iterations=1) > 0
        if not doc_test.any():
            doc_test = doc_bin

    def _to_pixel(fx, fy):
        return (fx.astype(np.float32) + 1.0) * 0.5 * sx_max, \
               (fy.astype(np.float32) + 1.0) * 0.5 * sy_max

    def _outside_of(fx, fy):
        px_, py_ = _to_pixel(fx, fy)
        xi_ = np.clip(np.rint(px_), 0, W - 1).astype(np.int32)
        yi_ = np.clip(np.rint(py_), 0, H - 1).astype(np.int32)
        return ~doc_bin[yi_, xi_] | (px_ < 0) | (px_ > W - 1) | (py_ < 0) | (py_ > H - 1)

    def _bail_out(valid_):
        if return_debug:
            return bm0, bm1, valid_, zeros
        return bm0, bm1

    # 没有边缘信息时保持原样
    if not edge_bin.any() or not doc_bin.any():
        return _bail_out(zeros)

    # ====================== 1) 删除：落在文档边缘之外的采样点 ======================
    px, py = _to_pixel(bm0, bm1)
    xi = np.clip(np.rint(px), 0, W - 1).astype(np.int32)
    yi = np.clip(np.rint(py), 0, H - 1).astype(np.int32)
    in_img = (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
    valid = in_img & doc_test[yi, xi]
    valid = _largest_valid_component(valid)

    # 边缘掩码不可靠（有效点太少）时不改动形变场
    if valid.mean() < min_valid_ratio:
        return _bail_out(valid)

    # 被删除的点先用"最近的有效采样点"填坑，仅用于保证后续双线性插值不取到垃圾坐标；
    # 重参数化之后输出图只会采样到有效区域内部，这些点本身不会再被采到。
    if (~valid).any():
        ny, nx = _nearest_edge_index_map(valid)
        bm0_f = np.where(valid, bm0, bm0[ny, nx])
        bm1_f = np.where(valid, bm1, bm1[ny, nx])
    else:
        bm0_f, bm1_f = bm0, bm1

    # ====================== 2) 重参数化：把有效区域的边界拉伸到输出图四边 ======================
    # 2.1 水平方向：第 i 行的 [最左, 最右] 有效列 -> 输出列 [0, W-1]
    left, right = _row_spans(valid)
    L = _interp_and_smooth_span(left, H, smooth)
    R = _interp_and_smooth_span(right, H, smooth)
    if L is None or R is None or float(np.median(R - L)) < 2.0:
        return _bail_out(valid)
    u = np.arange(W, dtype=np.float32) / float(max(W - 1, 1))
    col_map = L[:, None] + (R - L)[:, None] * u[None, :]     # (H, W)：原始行 -> 采样列
    col_map = np.clip(col_map, 0.0, float(W - 1))

    # 2.2 垂直方向：在水平拉伸后的坐标系里逐输出列求 [最上, 最下] 有效行 -> 输出行 [0, H-1]
    j_int = np.clip(np.rint(col_map), 0, W - 1).astype(np.int32)
    valid_col = valid[np.arange(H)[:, None], j_int]          # (H, W)：行=原始行，列=输出列
    top, bottom = _row_spans(valid_col.T)
    T = _interp_and_smooth_span(top, W, smooth)
    B = _interp_and_smooth_span(bottom, W, smooth)
    if T is None or B is None or float(np.median(B - T)) < 2.0:
        return _bail_out(valid)
    vv = np.arange(H, dtype=np.float32) / float(max(H - 1, 1))
    row_map = T[None, :] + (B - T)[None, :] * vv[:, None]    # (H, W)：输出行 -> 采样行
    row_map = np.clip(row_map, 0.0, float(H - 1))

    # 2.3 组合两次拉伸：输出像素 (y', x') 的采样列取 col_map 在第 row_map(y', x') 行处的值
    i0 = np.clip(np.floor(row_map).astype(np.int32), 0, H - 1)
    i1 = np.clip(i0 + 1, 0, H - 1)
    frac = (row_map - i0).astype(np.float32)
    cols = np.arange(W, dtype=np.int32)[None, :]
    map_x = (col_map[i0, cols] * (1.0 - frac) + col_map[i1, cols] * frac).astype(np.float32)
    map_y = row_map.astype(np.float32)

    bm0_new = cv2.remap(bm0_f, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE).astype(bm0.dtype)
    bm1_new = cv2.remap(bm1_f, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE).astype(bm1.dtype)

    if return_debug:
        return bm0_new, bm1_new, valid, _outside_of(bm0_new, bm1_new)
    return bm0_new, bm1_new


# ==============================================================================
#  把 bm 的采样区域（bm 的像）重参数化到整幅输入图
# ==============================================================================

def _bm_to_pixel(bm0, bm1):
    """归一化 [-1, 1] 的后向形变场 -> 原图像素坐标（align_corners=True 约定）。"""
    H, W = bm0.shape[:2]
    sx = float(max(W - 1, 1))
    sy = float(max(H - 1, 1))
    px = (np.asarray(bm0, dtype=np.float32) + 1.0) * 0.5 * sx
    py = (np.asarray(bm1, dtype=np.float32) + 1.0) * 0.5 * sy
    return px, py, sx, sy


def _pixel_to_bm(px, py, sx, sy, dtype):
    """原图像素坐标 -> 归一化 [-1, 1]。"""
    bm0 = (px / sx * 2.0 - 1.0).astype(dtype)
    bm1 = (py / sy * 2.0 - 1.0).astype(dtype)
    return bm0, bm1


def _support_range(px, py, quantile):
    """映射区域在 x / y 两个轴上的支撑区间 [lo, hi]。

    quantile 的单位是 %：取 [q, 100-q] 分位数而不是 min/max，
    是为了让少量因纸张边缘弯曲而凸出的离群采样点不至于把整张网格压缩。
    """
    if quantile and quantile > 0:
        q = float(quantile)
        x0, x1 = np.percentile(px, q), np.percentile(px, 100.0 - q)
        y0, y1 = np.percentile(py, q), np.percentile(py, 100.0 - q)
    else:
        x0, x1 = px.min(), px.max()
        y0, y1 = py.min(), py.max()
    return float(x0), float(x1), float(y0), float(y1)


def _bm_corner_points(px, py, k=5):
    """输出网格四个角映射回原图后的源点，顺序 TL, TR, BR, BL。

    每个角取 k×k 邻域的均值，抑制单个离群像素把整张单应带偏。
    """
    H, W = px.shape[:2]
    k = max(1, min(int(k), H, W))

    def _blk(a, bottom, right):
        r0 = H - k if bottom else 0
        c0 = W - k if right else 0
        return float(a[r0:r0 + k, c0:c0 + k].mean())

    return np.array([
        [_blk(px, 0, 0), _blk(py, 0, 0)],          # TL
        [_blk(px, 0, 1), _blk(py, 0, 1)],          # TR
        [_blk(px, 1, 1), _blk(py, 1, 1)],          # BR
        [_blk(px, 1, 0), _blk(py, 1, 0)],          # BL
    ], dtype=np.float32)


def bm_coverage(bm0, bm1, quantile=0.5):
    """形变场采样区域占整幅输入图的比例 (cov_x, cov_y)，1.0 表示正好铺满。"""
    px, py, sx, sy = _bm_to_pixel(bm0, bm1)
    x0, x1, y0, y1 = _support_range(px, py, quantile)
    return (x1 - x0) / sx, (y1 - y0) / sy


def fit_bm_to_image(bm0, bm1, mode='homography', quantile=0.5, min_cover=0.99,
                    corner_k=5, clamp=True, min_area_ratio=0.05, max_expand=5.0,
                    return_debug=False):
    """把后向形变场的采样区域（bm 的像）拉伸到整幅输入图，使 bm 覆盖全图。

    问题：网络输出的 bm 把"输出网格"映射回原图后，其像 R 通常只是原图中间的一块
    （比整幅图小一圈）。此时原图最外圈的像素永远不会被采到，画出来的形变网格铺不满
    输入图，矫正结果也会丢掉文档最外圈的内容。

    做法（纯几何后处理，不动网络、不改标签）：对 bm 的取值整体施加一个全局变换 G，
    使 G(R) 精确等于整幅输入图矩形：

    1. 【单应，可选】取输出网格四个角映射回原图的源点作为 R 的四个角点，求把它们
       顶到输入图四个角的单应 G 并作用到全部采样点上。这一步负责消掉旋转 / 透视 /
       非均匀的整体偏移；mode='bbox' 时跳过，只做第 2 步。
    2. 【逐轴线性拉伸，必做】在 G 作用后重新统计 R 的支撑区间 [x0,x1]×[y0,y1]，
       线性映射到 [0,W-1]×[0,H-1]。这一步保证"铺满"是精确的（单应只能把四个角
       顶到角上，弯曲的纸边中间会凸出或凹进，残差由这一层吃掉）。
    3. 若 R 本来就铺满（两个方向覆盖率都 ≥ min_cover），直接原样返回，不做无谓形变。

    注意：这里拉伸的是【采样区域】，不是输出图——输出图尺寸不变，变的是"输出图每个
    像素从原图的哪一块取像素"，因此原来漏掉的文档最外圈内容会被补回来。

    Args:
        bm0, bm1: (H, W) float32 后向形变场（x / y），取值归一化到 [-1, 1]，
                  分辨率与原图一致（H = img_h, W = img_w）
        mode: 'homography'（默认，先单应再逐轴拉伸）/ 'bbox'（只逐轴拉伸）/ 'off'
        quantile: 支撑区间统计时两端各忽略的比例（%），默认 0.5。用于抑制纸边弯曲
                  造成的离群凸出点；设 0 则严格用 min/max
        min_cover: 该覆盖率以上认为已经铺满，直接返回（默认 0.99）
        corner_k: 求单应时角点邻域的大小（默认 5×5 取均值）
        clamp: True 时把最终采样坐标夹到图像范围内，避免 grid_sample 采到界外产生黑边
        min_area_ratio: R 的角点四边形面积小于整幅图的该比例时认为退化，跳过单应
        max_expand: 单应后支撑区间超过图像该倍数认为病态，回退到不做单应
        return_debug: 为 True 时额外返回调试信息

    Returns:
        bm0_new, bm1_new: 与输入同形状、同 dtype 的形变场（仍为 [-1, 1]）
        info (dict, 仅 return_debug=True): cov_before / cov_after / used_homo
    """
    bm0 = np.asarray(bm0)
    bm1 = np.asarray(bm1)
    H, W = bm0.shape[:2]
    px, py, sx, sy = _bm_to_pixel(bm0, bm1)

    def _cov(pxx, pyy):
        x0, x1, y0, y1 = _support_range(pxx, pyy, quantile)
        return (x1 - x0) / sx, (y1 - y0) / sy

    cov_before = _cov(px, py)
    used_homo = False

    # 已经铺满就没必要动
    if mode != 'off' and min(cov_before) < min_cover:
        # ---------- 1) 单应：把 R 的四个角顶到输入图四个角 ----------
        if mode in ('homography', 'homo', 'auto'):
            src = _bm_corner_points(px, py, corner_k)
            dst = np.array([[0.0, 0.0], [sx, 0.0], [sx, sy], [0.0, sy]],
                           dtype=np.float32)
            if _quad_area(src) > float(min_area_ratio) * sx * sy:
                try:
                    G = cv2.getPerspectiveTransform(src, dst)
                    pts = np.stack([px, py], -1).reshape(-1, 1, 2)
                    t = cv2.perspectiveTransform(pts, G).reshape(H, W, 2)
                    if np.isfinite(t).all():
                        tx0, tx1, ty0, ty1 = _support_range(t[..., 0], t[..., 1], quantile)
                        # 病态四点会把网格顶飞，超过阈值就当单应没算对，退回纯拉伸
                        if (tx1 - tx0) < max_expand * sx and (ty1 - ty0) < max_expand * sy:
                            px, py = t[..., 0], t[..., 1]
                            used_homo = True
                except cv2.error:
                    pass

        # ---------- 2) 逐轴线性拉伸：支撑区间精确映射到 [0, W-1] × [0, H-1] ----------
        x0, x1, y0, y1 = _support_range(px, py, quantile)
        span_x = max(x1 - x0, 1e-6)
        span_y = max(y1 - y0, 1e-6)
        px = (px - x0) * (sx / span_x)
        py = (py - y0) * (sy / span_y)

        if clamp:
            px = np.clip(px, 0.0, sx)
            py = np.clip(py, 0.0, sy)

    bm0_new, bm1_new = _pixel_to_bm(px, py, sx, sy, bm0.dtype)

    if return_debug:
        info = {'cov_before': cov_before, 'cov_after': _cov(px, py),
                'used_homo': used_homo}
        return bm0_new, bm1_new, info
    return bm0_new, bm1_new
