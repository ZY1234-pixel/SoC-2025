"""保护式接缝组装：受限曲面对齐，拟合不足时回退梯度域方案。

两种组装器使用同一批网络预测，不额外运行模型，也不使用GT或原图调色。
"""
from __future__ import annotations

import hashlib
import time
import numpy as np

from tiling import TileConfig, infer_tiled, tile_positions, cosine_vector, _axis_window

def axis_basis(coordinates):
    """计算归一化坐标的一维二次基：常数项、一次项和二次Legendre基。"""
    z = np.asarray(coordinates, dtype=np.float64)
    return np.stack([np.ones_like(z), z, 0.5 * (3 * z * z - 1)])



TERMS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))



class OverlapSurface:
    """根据相邻块的预测差异，联合拟合平滑RGB校正曲面。

    每块、每个颜色通道最多使用6项二次基。拟合只读取预测块的重叠区域，
    不使用原图颜色或GT做匹配；校正幅度受限，并移除整图的多项式趋势。
    """

    def __init__(
        self,
        shape,
        config,
        degree=2,
        reduction=8,
        max_shift=0.05,
        native_diagnostics=False,
    ):
        """建立块网格、归一化融合窗和重叠约束的缓存。

        reduction=8用于压缩拟合数据，max_shift=0.05约束0～1颜色范围中的修正幅度。
        previous和left只保留下一批相邻比较需要的重叠条带。
        """
        if degree not in (1, 2):
            raise ValueError(degree)
        self.h, self.w = shape
        self.config = config
        self.degree = degree
        self.terms = TERMS[:3] if degree == 1 else TERMS
        self.k = len(self.terms)
        self.r = reduction
        self.max_shift = max_shift
        self.ys = tile_positions(self.h, config.tile, config.step)
        self.xs = tile_positions(self.w, config.tile, config.step)
        self.n = len(self.ys) * len(self.xs)
        self.index = 0
        self.edges = []
        self.native_diagnostics = native_diagnostics
        self.native_edges = []
        self.previous = [None] * len(self.xs)
        self.left = None
        vector = cosine_vector(config.tile, config.overlap)

        def windows(starts, length):
            """把各块余弦窗归一化到每个像素权重之和为1，用于全图趋势计算。"""
            values = [
                _axis_window(vector, start, length, config.overlap).astype(np.float64)
                for start in starts
            ]
            total = np.zeros(length, np.float64)
            for start, value in zip(starts, values):
                total[start : start + len(value)] += value
            assert np.all(total > 0)
            return [
                value / total[start : start + len(value)]
                for start, value in zip(starts, values)
            ]

        self.x_windows = windows(self.xs, self.w)
        self.y_windows = windows(self.ys, self.h)

    def pooled(self, image):
        """按小网格求均值并保留有效像素数，边缘不足一格时也按真实数量计权。"""
        h, w = image.shape[:2]
        r = self.r
        values = np.zeros(((h + r - 1) // r, (w + r - 1) // r, 3), np.float64)
        counts = np.zeros(values.shape[:2], np.float64)
        for dy in range(r):
            for dx in range(r):
                part = image[dy::r, dx::r]
                values[: len(part), : part.shape[1]] += part
                counts[: len(part), : part.shape[1]] += 1
        return (values / counts[..., None], counts)

    def pooled_basis(self, y, x, h, w, origin_y, origin_x):
        """计算重叠区域内各网格的平均多项式基。

        对二次项保留格内方差，不直接用网格中心代替整格积分。
        """

        def moments(start, length, origin):
            """将像素格的均值和方差转换为当前块局部坐标中的多项式矩。"""
            starts = np.arange(0, length, self.r)
            counts = np.minimum(self.r, length - starts)
            mean = start + starts + counts / 2 - origin
            u = 2 * mean / self.config.tile - 1
            variance = (counts * counts - 1) / (3 * self.config.tile * self.config.tile)
            return np.stack(
                [np.ones_like(u), u, 0.5 * (3 * (u * u + variance) - 1)], axis=-1
            )

        bx = moments(x, w, origin_x)
        by = moments(y, h, origin_y)
        return np.stack(
            [by[:, iy, None] * bx[None, :, ix] for ix, iy in self.terms], axis=-1
        ).reshape(-1, self.k)

    def edge(self, i, j, difference, box, other_origin):
        """记录一对相邻块的颜色差、局部基矩阵和像素权重，形成联合拟合约束。"""
        y, x, h, w = box
        row, col = divmod(i, len(self.xs))
        pooled, counts = self.pooled(difference)
        a = self.pooled_basis(y, x, h, w, self.ys[row], self.xs[col])
        b = self.pooled_basis(y, x, h, w, *other_origin)
        self.edges.append(
            (
                i,
                j,
                a,
                b,
                pooled.reshape(-1, 3),
                counts.ravel() / (self.config.overlap * self.config.tile),
            )
        )
        if self.native_diagnostics:
            self.native_edges.append((i, j, box, difference.copy()))

    def observe(self, prediction):
        """按与推理完全一致的行列顺序，收集当前预测块与左、上邻居的重叠差。

        只取去掉halo后的有效中心；不缓存全部512像素预测，避免内存随块数增长。
        """
        row, col = divmod(self.index, len(self.xs))
        y, x = (self.ys[row], self.xs[col])
        c = self.config
        h, w = (min(c.tile, self.h - y), min(c.tile, self.w - x))
        center = prediction[:, c.halo : c.halo + h, c.halo : c.halo + w].transpose(
            1, 2, 0
        )
        if col:
            width = min(c.overlap, w)
            self.edge(
                self.index,
                self.index - 1,
                center[:, :width] - self.left[:h, :width],
                (y, x, h, width),
                (y, self.xs[col - 1]),
            )
        if row:
            height = min(c.overlap, h)
            self.edge(
                self.index,
                self.index - len(self.xs),
                center[:height] - self.previous[col][:height, :w],
                (y, x, height, w),
                (self.ys[row - 1], x),
            )
        if col + 1 < len(self.xs):
            self.left = center[:, c.step : c.step + c.overlap].copy()
        if row + 1 < len(self.ys):
            self.previous[col] = center[c.step : c.step + c.overlap].copy()
        self.index += 1

    def detrend(self, coefficients):
        """移除整张校正图的低阶多项式趋势，保护原重建的全局色彩和照明走势。

        在原生像素网格上计算精确可分离矩，求出整图投影后转回每块局部坐标。
        这约束的是最终裁剪、量化之前的校正场，不代表能修复模型本身的色偏。
        """
        gx = axis_basis(2 * (np.arange(self.w) + 0.5) / self.w - 1)
        gy = axis_basis(2 * (np.arange(self.h) + 0.5) / self.h - 1)
        gram = np.empty((self.k, self.k), np.float64)
        for a, (ax, ay) in enumerate(self.terms):
            for b, (bx, by) in enumerate(self.terms):
                gram[a, b] = np.dot(gx[ax], gx[bx]) * np.dot(gy[ay], gy[by])
        mx = []
        my = []
        for x, window in zip(self.xs, self.x_windows):
            local = axis_basis(
                2 * (np.arange(len(window)) + 0.5) / self.config.tile - 1
            )
            mx.append(local * window[None] @ gx[:, x : x + len(window)].T)
        for y, window in zip(self.ys, self.y_windows):
            local = axis_basis(
                2 * (np.arange(len(window)) + 0.5) / self.config.tile - 1
            )
            my.append(local * window[None] @ gy[:, y : y + len(window)].T)
        rhs = np.zeros((self.k, 3), np.float64)
        for index, values in enumerate(coefficients):
            row, col = divmod(index, len(self.xs))
            for k, (kx, ky) in enumerate(self.terms):
                for j, (jx, jy) in enumerate(self.terms):
                    rhs[k] += mx[col][jx, kx] * my[row][jy, ky] * values[j]
        trend = np.linalg.lstsq(gram, rhs, rcond=1e-12)[0]
        for index in range(self.n):
            row, col = divmod(index, len(self.xs))
            ax = self.config.tile / self.w
            ay = self.config.tile / self.h
            bx = (2 * self.xs[col] + self.config.tile) / self.w - 1
            by = (2 * self.ys[row] + self.config.tile) / self.h - 1
            transform = np.zeros((6, 6), np.float64)
            transform[0, 0] = 1
            transform[1, :2] = [bx, ax]
            transform[2, [0, 2]] = [by, ay]
            transform[3, [0, 1, 3]] = [
                0.5 * ax * ax + 1.5 * bx * bx - 0.5,
                3 * ax * bx,
                ax * ax,
            ]
            transform[4, [0, 1, 2, 4]] = [bx * by, ax * by, bx * ay, ax * ay]
            transform[5, [0, 2, 5]] = [
                0.5 * ay * ay + 1.5 * by * by - 0.5,
                3 * ay * by,
                ay * ay,
            ]
            coefficients[index] -= transform[: self.k, : self.k].T @ trend
        return (coefficients, trend)

    def maximum_surface(self, coefficients):
        """检查每块RGB曲面在有效区域内的最大绝对值。

        同时检查角点、边界驻点和内部驻点，避免仅用少量采样点漏掉幅度峰值。
        """
        maximum = 0.0
        for index, values in enumerate(coefficients):
            row, col = divmod(index, len(self.xs))
            h = len(self.y_windows[row])
            w = len(self.x_windows[col])
            lo_x, hi_x = (
                -1 + 1 / self.config.tile,
                2 * (w - 0.5) / self.config.tile - 1,
            )
            lo_y, hi_y = (
                -1 + 1 / self.config.tile,
                2 * (h - 0.5) / self.config.tile - 1,
            )
            for channel in range(3):
                c = np.zeros(6)
                c[: self.k] = values[:, channel]
                points = [(u, v) for u in (lo_x, hi_x) for v in (lo_y, hi_y)]
                for u in (lo_x, hi_x):
                    if abs(c[5]) > 1e-15:
                        v = -(c[2] + c[4] * u) / (3 * c[5])
                        if lo_y <= v <= hi_y:
                            points.append((u, v))
                for v in (lo_y, hi_y):
                    if abs(c[3]) > 1e-15:
                        u = -(c[1] + c[4] * v) / (3 * c[3])
                        if lo_x <= u <= hi_x:
                            points.append((u, v))
                matrix = np.array([[3 * c[3], c[4]], [c[4], 3 * c[5]]])
                if abs(np.linalg.det(matrix)) > 1e-15:
                    u, v = np.linalg.solve(matrix, -c[1:3])
                    if lo_x <= u <= hi_x and lo_y <= v <= hi_y:
                        points.append((u, v))
                for u, v in points:
                    value = (
                        c[0]
                        + c[1] * u
                        + c[2] * v
                        + c[3] * (1.5 * u * u - 0.5)
                        + c[4] * u * v
                        + c[5] * (1.5 * v * v - 0.5)
                    )
                    maximum = max(maximum, abs(value))
        return maximum

    def solve(self, regularization=None, iterations=3):
        """联合求解所有块的RGB曲面，并评估拟合前后的重叠误差。

        正则项抑制过大的校正，后续迭代降低异常重叠点的权重。
        去趋势后整体缩放曲面，使实际连续曲面的幅度不超过max_shift。
        返回系数和诊断信息；是否采用由保护规则决定，不直接按PSNR选分支。
        """
        assert self.index == self.n
        if regularization is None:
            regularization = np.array([0.01, 0.05, 0.05, 0.2, 0.1, 0.2])[: self.k]
        regularization = np.broadcast_to(
            np.asarray(regularization, dtype=np.float64), (self.k,)
        )
        coefficients = np.zeros((self.n, self.k, 3), np.float64)
        for iteration in range(iterations):
            matrix = np.diag(np.tile(regularization, self.n))
            rhs = np.zeros((self.n * self.k, 3), np.float64)
            for i, j, a, b, d, weight in self.edges:
                residual = d + a @ coefficients[i] - b @ coefficients[j]
                robust = (
                    np.minimum(
                        1.0,
                        2
                        / 255
                        / np.maximum(np.sqrt(np.square(residual).mean(axis=1)), 1e-12),
                    )
                    if iteration
                    else 1.0
                )
                w = weight * robust
                ii = slice(i * self.k, (i + 1) * self.k)
                jj = slice(j * self.k, (j + 1) * self.k)
                aa = a.T @ (w[:, None] * a)
                bb = b.T @ (w[:, None] * b)
                ab = a.T @ (w[:, None] * b)
                matrix[ii, ii] += aa
                matrix[jj, jj] += bb
                matrix[ii, jj] -= ab
                matrix[jj, ii] -= ab.T
                rhs[ii] -= a.T @ (w[:, None] * d)
                rhs[jj] += b.T @ (w[:, None] * d)
            coefficients = np.linalg.solve(matrix, rhs).reshape(self.n, self.k, 3)
        # 先去掉整图趋势，再限制连续曲面的最大幅度，防止整体调色。
        coefficients, trend = self.detrend(coefficients)
        maximum = self.maximum_surface(coefficients)
        scale = min(1.0, self.max_shift / max(maximum, 1e-12))
        coefficients *= scale
        before = []
        after = []
        for i, j, a, b, d, weight in self.edges:
            before.append(float(np.average(np.abs(d).mean(axis=1), weights=weight)))
            after.append(
                float(
                    np.average(
                        np.abs(d + a @ coefficients[i] - b @ coefficients[j]).mean(
                            axis=1
                        ),
                        weights=weight,
                    )
                )
            )
        # 求解后不再需要历史重叠条带，及时释放。
        self.previous = []
        self.left = None
        report = dict(
            kind="joint_overlap_polynomial_surfaces",
            degree=self.degree,
            terms=self.k,
            tiles=self.n,
            edges=len(self.edges),
            pooling=self.r,
            regularization=regularization.tolist(),
            robust_iterations=iterations,
            max_allowed_correction=self.max_shift,
            actual_surface_bound=maximum * scale,
            scale=scale,
            overlap_lowfreq_mae_before_levels=float(np.mean(before) * 255)
            if before
            else 0.0,
            overlap_lowfreq_mae_after_levels=float(np.mean(after) * 255)
            if after
            else 0.0,
            removed_global_polynomial=trend.tolist(),
            global_color_and_polynomial_moments="Preserved before final clipping, using exact native-pixel moments",
            extra_network_calls=0,
            scope="Predicted overlap alignment; not a GT metric or proof of moire removal",
        )
        if self.native_diagnostics:
            before_sum = 0.0
            after_sum = 0.0
            pixels = 0

            def values(index, y, x, h, w):
                """在原生重叠像素上展开校正曲面，仅供显式开启的诊断模式使用。"""
                row, col = divmod(index, len(self.xs))
                fx = axis_basis(
                    2 * (x + np.arange(w) + 0.5 - self.xs[col]) / self.config.tile - 1
                )
                fy = axis_basis(
                    2 * (y + np.arange(h) + 0.5 - self.ys[row]) / self.config.tile - 1
                )
                out = np.zeros((h, w, 3), np.float64)
                for k, (ix, iy) in enumerate(self.terms):
                    out += (fy[iy, :, None] * fx[ix, None, :])[
                        ..., None
                    ] * coefficients[index, k]
                return out

            for i, j, (y, x, h, w), difference in self.native_edges:
                corrected = difference + values(i, y, x, h, w) - values(j, y, x, h, w)
                before_sum += np.abs(difference).sum(dtype=np.float64)
                after_sum += np.abs(corrected).sum(dtype=np.float64)
                pixels += difference.size
            report["native_overlap_mae_before_levels"] = (
                before_sum / max(1, pixels) * 255
            )
            report["native_overlap_mae_after_levels"] = after_sum / max(1, pixels) * 255
        self.native_edges = []
        self.edges = []
        return (coefficients.astype(np.float32), report)

    def apply(self, image, coefficients):
        """把已通过保护规则的曲面融合后，逐条带加到普通重建图上。

        仅对有效输出区域校正，结果原地写回；uint8结果仍按0～1裁剪后四舍五入。
        """
        c = self.config
        strip = min(c.tile, self.h)
        accumulator = np.zeros((strip, self.w, 3), np.float32)
        for row, (y, yw) in enumerate(zip(self.ys, self.y_windows)):
            fy = axis_basis(2 * (np.arange(len(yw)) + 0.5) / c.tile - 1).astype(
                np.float32
            )
            for col, (x, xw) in enumerate(zip(self.xs, self.x_windows)):
                fx = axis_basis(2 * (np.arange(len(xw)) + 0.5) / c.tile - 1).astype(
                    np.float32
                )
                value = np.zeros((len(yw), len(xw), 3), np.float32)
                for k, (ix, iy) in enumerate(self.terms):
                    value += (fy[iy, :, None] * fx[ix, None, :])[
                        ..., None
                    ] * coefficients[row * len(self.xs) + col, k]
                value *= (yw[:, None] * xw[None, :])[..., None]
                accumulator[: len(yw), x : x + len(xw)] += value
            next_y = self.ys[row + 1] if row + 1 < len(self.ys) else self.h
            emit = next_y - y
            for begin in range(0, emit, 32):
                end = min(emit, begin + 32)
                view = image[y + begin : y + end]
                field = accumulator[begin:end]
                if image.dtype == np.uint8:
                    view[:] = np.rint(
                        np.clip(view.astype(np.float32) / 255 + field, 0, 1) * 255
                    ).astype(np.uint8)
                else:
                    view += field
                    np.clip(view, 0, 1, out=view)
            if row + 1 < len(self.ys):
                keep = strip - emit
                if keep:
                    accumulator[:keep] = accumulator[emit:]
                accumulator[keep:] = 0
        return image



def apply_field(image, delta):
    """把低分辨率RGB校正场双线性放大，逐32行加回原图。

    坐标采用半像素对齐；只分配一个小条带的全分辨率校正，控制峰值内存。
    """
    h, w = image.shape[:2]
    dh, dw = delta.shape[:2]
    xs = np.clip((np.arange(w) + 0.5) * dw / w - 0.5, 0, dw - 1)
    x0 = xs.astype(int)
    x1 = np.minimum(x0 + 1, dw - 1)
    ax = (xs - x0).astype(np.float32)
    horizontal = (
        delta[:, x0] * (1 - ax)[None, :, None] + delta[:, x1] * ax[None, :, None]
    )
    for start in range(0, h, 32):
        end = min(h, start + 32)
        ys = np.clip((np.arange(start, end) + 0.5) * dh / h - 0.5, 0, dh - 1)
        y0 = ys.astype(int)
        y1 = np.minimum(y0 + 1, dh - 1)
        ay = (ys - y0).astype(np.float32)
        field = (
            horizontal[y0] * (1 - ay)[:, None, None]
            + horizontal[y1] * ay[:, None, None]
        )
        if image.dtype == np.uint8:
            image[start:end] = np.rint(
                np.clip(image[start:end].astype(np.float32) / 255 + field, 0, 1) * 255
            ).astype(np.uint8)
        else:
            image[start:end] += field
            np.clip(image[start:end], 0, 1, out=image[start:end])
    return image



class GradientFusion:
    """收集预测块内部的梯度，在粗网格上重建一致的颜色走势。

    这里平均的是块内梯度，避免将融合窗变化引入的亮度台阶也当成图像梯度。
    仅把平滑差值加回重建图，不再调用网络。
    """

    def __init__(self, shape, config, reduction=8):
        """建立八倍降采样的横、纵梯度累加器，以及普通融合结果的粗尺度参考。"""
        self.height, self.width = shape
        self.config = config
        self.reduction = reduction
        if any((n % reduction for n in (config.tile, config.halo, config.overlap))):
            raise ValueError("梯度粗网格必须与保留区域对齐")
        self.ys = tile_positions(self.height, config.tile, config.step)
        self.xs = tile_positions(self.width, config.tile, config.step)
        self.index = 0
        h = (self.height + reduction - 1) // reduction
        w = (self.width + reduction - 1) // reduction
        self.gx = np.zeros((h, w - 1, 3), np.float32)
        self.gy = np.zeros((h - 1, w, 3), np.float32)
        self.wx = np.zeros((h, w - 1), np.float32)
        self.wy = np.zeros((h - 1, w), np.float32)
        self.anchor = np.empty((h, w, 3), np.float32)
        self.anchor_rows = 0
        self.vector = cosine_vector(config.tile, config.overlap)

    def pool(self, image):
        """按8×8区域求均值，图像末端不足整格时使用边缘值补齐。"""
        r = self.reduction
        h, w = image.shape[:2]
        padded = (
            np.pad(image, ((0, -h % r), (0, -w % r), (0, 0)), mode="edge")
            if h % r or w % r
            else image
        )
        return padded.reshape((h + r - 1) // r, r, (w + r - 1) // r, r, 3).mean(
            axis=(1, 3), dtype=np.float32
        )

    def observe(self, prediction):
        """从当前块有效中心提取粗尺度梯度，按余弦窗权重累加到整图粗网格。"""
        c, r = (self.config, self.reduction)
        row, col = divmod(self.index, len(self.xs))
        y, x = (self.ys[row], self.xs[col])
        center = prediction[:, c.halo : c.halo + c.tile, c.halo : c.halo + c.tile]
        block = self.pool(
            center[
                :, : min(c.tile, self.height - y), : min(c.tile, self.width - x)
            ].transpose(1, 2, 0)
        )

        def axis(start, length):
            """将当前轴的有效融合窗补齐后分组平均，得到粗网格权重。"""
            v = _axis_window(self.vector, start, length, c.overlap)
            if len(v) < c.tile:
                v = np.pad(v, (0, c.tile - len(v)), mode="edge")
            return v.reshape(c.tile // r, r).mean(axis=1)

        window = axis(y, self.height)[:, None] * axis(x, self.width)[None, :]
        yy, xx = (y // r, x // r)
        h, w = (
            min(len(block), self.wx.shape[0] - yy),
            min(block.shape[1], self.wy.shape[1] - xx),
        )
        block, window = (block[:h, :w], window[:h, :w])
        if w > 1:
            weight = (window[:, :-1] + window[:, 1:]) * 0.5
            self.gx[yy : yy + h, xx : xx + w - 1] += (
                block[:, 1:] - block[:, :-1]
            ) * weight[..., None]
            self.wx[yy : yy + h, xx : xx + w - 1] += weight
        if h > 1:
            weight = (window[:-1] + window[1:]) * 0.5
            self.gy[yy : yy + h - 1, xx : xx + w] += (block[1:] - block[:-1]) * weight[
                ..., None
            ]
            self.wy[yy : yy + h - 1, xx : xx + w] += weight
        self.index += 1

    def observe_strip(self, y, strip):
        """在普通融合条带量化前保存粗网格参考，作为后续亮度锚点。"""
        pooled = self.pool(strip)
        start = y // self.reduction
        self.anchor[start : start + len(pooled)] = pooled
        self.anchor_rows += len(pooled)

    def apply(self, reconstruction, screening=0.0):
        """用离散余弦变换求解粗网格上的Neumann泊松方程。

        直流项沿用普通融合图的均值，固定整体亮度基准；再将解与参考的差值放大加回。
        这一分支用于曲面对齐不足时的回退，并不意味着图内屏纹已被消除。
        """
        from scipy.fft import dctn, idctn

        if self.index != len(self.ys) * len(self.xs):
            raise RuntimeError("梯度收集尚未覆盖全部分块")
        if np.any(self.wx <= 0) or np.any(self.wy <= 0):
            raise RuntimeError("粗网格的部分梯度边未被覆盖")
        self.gx /= self.wx[..., None]
        self.gy /= self.wy[..., None]
        h, w = (self.wx.shape[0], self.wy.shape[1])
        if self.anchor_rows == h:
            anchor = self.anchor
        else:
            anchor = self.pool(reconstruction)
            if reconstruction.dtype == np.uint8:
                anchor *= 1 / 255
        rhs = np.zeros((h, w, 3), np.float32)
        rhs[:, :-1] -= self.gx
        rhs[:, 1:] += self.gx
        rhs[:-1] -= self.gy
        rhs[1:] += self.gy
        eigenvalues = (
            4 * np.sin(np.pi * np.arange(h) / (2 * h))[:, None] ** 2
            + 4 * np.sin(np.pi * np.arange(w) / (2 * w))[None, :] ** 2
        ).astype(np.float32)
        denominator = eigenvalues + screening
        # 零频率的拉普拉斯特征值为零，单独处理直流，避免除零。
        denominator[0, 0] = 1
        result = np.empty_like(anchor)
        for channel in range(3):
            transformed = dctn(
                rhs[..., channel] + screening * anchor[..., channel], norm="ortho"
            )
            transformed /= denominator
            # 保留普通重建的通道均值，不引入来自原图或GT的颜色匹配。
            transformed[0, 0] = np.mean(
                anchor[..., channel], dtype=np.float64
            ) * np.sqrt(h * w)
            result[..., channel] = idctn(transformed, norm="ortho")
        delta = result - anchor

        def mismatch(image):
            """统计粗网格图像梯度与目标预测梯度的均方差，供内部诊断。"""
            return float(
                (
                    np.square(np.diff(image, axis=1) - self.gx).sum(dtype=np.float64)
                    + np.square(np.diff(image, axis=0) - self.gy).sum(dtype=np.float64)
                )
                / max(1, self.gx.size + self.gy.size)
            )

        report = dict(
            kind="coarse_gradient_domain_assembly",
            reduction=self.reduction,
            coarse_shape=[h, w],
            screening=screening,
            extra_network_calls=0,
            correction_abs_mean=float(np.abs(delta).mean()),
            correction_abs_max=float(np.abs(delta).max()),
            gradient_mismatch_before=mismatch(anchor),
            gradient_mismatch_after_coarse_solve=mismatch(result),
            constant_mode="Ordinary reconstruction RGB mean; no input/GT color matching",
            scope="Smooth assembly correction; original tile predictions remain unchanged",
        )
        apply_field(reconstruction, delta)
        return (reconstruction, report)



def choose_assembly(surface_report):
    """仅在曲面拟合至少消除一半重叠误差时采用它，否则回退梯度域组装。

    判断依据只有预测重叠差，不读取GT、图片名称或外部画质指标。
    """
    before = surface_report["overlap_lowfreq_mae_before_levels"]
    after = surface_report["overlap_lowfreq_mae_after_levels"]
    accepted = before <= 1e-09 or after <= 0.5 * before
    return dict(
        selected="overlap_surface" if accepted else "gradient",
        accepted=accepted,
        maximum_residual_ratio=0.5,
        observed_residual_ratio=after / before if before > 1e-09 else 0.0,
        decision_inputs="Predicted overlap errors only; no GT, quality metric or image identity",
        reason="Fit removes at least half of the overlap mismatch"
        if accepted
        else "Bounded fit leaves more than half of the overlap mismatch; retain gradient assembly",
        extra_network_calls=0,
    )



def restore_consistent(
    model, source, *, progress=None, baseline_sink=None, assembly="guarded"
):
    """执行完整流程：512分块预测→余弦融合→保护式接缝组装。

    两种后处理共享同一批网络预测，不增加网络调用。source使用RGB HWC。
    返回同尺寸uint8结果及内存中的诊断信息；命令行只保存结果PNG。
    """
    if assembly not in ("guarded", "cosine"):
        raise ValueError(assembly)
    config = TileConfig(384, 64, 64, 32)
    surface = (
        OverlapSurface(source.shape[:2], config, degree=2)
        if assembly == "guarded"
        else None
    )
    gradient = (
        GradientFusion(source.shape[:2], config) if assembly == "guarded" else None
    )
    started = time.perf_counter()
    model.prepare_image(source)
    previewed = time.perf_counter()

    def predict(patch):
        """预测一块后立即交给两个组装器收集重叠差和梯度，不保留整块历史。"""
        output = model.infer_chw(patch)
        if gradient is not None:
            gradient.observe(output)
        if surface is not None:
            surface.observe(output)
        return output

    result = infer_tiled(
        source,
        predict,
        config,
        output_dtype=np.uint8,
        blend_window="cosine",
        on_emit=gradient.observe_strip if gradient is not None else None,
        progress=progress,
    )
    inferred = time.perf_counter()
    baseline_hash = hashlib.sha256(result).hexdigest()
    control_started = time.perf_counter()
    if baseline_sink is not None:
        baseline_sink(result)
    control_finished = time.perf_counter()
    gradient_report = None
    surface_report = None
    guard = None
    if surface is not None:
        coefficients, surface_report = surface.solve()
        guard = choose_assembly(surface_report)
        surface_report["applied"] = guard["accepted"]
        if guard["accepted"]:
            gradient = None
            result = surface.apply(result, coefficients)
        else:
            result, gradient_report = gradient.apply(result)
    finished = time.perf_counter()
    return (
        result,
        dict(
            context=model.context,
            postprocess="overlap_guarded" if assembly == "guarded" else "cosine",
            preview_seconds=previewed - started,
            tiled_inference_and_blend_seconds=inferred - previewed,
            control_png_seconds=control_finished - control_started,
            postprocess_seconds=finished - control_finished,
            inference_and_postprocess_seconds=finished
            - started
            - (control_finished - control_started),
            baseline_RGB_sha256=baseline_hash,
            surface_correction=surface_report,
            gradient_correction=gradient_report,
            guard_correction=guard,
            native_calls=model.native_calls,
            preview_calls=model.preview_calls,
            native_input_shape=[1, 3, 512, 512],
            retained_tile=384,
            halo=64,
            overlap=64,
            stride=320,
            extra_network_calls_for_assembly=0,
            extra_learned_parameters=0,
        ),
    )

