"""图像补边、规则分块与条带式余弦融合。

网络输入512像素，保留中心384像素，相邻保留区域重叠64像素。
不为整张4K图建立浮点累加器，降低大图处理的内存开销。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable
import numpy as np

@dataclass(frozen=True)
class TileConfig:
    """分块配置。tile 表示保留的输出边长，不是网络输入边长。

    默认保留384像素，两侧各带64像素上下文，得到512像素网络输入；
    相邻输出重叠64像素，移动步长为320。整幅图按同一网格处理。
    """

    tile: int = 384
    overlap: int = 64
    halo: int = 64
    alignment: int = 8

    def __post_init__(self):
        """检查尺寸与对齐约束，尽早拦截会导致补边或拼接错误的配置。"""
        if self.alignment <= 0 or self.tile <= 0 or self.halo < 0:
            raise ValueError("tile和alignment必须为正数，halo不能为负数")
        if not 0 <= self.overlap < self.tile:
            raise ValueError("重叠宽度必须满足0 <= overlap < tile")
        if any(
            (value % self.alignment for value in (self.tile, self.overlap, self.halo))
        ):
            raise ValueError(
                f"保留区域、重叠与halo尺寸必须是{self.alignment}的整数倍"
            )

    @property
    def step(self) -> int:
        """返回相邻两个保留区域的起点间隔：384减64，得到320。"""
        return self.tile - self.overlap

    @property
    def model_tile(self) -> int:
        """返回送入网络的边长：384加两侧64，得到512。"""
        return self.tile + 2 * self.halo



def tile_positions(length: int, tile: int, step: int) -> list[int]:
    """生成固定网格上的分块起点。

    最后一块也保持相同步长；超出图像的部分由补边解决，不向回挪动。
    这样可以避免最后一块与其他块落在不同的特征采样相位上。
    """
    if length <= 0 or tile <= 0 or (not 0 < step <= tile):
        raise ValueError("长度和块尺寸必须为正数，且0 < step <= tile")
    count = max(1, (max(0, length - tile) + step - 1) // step + 1)
    return [index * step for index in range(count)]



@lru_cache(maxsize=32)
def cosine_vector(tile: int, overlap: int) -> np.ndarray:
    """生成一维余弦融合窗，中心权重为1，重叠端逐渐衰减。

    缓存只读窗函数，避免每一块重复计算；二维窗由横、纵窗相乘得到。
    """
    if tile <= 0 or not 0 <= overlap < tile:
        raise ValueError("无效的保留区域或重叠宽度")
    vector = np.ones(tile, dtype=np.float32)
    ramp_size = min(overlap, tile // 2)
    if ramp_size:
        coord = np.arange(1, ramp_size + 1, dtype=np.float64)
        ramp = (0.5 - 0.5 * np.cos(np.pi * coord / (ramp_size + 1))).astype(np.float32)
        vector[:ramp_size] = ramp
        vector[-ramp_size:] = ramp[::-1]
    vector.setflags(write=False)
    return vector



def _axis_window(
    vector: np.ndarray, start: int, length: int, overlap: int
) -> np.ndarray:
    """裁出当前块真正覆盖的窗；图像最外缘没有相邻块，因此不做端部衰减。"""
    window = vector.copy()
    if overlap:
        ramp_size = min(overlap, len(vector) // 2)
        if start == 0:
            window[:ramp_size] = 1
        if start + len(vector) >= length:
            window[-ramp_size:] = 1
    return window[: min(len(vector), length - start)]



def _extract_patch(image: np.ndarray, y: int, x: int, config: TileConfig) -> np.ndarray:
    """取出带上下文的512像素输入块，并转换成网络需要的格式。

    输入图像为RGB、HWC排列；图像外侧优先镜像补边，退化小尺寸用边缘复制。
    输出为连续内存的float32 CHW，uint8输入除以255，不做均值方差标准化。
    """
    height, width = image.shape[:2]
    top, left = (y - config.halo, x - config.halo)
    bottom, right = (top + config.model_tile, left + config.model_tile)
    patch = image[max(0, top) : min(height, bottom), max(0, left) : min(width, right)]
    pads = (
        (max(0, -top), max(0, bottom - height)),
        (max(0, -left), max(0, right - width)),
        (0, 0),
    )
    if any((before or after for before, after in pads)):
        patch = np.pad(
            patch, pads, mode="reflect" if min(patch.shape[:2]) > 1 else "edge"
        )
    patch = np.ascontiguousarray(patch.transpose(2, 0, 1), dtype=np.float32)
    if image.dtype == np.uint8:
        patch *= 1.0 / 255.0
    return patch



def infer_tiled(
    image: np.ndarray,
    predict: Callable[[np.ndarray], np.ndarray],
    config: TileConfig = TileConfig(),
    *,
    output_dtype=np.uint8,
    progress: Callable[[int, int], None] | None = None,
    overlap_color_correction: bool = False,
    blend_window: str = "cosine",
    before_predict: Callable[[int, int], None] | None = None,
    on_emit: Callable[[int, np.ndarray], None] | None = None,
) -> np.ndarray:
    """逐块预测，并用条带缓冲做重叠加权融合。

    image：RGB图像，uint8为0～255，浮点图像应在0～1。
    predict：接收CHW浮点输入，返回同尺寸恢复图；已包含网络RGB残差。
    on_emit：在条带量化前接收融合结果，供梯度域组装保存粗尺度参考。

    缓冲区只有一个保留块高度，不为整幅4K图分配浮点累加器。
    每一行收到所有贡献后才裁剪、量化，避免逐块量化放大拼接差异。
    """
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError(f"需要非空的HWC RGB图像，实际为{image.shape}")
    if image.dtype != np.uint8 and (not np.issubdtype(image.dtype, np.floating)):
        raise TypeError("输入必须是uint8图像或0～1范围的浮点RGB图像")
    if output_dtype not in (np.uint8, np.float32):
        raise TypeError("输出类型必须是np.uint8或np.float32")
    height, width = image.shape[:2]
    ys = tile_positions(height, config.tile, config.step)
    xs = tile_positions(width, config.tile, config.step)
    if blend_window == "cosine":
        vector = cosine_vector(config.tile, config.overlap)
    elif blend_window == "hann":
        vector = (
            np.sin(
                np.pi * (np.arange(config.tile, dtype=np.float64) + 0.5) / config.tile
            ).astype(np.float32)
            ** 2
        )
    else:
        raise ValueError(blend_window)
    x_windows = [_axis_window(vector, x, width, config.overlap) for x in xs]
    # 只缓存一条带的RGB和权重；已完成的行立即写入uint8结果。
    strip_height = min(config.tile, height)
    accumulator = np.zeros((strip_height, width, 3), dtype=np.float32)
    weights = np.zeros((strip_height, width), dtype=np.float32)
    result = np.empty((height, width, 3), dtype=output_dtype)
    total, done = (len(ys) * len(xs), 0)
    for row, y in enumerate(ys):
        y_window = _axis_window(vector, y, height, config.overlap)
        for x, x_window in zip(xs, x_windows):
            if before_predict is not None:
                before_predict(y, x)
            prediction = np.asarray(
                predict(_extract_patch(image, y, x, config)), dtype=np.float32
            )
            expected = (3, config.model_tile, config.model_tile)
            if prediction.shape != expected:
                raise ValueError(
                    f"网络返回{prediction.shape}，预期为{expected}"
                )
            if not np.isfinite(prediction).all():
                raise ValueError("网络输出包含NaN或无穷值")
            h, w = (len(y_window), len(x_window))
            center = prediction[
                :, config.halo : config.halo + h, config.halo : config.halo + w
            ].transpose(1, 2, 0)
            if overlap_color_correction:
                center = np.clip(center, 0.0, 1.0)
                previous = weights[:h, x : x + w]
                mask = previous > 0.05
                if int(mask.sum()) >= 64:
                    placed = accumulator[:h, x : x + w] / np.maximum(
                        previous[..., None], 1e-06
                    )
                    delta = (placed[mask] - center[mask]).mean(axis=0)
                    center = np.clip(center + delta, 0.0, 1.0)
            window = y_window[:, None] * x_window[None, :]
            accumulator[:h, x : x + w] += center * window[..., None]
            weights[:h, x : x + w] += window
            done += 1
            if progress:
                progress(done, total)
        # 下一行块不会再覆盖next_y之前的行，现在可以安全输出这些行。
        next_y = ys[row + 1] if row + 1 < len(ys) else height
        emit = next_y - y
        active = accumulator[:emit]
        if np.any(weights[:emit] <= 0):
            raise RuntimeError("融合网格中存在未被有效权重覆盖的像素")
        # 必须除以真实权重和；人为抬高分母会让图像边缘变暗。
        np.divide(active, weights[:emit, :, None], out=active)
        np.clip(active, 0.0, 1.0, out=active)
        if on_emit is not None:
            on_emit(y, active)
        if output_dtype == np.uint8:
            active *= 255.0
            np.rint(active, out=active)
        result[y:next_y] = active
        if row + 1 < len(ys):
            keep = strip_height - emit
            if keep:
                accumulator[:keep] = accumulator[emit:]
                weights[:keep] = weights[emit:]
            accumulator[keep:] = 0
            weights[keep:] = 0
    return result

