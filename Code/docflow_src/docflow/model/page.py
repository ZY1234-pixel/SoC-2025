"""页面与文档 —— 版面模型的顶层容器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from docflow.model.base import BBox, Block
from docflow.model.zone import Zone
from docflow.utils.constants import (
    A4_HEIGHT_PT,
    A4_WIDTH_PT,
    DEFAULT_MARGIN_PT,
    MM_PER_INCH,
    PAGE_SIZES,
    PT_PER_INCH,
)


# ---------------------------------------------------------------------------
# CoordMapper —— 像素 <-> 磅值转换
# ---------------------------------------------------------------------------

class CoordMapper:
    """像素坐标与磅值坐标之间的转换器。

    基于目标页面的可用区域（总尺寸减去页边距）计算缩放因子，
    将源图像像素映射到 Word 磅值。

    当页边距不可用（全为零）时使用完整页面尺寸进行简单线性缩放。
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        page_width_pt: float,
        page_height_pt: float,
        margin_left_pt: float = 0.0,
        margin_right_pt: float = 0.0,
        margin_top_pt: float = 0.0,
        margin_bottom_pt: float = 0.0,
    ) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.page_width_pt = page_width_pt
        self.page_height_pt = page_height_pt
        usable_w = page_width_pt - margin_left_pt - margin_right_pt
        usable_h = page_height_pt - margin_top_pt - margin_bottom_pt
        self._sx = usable_w / max(image_width, 1)
        self._sy = usable_h / max(image_height, 1)

    # 标量转换
    def w(self, px: float) -> float:
        """将水平像素距离转换为磅值。"""
        return px * self._sx

    def h(self, px: float) -> float:
        """将垂直像素距离转换为磅值。"""
        return px * self._sy

    def x(self, px: float) -> float:
        """将 x 坐标从像素转换为磅值。"""
        return px * self._sx

    def y(self, px: float) -> float:
        """将 y 坐标从像素转换为磅值。"""
        return px * self._sy


def _pct(values: List[float], p: float) -> float:
    """计算百分位数（线性插值）。"""
    if not values:
        return 54.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    d = k - f
    return s[f] + d * (s[c] - s[f])


# ---------------------------------------------------------------------------
# Page —— 单页
# ---------------------------------------------------------------------------

@dataclass
class Page:
    """文档的一页，持有尺寸、页边距、区域以及可选的栅格图像（用于图形提取）。"""

    index: int
    image_width: int
    image_height: int

    page_width_pt: float = A4_WIDTH_PT
    page_height_pt: float = A4_HEIGHT_PT

    margin_left_pt: float = DEFAULT_MARGIN_PT
    margin_right_pt: float = DEFAULT_MARGIN_PT
    margin_top_pt: float = DEFAULT_MARGIN_PT
    margin_bottom_pt: float = DEFAULT_MARGIN_PT

    orientation: str = "portrait"

    zones: List[Zone] = field(default_factory=list)

    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    style_defaults: Optional[dict] = None
    attributes: Optional[dict] = None
    relations: List[dict] = field(default_factory=list)

    # -- 延迟初始化的坐标映射器 ---------------------------------------------------

    _coord_mapper: Optional[CoordMapper] = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def usable_width_pt(self) -> float:
        return self.page_width_pt - self.margin_left_pt - self.margin_right_pt

    @property
    def usable_height_pt(self) -> float:
        return self.page_height_pt - self.margin_top_pt - self.margin_bottom_pt

    @property
    def coord_mapper(self) -> CoordMapper:
        """返回（并延迟创建）:class:`CoordMapper` 实例。"""
        if self._coord_mapper is None:
            self._coord_mapper = CoordMapper(
                image_width=self.image_width,
                image_height=self.image_height,
                page_width_pt=self.page_width_pt,
                page_height_pt=self.page_height_pt,
                margin_left_pt=self.margin_left_pt,
                margin_right_pt=self.margin_right_pt,
                margin_top_pt=self.margin_top_pt,
                margin_bottom_pt=self.margin_bottom_pt,
            )
        return self._coord_mapper

    # -- 页面尺寸检测 -------------------------------------------------

    def detect_page_size(self) -> None:
        """将图片宽高比匹配到最接近的标准纸张尺寸。

        同时检查纵向和横向方向，设置 :attr:`page_width_pt`、
        :attr:`page_height_pt` 和 :attr:`orientation`。

        当多个尺寸的宽高比几乎相同时（如 A3 与 A4），
        优先选择较小的尺寸以避免过度缩放。
        """
        if self.image_width <= 0 or self.image_height <= 0:
            return

        img_ratio = self.image_width / self.image_height

        # 收集所有候选项：(ratio_diff, area_mm2, name, is_portrait)
        candidates: list[tuple[float, float, str, bool]] = []
        for name, w_mm, h_mm in PAGE_SIZES:
            area = w_mm * h_mm
            diff_p = abs(img_ratio - w_mm / h_mm)
            candidates.append((diff_p, area, name, True))
            diff_l = abs(img_ratio - h_mm / w_mm)
            candidates.append((diff_l, area, name, False))

        # 先按宽高比差值排序，再以较小面积作为平局决胜。
        # 将差值量化到 0.1% 容差，使几乎相同的宽高比
        #（如 A3 与 A4，均为 √2:1）视为相等，较小的（更常见的）尺寸胜出。
        _RATIO_TOL = 0.001
        candidates.sort(key=lambda c: (round(c[0] / _RATIO_TOL), c[1]))
        best_diff, best_area, best_name, best_portrait = candidates[0]

        if best_diff > 0.06:
            ratio = max(img_ratio, 1e-6)
            if ratio >= 1.0:
                width_mm = (best_area * ratio) ** 0.5
                height_mm = best_area / max(width_mm, 1e-6)
                self.orientation = "landscape"
            else:
                height_mm = (best_area / ratio) ** 0.5
                width_mm = best_area / max(height_mm, 1e-6)
                self.orientation = "portrait"
            self.page_width_pt = width_mm / MM_PER_INCH * PT_PER_INCH
            self.page_height_pt = height_mm / MM_PER_INCH * PT_PER_INCH
            self._coord_mapper = None
            return

        # 将胜出的尺寸从毫米转换为磅值
        for name, w_mm, h_mm in PAGE_SIZES:
            if name == best_name:
                w_pt = w_mm / MM_PER_INCH * PT_PER_INCH
                h_pt = h_mm / MM_PER_INCH * PT_PER_INCH
                if best_portrait:
                    self.page_width_pt = w_pt
                    self.page_height_pt = h_pt
                    self.orientation = "portrait"
                else:
                    self.page_width_pt = h_pt
                    self.page_height_pt = w_pt
                    self.orientation = "landscape"
                break

        # 使缓存的坐标映射器失效以使用新的尺寸
        self._coord_mapper = None

    # -- 页边距估算 ---------------------------------------------------

    def estimate_margins(self, blocks: List[Block]) -> None:
        """从区块边界框估算页边距。

        使用全页缩放（无页边距）将像素距离转换为磅值，
        然后设置页边距并使缓存的 coord_mapper 失效，
        使后续调用使用含页边距的缩放。

        策略：
          - 先分别计算左/右边界，再对称化
          - 左边距：较宽区块（≥ 25% 页宽）的 5th 百分位数
          - 右边距：所有区块右边缘的 15th 百分位数
          - 对称化：以较大边距为基准，确保左右对称
            （大多数文档使用对称页边距，不对称会导致内容偏移）
          - 上边距：顶部 15% 区域内块的最小 y1；若无则用全页最小 y1
          - 下边距：底部 15% 区域内块的最小 (page_h - y2)；若无则用全页最大 y2
          - 钳位范围：左右边距 [36, 120] pt，上下边距 [24, 120] pt
            （左右边距下限 36pt ≈ 12.7mm，避免内容从侧边溢出）
        """
        if not blocks:
            return

        sx = self.page_width_pt / max(self.image_width, 1)
        sy = self.page_height_pt / max(self.image_height, 1)

        page_w = max(self.image_width, 1)
        page_h = max(self.image_height, 1)

        # ── 左边距：较宽区块的 5th 百分位数 ──
        min_w = page_w * 0.25
        wide = [b for b in blocks if b.bbox.width >= min_w]
        if not wide:
            wide = blocks
        ml_raw = _pct([b.bbox.x1 * sx for b in wide], 0.05)

        # ── 右边距：所有区块的 15th 百分位数 ──
        mr_raw = _pct([self.page_width_pt - b.bbox.x2 * sx for b in blocks], 0.15)

        # ── 对称化：取较大边距为基准，左右统一 ──
        # 大多数文档（学术/教材/书籍）使用对称页边距。
        # 若分别计算，内容会偏向边距小的一侧，视觉上"挤压"另一边。
        # 以较大边距为准，确保内容居中。
        margin_lr = max(ml_raw, mr_raw)

        # ── 上边距：顶部 15% 区域内块的最小 y1 ──
        top_y = page_h * 0.15
        top_blocks = [b for b in blocks if b.bbox.y1 < top_y]
        if top_blocks:
            mt = min(b.bbox.y1 for b in top_blocks) * sy
        else:
            mt = min(b.bbox.y1 for b in blocks) * sy

        # ── 下边距：底部 15% 区域内块的最小 (page_h - y2) ──
        bottom_y = page_h * 0.85
        bottom_blocks = [b for b in blocks if b.bbox.y2 > bottom_y]
        if bottom_blocks:
            mb = min(self.page_height_pt - b.bbox.y2 * sy for b in bottom_blocks)
        else:
            max_y2 = max(b.bbox.y2 for b in blocks)
            mb = (self.page_height_pt - max_y2 * sy)

        # 限制在合理范围：
        # 左右边距至少 36pt（~12.7mm），避免内容从侧边溢出
        # 上下边距至少 24pt（~8.5mm）
        # 所有边距至多 120pt（~42mm）
        def _clamp_lr(val: float) -> float:
            return max(36.0, min(120.0, val))

        def _clamp_tb(val: float) -> float:
            return max(24.0, min(120.0, val))

        self.margin_left_pt = _clamp_lr(margin_lr)
        self.margin_right_pt = _clamp_lr(margin_lr)
        self.margin_top_pt = _clamp_tb(mt)
        self.margin_bottom_pt = _clamp_tb(mb)

        # 使缓存失效，使下次 coord_mapper 调用使用新的页边距
        self._coord_mapper = None


# ---------------------------------------------------------------------------
# Document —— 完整文档
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """由一个或多个 :class:`Page` 对象组成的完整文档。"""

    pages: List[Page] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
