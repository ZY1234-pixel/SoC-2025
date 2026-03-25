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
        _, _, best_name, best_portrait = candidates[0]

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
        """
        if not blocks:
            return

        # 使用全页缩放（无页边距）进行初始估算
        sx = self.page_width_pt / max(self.image_width, 1)
        sy = self.page_height_pt / max(self.image_height, 1)

        min_left_px = min(b.bbox.x1 for b in blocks)
        max_right_px = max(b.bbox.x2 for b in blocks)
        min_top_px = min(b.bbox.y1 for b in blocks)
        max_bottom_px = max(b.bbox.y2 for b in blocks)

        ml = min_left_px * sx
        mr = self.page_width_pt - max_right_px * sx
        mt = min_top_px * sy
        mb = self.page_height_pt - max_bottom_px * sy

        # 限制在合理范围：至少 18 pt（~6 mm），至多 72 pt（~25 mm）
        def _clamp(val: float) -> float:
            return max(18.0, min(72.0, val))

        self.margin_left_pt = _clamp(ml)
        self.margin_right_pt = _clamp(mr)
        self.margin_top_pt = _clamp(mt)
        self.margin_bottom_pt = _clamp(mb)

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
