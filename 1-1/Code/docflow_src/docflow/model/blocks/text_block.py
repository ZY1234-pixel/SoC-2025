"""文本相关版面区块：TextLine、Paragraph、TextBlock。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from docflow.model.base import Alignment, BBox, Block, BlockType
from docflow.schema.models import BlockStyle, TextLineStyle


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
    )


def should_insert_space(prev_text: str, curr_text: str) -> bool:
    """判断两段文本之间是否应插入空格（主要用于拉丁词边界）。"""
    if not prev_text or not curr_text:
        return False
    prev = prev_text.rstrip()
    curr = curr_text.lstrip()
    if not prev or not curr:
        return False

    prev_last = prev[-1]
    curr_first = curr[0]
    if prev_last.isspace() or curr_first.isspace():
        return False
    if _is_cjk_char(prev_last) or _is_cjk_char(curr_first):
        return False

    if prev_last.isalnum() and curr_first.isalnum():
        return True
    if prev_last in ",.;:!?)%]”’" and curr_first.isalnum():
        return True
    return False


def join_text_segments(parts: List[str]) -> str:
    """语言感知文本拼接：CJK 直连，拉丁词边界自动补空格。"""
    out: List[str] = []
    prev = ""
    for raw in parts:
        text = str(raw or "")
        if not text:
            continue
        if out and should_insert_space(prev, text):
            out.append(" ")
        out.append(text)
        prev = text
    return "".join(out)


# ---------------------------------------------------------------------------
# TextLine —— 单条 OCR 文本行
# ---------------------------------------------------------------------------

@dataclass
class TextLine:
    """单条识别文本行（来自 OCR / 文本检测）。"""

    text: str
    confidence: float = 1.0
    text_region: Optional[List[List[float]]] = None
    style: Optional[TextLineStyle] = None   # JSON text_line.style

    # -- 几何辅助属性（*text_region* 缺失时返回 ``None``）------

    @property
    def x1(self) -> Optional[float]:
        if self.text_region is None:
            return None
        return min(pt[0] for pt in self.text_region)

    @property
    def y1(self) -> Optional[float]:
        if self.text_region is None:
            return None
        return min(pt[1] for pt in self.text_region)

    @property
    def x2(self) -> Optional[float]:
        if self.text_region is None:
            return None
        return max(pt[0] for pt in self.text_region)

    @property
    def y2(self) -> Optional[float]:
        if self.text_region is None:
            return None
        return max(pt[1] for pt in self.text_region)

    @property
    def height(self) -> Optional[float]:
        if self.text_region is None:
            return None
        ys = [pt[1] for pt in self.text_region]
        return max(ys) - min(ys)


# ---------------------------------------------------------------------------
# Paragraph —— 由多条 TextLine 组成的逻辑段落
# ---------------------------------------------------------------------------

@dataclass
class Paragraph:
    """由一条或多条 :class:`TextLine` 组装而成的逻辑段落。"""

    lines: List[TextLine] = field(default_factory=list)
    first_line_indent_px: float = 0.0

    @property
    def text(self) -> str:
        """按语言规则拼接段落文本。"""
        return join_text_segments([line.text for line in self.lines])


# ---------------------------------------------------------------------------
# TextBlock —— 包含文本的版面区块
# ---------------------------------------------------------------------------

@dataclass
class TextBlock(Block):
    """页面上的矩形文本区域。"""

    block_type: BlockType = BlockType.TEXT
    lines: List[TextLine] = field(default_factory=list)
    paragraphs: List[Paragraph] = field(default_factory=list)
    alignment: Alignment = Alignment.LEFT
    estimated_font_size_pt: float = 10.5
    style: Optional[BlockStyle] = None   # JSON block.style，优先于推断值

    # -- 分析辅助方法 ----------------------------------------------------

    def count_lines(self) -> int:
        """通过 y 区间合并来统计逻辑文本行数。

        共享相同垂直区间的并排 OCR 片段会被合并为一行，
        避免重复计数。
        """
        intervals: List[List[float]] = []
        for tl in self.lines:
            if tl.text_region is None:
                continue
            ys = [pt[1] for pt in tl.text_region]
            lo, hi = min(ys), max(ys)
            intervals.append([lo, hi])

        if not intervals:
            return max(1, len(self.lines))

        # 按下界排序，然后合并重叠区间
        intervals.sort(key=lambda iv: iv[0])
        merged: List[List[float]] = [intervals[0]]
        for lo, hi in intervals[1:]:
            if lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])

        return max(1, len(merged))

    def estimate_font_size(self, mapper: Any) -> None:
        """从文本行高度或区块 bbox 估算字体大小。

        当 text_region 可用时，使用中位行高来获取更准确的估算
        （避免 bbox 内边距膨胀影响）。否则回退为 bbox 高度 / 行数。

        行高 = font_size × line_spacing，其中 line_spacing 通常在 1.0–1.15 之间。
        因此除数用 1.15 而非 1.25，避免系统性地低估字号。

        结果被限制在 [6, 36] pt 范围内，并四舍五入到最近的 0.5 pt，
        最后吸附到常见字号网格（9/9.5/10/10.5/11/12/14/16/18/20/24/28/36）。
        """
        # 先尝试使用实际 text_region 高度
        line_heights: List[float] = []
        for tl in self.lines:
            h = tl.height
            if h is not None and h > 0:
                line_heights.append(h)

        if line_heights:
            # 使用中位行高以提高对异常值的鲁棒性
            line_heights.sort()
            median_h = line_heights[len(line_heights) // 2]
            # text_region 测量的是文本行的实际高度（含升部/降部），
            # 不是行间距。除数 1.05 仅补偿行内额外空间，不混入行间空白。
            fs = mapper.h(median_h) / 1.05
        else:
            n_lines = self.count_lines()
            if n_lines <= 0:
                return
            # bbox 高度包含行间空白，用更大的除数补偿
            fs = (mapper.h(self.bbox.height) / n_lines) / 1.20

        fs = max(6.0, min(36.0, fs))
        fs = round(fs * 2.0) / 2.0
        fs = self._snap_to_font_grid(fs)
        self.estimated_font_size_pt = fs

    @staticmethod
    def _snap_to_font_grid(raw_pt: float) -> float:
        """将字号吸附到常见字号值。"""
        grid = [9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0, 28.0, 36.0]
        if raw_pt < 9.0:
            return max(raw_pt, 6.0)
        best = min(grid, key=lambda v: abs(v - raw_pt))
        # 仅当差值 < 1.5 pt 时才吸附，避免过度修正
        return best if abs(best - raw_pt) < 1.5 else raw_pt

    def detect_alignment(
        self,
        col_left_px: float,
        col_right_px: float,
        threshold_ratio: float = 0.12,
    ) -> None:
        """从文本行位置检测段落对齐方式。

        将文本行的平均左/右缩进与列边界进行比较，
        并相应设置 :attr:`alignment`。
        """
        if not self.lines:
            return

        left_offsets: List[float] = []
        right_offsets: List[float] = []
        for tl in self.lines:
            if tl.x1 is None or tl.x2 is None:
                continue
            left_offsets.append(tl.x1 - col_left_px)
            right_offsets.append(col_right_px - tl.x2)

        if not left_offsets:
            return

        col_width = col_right_px - col_left_px
        if col_width <= 0:
            return

        avg_left = sum(left_offsets) / len(left_offsets)
        avg_right = sum(right_offsets) / len(right_offsets)

        left_ratio = avg_left / col_width
        right_ratio = avg_right / col_width

        both_small = left_ratio < threshold_ratio and right_ratio < threshold_ratio
        left_big = left_ratio >= threshold_ratio
        right_big = right_ratio >= threshold_ratio

        if both_small:
            self.alignment = Alignment.JUSTIFY
        elif left_big and right_big:
            self.alignment = Alignment.CENTER
        elif left_big:
            self.alignment = Alignment.RIGHT
        else:
            self.alignment = Alignment.LEFT

    def full_text(self) -> str:
        """返回所有行文本的拼接结果。"""
        return join_text_segments([line.text for line in self.lines])

