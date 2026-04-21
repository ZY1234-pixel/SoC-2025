"""检测 OCR 文本行序列中的段落边界。

原始代码源自 recovery_to_doc lines 445-543。

结合两种启发式来判断段落边界：

1. **首行缩进** -- 若某行左边缘相对于周围行向右偏移至少
   *min_indent_px* 像素，则视为新段落的开始。
2. **垂直间距** -- 若相邻两行的间距超过
   ``avg_line_height * 1.4`` 则插入段落分隔（除非两行并排，
   即存在显著的 y 轴重叠）。
"""

from __future__ import annotations

import re
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from docflow.model.blocks.text_block import TextLine
    from docflow.model.page import CoordMapper

from docflow.utils.constants import PARAGRAPH_INDENT_MIN_PX, PARAGRAPH_GAP_RATIO

_LIST_MARKER_PATTERNS = (
    re.compile(r"^\s*\d+[\.．、\)](?!\d)\s*"),
    re.compile(r"^\s*[A-Za-z][\.\)]\s*"),
    re.compile(r"^\s*[（(]?[一二三四五六七八九十百]+[)）\.、]?\s*"),
    re.compile(r"^\s*[•\-·]\s+"),
)
_SENTENCE_END_RE = re.compile(r"[。！？!?；;…][”\"']?\s*$")


def _is_list_marker_line(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    return any(p.match(stripped) for p in _LIST_MARKER_PATTERNS)


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search((text or "").strip()))


def _looks_like_continuation(prev_text: str, curr_text: str) -> bool:
    prev = (prev_text or "").strip()
    curr = (curr_text or "").strip()
    if not prev or not curr or _ends_sentence(prev) or _is_list_marker_line(curr):
        return False
    if curr[:1] in ",.;:)]}，；：、》」】":
        return True
    if curr[:1].islower():
        return True
    if len(prev) <= 24 or len(curr) <= 24:
        return True
    return False


# ------------------------------------------------------------------
# 首行缩进检测
# ------------------------------------------------------------------

def detect_first_line_indent(
    lines: List["TextLine"],
    mapper: "CoordMapper",
    min_indent_px: float = 15,
) -> float:
    """若首行存在缩进，返回缩进量（Pt）。

    比较首行的 ``x1`` 与其余行的最小 ``x1``。若差值至少
    为 *min_indent_px* 像素，则通过 *mapper* 转换为 Pt 并四舍五入
    到最近的 0.5 pt。

    Parameters
    ----------
    lines:
        OCR 文本行（需暴露 ``x1`` 属性）。
    mapper:
        用于像素到 Pt 转换的 :class:`CoordMapper`。
    min_indent_px:
        被认定为缩进的最小像素差值。

    Returns
    -------
    缩进量（Pt，四舍五入到 0.5 pt），若未检测到缩进则返回 ``0``。
    """
    if len(lines) < 2:
        return 0.0

    first_x1 = lines[0].x1
    if first_x1 is None:
        return 0.0
    rest_x1s = [ln.x1 for ln in lines[1:] if ln.x1 is not None]
    if not rest_x1s:
        return 0.0
    rest_min_x1 = min(rest_x1s)
    indent_px = first_x1 - rest_min_x1

    if indent_px >= min_indent_px:
        indent_pt = mapper.w(indent_px)
        # 四舍五入到最近的 0.5 pt
        return round(indent_pt * 2) / 2
    return 0.0


# ------------------------------------------------------------------
# 段落拆分
# ------------------------------------------------------------------

def split_into_paragraphs(
    lines: List["TextLine"],
    min_indent_px: float = PARAGRAPH_INDENT_MIN_PX,
    list_marker_enabled: bool = True,
) -> List[List["TextLine"]]:
    """将 OCR 文本行序列拆分为段落组。

    当当前行相对于前一行满足以下 **任一** 条件时，开始新段落：

    * 当前行的 ``x1`` 超过前一行的 ``x1`` 至少 *min_indent_px*
      像素（首行缩进）。
    * 两行之间的垂直间距超过
      ``avg_line_height * 1.4``（视觉段落间距）。

    但若相邻两行的 y 轴重叠超过平均行高的 40%，则视为
    并排文本，强制置于 **同一** 段落。

    Parameters
    ----------
    lines:
        按阅读顺序排列的 OCR 文本行，每行暴露
        ``x1``、``y1``、``y2``。
    min_indent_px:
        识别首行缩进的最小水平偏移（像素）。

    Returns
    -------
    段落组列表，每组为 :class:`TextLine` 对象的列表。
    """
    if not lines:
        return []
    if len(lines) == 1:
        return [list(lines)]

    # 计算平均行高（跳过缺失 text_region 的行）
    heights = []
    for ln in lines:
        if ln.y1 is not None and ln.y2 is not None:
            heights.append(ln.y2 - ln.y1)
    if not heights:
        # 无几何信息 -- 将所有行视为单个段落
        return [list(lines)]
    avg_height = sum(heights) / len(heights)

    gap_threshold = avg_height * PARAGRAPH_GAP_RATIO
    side_by_side_threshold = avg_height * 0.4

    paragraphs: List[List["TextLine"]] = []
    current_para: List["TextLine"] = [lines[0]]

    for i in range(1, len(lines)):
        prev = lines[i - 1]
        curr = lines[i]

        # 检查相邻行之间的 y 轴重叠
        prev_y1 = prev.y1
        prev_y2 = prev.y2
        curr_y1 = curr.y1
        curr_y2 = curr.y2
        curr_x1 = curr.x1
        prev_x1 = prev.x1

        # 若缺失几何信息，保持在同一段落
        if any(v is None for v in (prev_y1, prev_y2, curr_y1, curr_y2)):
            current_para.append(curr)
            continue

        y_overlap = min(prev_y2, curr_y2) - max(prev_y1, curr_y1)

        if y_overlap > side_by_side_threshold:
            # 并排行 -> 强制归入同一段落
            current_para.append(curr)
            continue

        # 检查基于缩进的段落分断
        if curr_x1 is not None and prev_x1 is not None:
            indent = curr_x1 - prev_x1
            is_indent_break = indent >= min_indent_px
        else:
            is_indent_break = False

        # 检查基于间距的段落分断
        vertical_gap = curr_y1 - prev_y2
        is_gap_break = vertical_gap > gap_threshold

        # 条目符号起段：1. / A. / （一）等
        curr_text = (curr.text or "").strip()
        is_list_break = bool(list_marker_enabled and _is_list_marker_line(curr_text))
        continuation_hint = _looks_like_continuation(prev.text, curr_text)

        # 句末后左对齐重启：上一行句末 + 当前行回到段首左边界
        para_xs = [ln.x1 for ln in current_para if ln.x1 is not None]
        para_left = min(para_xs) if para_xs else None
        if para_left is not None and curr_x1 is not None:
            left_restart = abs(curr_x1 - para_left) <= max(4.0, min_indent_px * 0.35)
        else:
            left_restart = False
        prev_text = (prev.text or "").strip()
        is_sentence_break = (
            _ends_sentence(prev_text)
            and left_restart
            and vertical_gap > avg_height * 0.08
        )

        if continuation_hint:
            is_indent_break = False
            is_gap_break = False
            is_sentence_break = False

        if is_indent_break or is_gap_break or is_list_break or is_sentence_break:
            paragraphs.append(current_para)
            current_para = [curr]
        else:
            current_para.append(curr)

    # 不要遗漏最后一个段落
    if current_para:
        paragraphs.append(current_para)

    return paragraphs
