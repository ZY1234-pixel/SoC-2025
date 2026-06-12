"""多栏版面检测与区块列分配。

对文本区块的左边缘（``bbox.x1``）进行贪心最近邻聚类以检测分栏；
然后根据 x 范围重叠将图片/公式等区块分配到相应列（或标记为跨列）。
"""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from docflow.model.base import Block

from docflow.utils.constants import MAX_COLS, COLUMN_CLUSTER_THRESH

_ROW_CAPTION_TYPES = frozenset({
    "table_caption",
    "figure_caption",
    "formula_caption",
    "table_footnote",
    "reference",
})
_ROW_TEXTLIKE_TYPES = frozenset({
    "text",
    "title",
    "reference",
    "header",
    "footer",
    "page_number",
    "table_caption",
    "figure_caption",
    "formula_caption",
    "table_footnote",
})
_CAPTION_TAG_ONLY_RE = (
    r"^\s*(表|图|公式|TABLE|FIG(?:URE)?|EQ(?:UATION)?)\s*"
    r"[A-Za-z0-9一二三四五六七八九十\-\._]*\s*$"
)


def _block_type_value(block: "Block") -> str:
    return getattr(getattr(block, "block_type", None), "value", str(getattr(block, "block_type", "")))


def _block_text(block: "Block") -> str:
    if hasattr(block, "full_text"):
        try:
            return str(block.full_text() or "")
        except Exception:
            return ""
    lines = getattr(block, "lines", None) or []
    return "".join(getattr(ln, "text", "") for ln in lines)


def _line_count(block: "Block") -> int:
    if hasattr(block, "count_lines"):
        try:
            return int(max(1, block.count_lines()))
        except Exception:
            return 1
    return 1


def _looks_like_caption_fragment_pair(blocks: List["Block"]) -> bool:
    """判断是否像“编号标签 + 标题正文”的 OCR 拆分碎片。"""
    import re

    texts = [_block_text(b).strip() for b in blocks]
    if len(texts) != 2:
        lengths = sorted(len(t) for t in texts if t)
        if len(lengths) >= 2:
            return lengths[0] <= 18 and lengths[-1] >= 24
        return False

    t1, t2 = texts
    if not t1 or not t2:
        return False

    m1 = bool(re.match(_CAPTION_TAG_ONLY_RE, t1, flags=re.IGNORECASE))
    m2 = bool(re.match(_CAPTION_TAG_ONLY_RE, t2, flags=re.IGNORECASE))
    if m1 ^ m2:
        return True

    l1, l2 = len(t1), len(t2)
    if min(l1, l2) <= 18 and max(l1, l2) >= 24:
        return True

    # TABLE I / TABLE II 这类并排独立标题不应合并。
    if m1 and m2:
        return False

    return False


def _should_collapse_single_row_fragments(blocks: List["Block"], image_width: int) -> bool:
    """判断同一视觉行的短文本碎片是否应合并为单栏。"""
    if len(blocks) < 2 or len(blocks) > 4:
        return False

    type_values = {_block_type_value(b) for b in blocks}
    if not type_values:
        return False

    heights = [max(1.0, float(b.bbox.height)) for b in blocks]
    heights_sorted = sorted(heights)
    median_h = heights_sorted[len(heights_sorted) // 2]
    band_top = min(float(b.bbox.y1) for b in blocks)
    band_bottom = max(float(b.bbox.y2) for b in blocks)
    if (band_bottom - band_top) > median_h * 1.8:
        return False

    # 同行约束：任意两块都应有一定 y 重叠
    n = len(blocks)
    for i in range(n):
        b1 = blocks[i]
        h1 = max(1.0, float(b1.bbox.height))
        for j in range(i + 1, n):
            b2 = blocks[j]
            h2 = max(1.0, float(b2.bbox.height))
            overlap = min(float(b1.bbox.y2), float(b2.bbox.y2)) - max(float(b1.bbox.y1), float(b2.bbox.y1))
            if overlap < min(h1, h2) * 0.35:
                return False

    # 1) 传统 caption 场景：直接合并为单栏。
    if type_values.issubset(_ROW_CAPTION_TYPES):
        return _looks_like_caption_fragment_pair(blocks)

    # 2) 通用短文本碎片：同一行、低条带、块数少时也应优先单栏。
    if not type_values.issubset(_ROW_TEXTLIKE_TYPES):
        return False

    page_w = max(float(image_width), 1.0)
    span_w = max(float(b.bbox.x2) for b in blocks) - min(float(b.bbox.x1) for b in blocks)
    if span_w > page_w * 0.82:
        return False

    ordered = sorted(blocks, key=lambda b: (float(b.bbox.x1), float(b.bbox.x2)))
    gaps = [
        max(0.0, float(ordered[idx + 1].bbox.x1) - float(ordered[idx].bbox.x2))
        for idx in range(len(ordered) - 1)
    ]
    # 并列独立标题/短块通常有明显大间隙，不应被视为 OCR 碎片。
    if gaps:
        max_gap = max(gaps)
        gap_limit = max(page_w * 0.055, median_h * 3.2)
        if max_gap > gap_limit:
            return False

    for b in blocks:
        bw = max(1.0, float(b.bbox.width))
        if bw > page_w * 0.48:
            return False
        if _line_count(b) > 2:
            return False
        text_len = len(_block_text(b).strip())
        if text_len > 42:
            return False

    return True


def _x_center(block: "Block") -> float:
    return (float(block.bbox.x1) + float(block.bbox.x2)) * 0.5


def _is_textlike_for_skeleton(block: "Block") -> bool:
    return _block_type_value(block) in _ROW_TEXTLIKE_TYPES


def _is_narrow_body_candidate(block: "Block", image_width: int) -> bool:
    page_w = max(float(image_width), 1.0)
    width = max(1.0, float(block.bbox.width))
    text = _block_text(block).strip()
    if not _is_textlike_for_skeleton(block):
        return False
    if _block_type_value(block) in _ROW_CAPTION_TYPES:
        return False
    # 极短的单字/单标签更像局部标记，不应成为全局列骨架锚点。
    if width <= page_w * 0.10 and _line_count(block) <= 1 and len(text) <= 8:
        return False
    # 先用更稳定的窄正文骨架做全局列提案，避免局部宽块把相邻列粘连。
    return width <= page_w * 0.32


def _cluster_by_centers(
    blocks: List["Block"],
    image_width: int,
    *,
    max_cols: int,
    cluster_thresh: float,
) -> List[Tuple[float, List["Block"]]]:
    if not blocks:
        return []

    threshold = max(float(image_width) * cluster_thresh, 1.0)
    columns: List[Tuple[float, List["Block"]]] = []
    for blk in sorted(blocks, key=lambda b: (_x_center(b), float(b.bbox.x1), float(b.bbox.y1))):
        center = _x_center(blk)
        best_idx = -1
        best_dist = float("inf")
        for idx, (avg_center, _members) in enumerate(columns):
            dist = abs(center - avg_center)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx >= 0 and best_dist < threshold:
            avg_center, members = columns[best_idx]
            members.append(blk)
            columns[best_idx] = (
                (avg_center * (len(members) - 1) + center) / len(members),
                members,
            )
        else:
            columns.append((center, [blk]))

    columns.sort(key=lambda item: item[0])
    while len(columns) > max_cols:
        min_gap = float("inf")
        merge_idx = 0
        for i in range(len(columns) - 1):
            gap = columns[i + 1][0] - columns[i][0]
            if gap < min_gap:
                min_gap = gap
                merge_idx = i
        avg1, members1 = columns[merge_idx]
        avg2, members2 = columns[merge_idx + 1]
        merged_members = members1 + members2
        merged_avg = (avg1 * len(members1) + avg2 * len(members2)) / max(len(merged_members), 1)
        columns[merge_idx] = (merged_avg, merged_members)
        del columns[merge_idx + 1]
    return columns


def _cluster_by_left_edges(
    blocks: List["Block"],
    image_width: int,
    *,
    max_cols: int,
    cluster_thresh: float,
) -> List[Tuple[float, List["Block"]]]:
    if not blocks:
        return []

    sorted_blocks = sorted(blocks, key=lambda b: float(b.bbox.x1))
    threshold = max(float(image_width) * cluster_thresh, 1.0)
    columns: List[Tuple[float, List["Block"]]] = []

    for blk in sorted_blocks:
        x1 = float(blk.bbox.x1)
        x_center = _x_center(blk)
        best_idx = -1
        best_dist = float("inf")

        for idx, (avg_x1, _members) in enumerate(columns):
            dist = abs(x1 - avg_x1)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_dist < threshold and best_idx >= 0:
            avg_x1, members = columns[best_idx]
            members.append(blk)
            columns[best_idx] = (
                (avg_x1 * (len(members) - 1) + x1) / len(members),
                members,
            )
            continue

        contained_idx = -1
        for idx, (_avg_x1, members) in enumerate(columns):
            col_x1 = min(float(b.bbox.x1) for b in members)
            col_x2 = max(float(b.bbox.x2) for b in members)
            if col_x1 <= x_center <= col_x2:
                contained_idx = idx
                break
        if contained_idx >= 0:
            avg_x1, members = columns[contained_idx]
            members.append(blk)
            columns[contained_idx] = (
                (avg_x1 * (len(members) - 1) + x1) / len(members),
                members,
            )
        else:
            columns.append((x1, [blk]))

    def _col_center(col_tuple: Tuple[float, List["Block"]]) -> float:
        _avg_x1, members = col_tuple
        return sum(_x_center(b) for b in members) / max(len(members), 1)

    columns.sort(key=_col_center)
    while len(columns) > max_cols:
        min_gap = float("inf")
        merge_idx = 0
        for i in range(len(columns) - 1):
            gap = _col_center(columns[i + 1]) - _col_center(columns[i])
            if gap < min_gap:
                min_gap = gap
                merge_idx = i

        avg1, members1 = columns[merge_idx]
        avg2, members2 = columns[merge_idx + 1]
        merged_members = members1 + members2
        merged_avg = (avg1 * len(members1) + avg2 * len(members2)) / max(len(merged_members), 1)
        columns[merge_idx] = (merged_avg, merged_members)
        del columns[merge_idx + 1]
    return columns


# ------------------------------------------------------------------
# 列检测
# ------------------------------------------------------------------

def detect_columns(
    text_blocks: List["Block"],
    image_width: int,
    max_cols: int = MAX_COLS,
    cluster_thresh: float = COLUMN_CLUSTER_THRESH,
) -> Tuple[List[List["Block"]], List[Tuple[float, float]]]:
    """按左边缘距离将 *text_blocks* 聚类为列。

    Parameters
    ----------
    text_blocks:
        单个区域内的文本区块（已与图片区块分离）。
    image_width:
        完整图像宽度（像素）。
    max_cols:
        允许的最大列数。若检测到更多列，则反复合并最近的
        相邻对直到满足限制。
    cluster_thresh:
        用作分配区块到已有列或创建新列的距离阈值，
        为 *image_width* 的比例。

    Returns
    -------
    columns:
        列组列表（每组为区块列表）。列按平均中心 x 从左到右排序。
    col_bounds:
        每列一个 ``(x1_min, x2_robust)`` 元组。*x2_robust* 为第二大的
        ``bbox.x2``，以排除突出到边距的异常区块。
    """
    if not text_blocks:
        return [], []

    if _should_collapse_single_row_fragments(text_blocks, image_width):
        members = sorted(text_blocks, key=lambda b: b.bbox.x1)
        x1_min = min(b.bbox.x1 for b in members)
        x2_values = sorted([b.bbox.x2 for b in members], reverse=True)
        x2_robust = x2_values[1] if len(x2_values) > 1 else x2_values[0]
        return [members], [(x1_min, x2_robust)]

    skeleton_source = [blk for blk in text_blocks if _is_narrow_body_candidate(blk, image_width)]
    if len(skeleton_source) >= 3:
        columns = _cluster_by_centers(
            skeleton_source,
            image_width,
            max_cols=max_cols,
            cluster_thresh=min(cluster_thresh, 0.08),
        )
    else:
        columns = _cluster_by_left_edges(
            text_blocks,
            image_width,
            max_cols=max_cols,
            cluster_thresh=cluster_thresh,
        )

    if not columns:
        return [], []

    if len(columns) == 1 and len(skeleton_source) >= 2:
        relaxed = _cluster_by_centers(
            skeleton_source,
            image_width,
            max_cols=max_cols,
            cluster_thresh=min(cluster_thresh, 0.055),
        )
        if len(relaxed) > len(columns):
            columns = relaxed

    def _col_center(col_tuple: Tuple[float, List["Block"]]) -> float:
        _avg, members = col_tuple
        return sum(_x_center(b) for b in members) / max(len(members), 1)

    # 构建输出列表
    col_blocks: List[List["Block"]] = []
    col_bounds: List[Tuple[float, float]] = []

    for _avg_x1, members in sorted(columns, key=_col_center):
        col_blocks.append(members)
        x1_min = min(b.bbox.x1 for b in members)
        x2_values = sorted([b.bbox.x2 for b in members], reverse=True)
        # 使用第二大 x2 排除异常值；兜底用最大值
        x2_robust = x2_values[1] if len(x2_values) > 1 else x2_values[0]
        col_bounds.append((x1_min, x2_robust))

    # 合并 X 范围明显重叠的相邻列（防止窄块被错误聚成独立列）
    merged_blocks: List[List["Block"]] = []
    merged_bounds: List[Tuple[float, float]] = []
    for i, (blocks_i, (x1_i, x2_i)) in enumerate(zip(col_blocks, col_bounds)):
        if merged_bounds:
            prev_blocks = merged_blocks[-1]
            _px1, px2 = merged_bounds[-1]
            overlap = min(x2_i, px2) - max(x1_i, _px1)
            narrower = min(x2_i - x1_i, px2 - _px1)
            if narrower > 0 and overlap > narrower * 0.30:
                merged_blocks[-1] = prev_blocks + blocks_i
                merged_bounds[-1] = (min(_px1, x1_i), max(px2, x2_i))
                continue
        merged_blocks.append(blocks_i)
        merged_bounds.append((x1_i, x2_i))

    return merged_blocks, merged_bounds


# ------------------------------------------------------------------
# 图片/公式等区块的跨列分配
# ------------------------------------------------------------------

def detect_spanned_blocks(
    fig_blocks: List["Block"],
    col_bounds: List[Tuple[float, float]],
) -> None:
    """确定每个图片/公式区块横跨的列。

    对 *fig_blocks* 中的每个区块设置：

    * ``block.spanned_cols`` -- 区块重叠的列索引列表
    * ``block.col_index``    -- 第一个（最左）重叠列

    若区块未与任何列边界重叠，则按中心 x 距离分配到最近的列。

    Parameters
    ----------
    fig_blocks:
        区域内的非文本区块（图片、公式、说明等）。
    col_bounds:
        :func:`detect_columns` 返回的列边界。
    """
    if not col_bounds:
        # 无列信息 -- 全部分配到第 0 列
        for blk in fig_blocks:
            blk.spanned_cols = [0]
            blk.col_index = 0
        return

    for blk in fig_blocks:
        bx1 = blk.bbox.x1
        bx2 = blk.bbox.x2
        overlapping: List[int] = []

        for col_idx, (cx1, cx2) in enumerate(col_bounds):
            # 检查 x 范围交集
            if bx1 < cx2 and bx2 > cx1:
                overlapping.append(col_idx)

        if not overlapping:
            # 按中心距离分配到最近的列
            blk_center = (bx1 + bx2) / 2
            best_col = 0
            best_dist = float("inf")
            for col_idx, (cx1, cx2) in enumerate(col_bounds):
                col_center = (cx1 + cx2) / 2
                dist = abs(blk_center - col_center)
                if dist < best_dist:
                    best_dist = dist
                    best_col = col_idx
            overlapping = [best_col]

        blk.spanned_cols = overlapping
        blk.col_index = overlapping[0]
