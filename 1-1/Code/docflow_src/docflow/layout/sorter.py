"""版面区块的顶层阅读顺序排序。

算法分三个阶段进行：

1. **区域拆分** -- 将页面划分为水平区域（zone）。
2. **候选分栏** -- 在每个区域内按 x 方向聚类得到候选列。
3. **置信回退** -- 若列结构置信不足则回退为单栏，避免错分栏。
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING, Tuple

from docflow.layout.zone_splitter import split_into_zones
from docflow.layout.column_detector import detect_columns, detect_spanned_blocks
from docflow.model.base import BlockType
from docflow.utils.constants import (
    SPAN_ELIGIBLE_TYPES,
    MAX_COLS,
    COLUMN_CLUSTER_THRESH,
)

if TYPE_CHECKING:
    from docflow.model.base import Block


_STRIP_TYPES = frozenset({
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.TITLE,
})
_PARALLEL_ROW_TYPES = frozenset({
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.REFERENCE,
    BlockType.TABLE_CAPTION,
    BlockType.FIGURE_CAPTION,
    BlockType.FORMULA_CAPTION,
    BlockType.TABLE_FOOTNOTE,
})


def _sort_blocks_by_reading_row(blocks: List["Block"], min_row_overlap: float = 6.0) -> List["Block"]:
    """单栏读序排序：先按 y 分行，行内按 x。"""
    if not blocks:
        return []

    by_y = sorted(blocks, key=lambda b: b.bbox.y1)
    rows: List[List["Block"]] = []
    current: List["Block"] = [by_y[0]]
    row_y1 = by_y[0].bbox.y1
    row_y2 = by_y[0].bbox.y2

    for blk in by_y[1:]:
        overlap = min(row_y2, blk.bbox.y2) - max(row_y1, blk.bbox.y1)
        if overlap >= min_row_overlap:
            current.append(blk)
            row_y1 = min(row_y1, blk.bbox.y1)
            row_y2 = max(row_y2, blk.bbox.y2)
        else:
            rows.append(current)
            current = [blk]
            row_y1 = blk.bbox.y1
            row_y2 = blk.bbox.y2
    rows.append(current)

    ordered: List["Block"] = []
    for row in rows:
        if len(row) > 1:
            row.sort(key=lambda b: (b.bbox.x1, b.bbox.y1))
        ordered.extend(row)
    return ordered


def _zone_bounds(blocks: List["Block"]) -> Tuple[float, float, float, float]:
    x1 = min(b.bbox.x1 for b in blocks)
    y1 = min(b.bbox.y1 for b in blocks)
    x2 = max(b.bbox.x2 for b in blocks)
    y2 = max(b.bbox.y2 for b in blocks)
    return x1, y1, x2, y2


def _column_center(col_members: List["Block"]) -> float:
    if not col_members:
        return 0.0
    return sum((b.bbox.x1 + b.bbox.x2) * 0.5 for b in col_members) / len(col_members)


def _y_coverage_ratio(blocks: List["Block"], zone_top: float, zone_bottom: float) -> float:
    if not blocks:
        return 0.0
    zone_h = max(zone_bottom - zone_top, 1.0)
    intervals = sorted([(b.bbox.y1, b.bbox.y2) for b in blocks], key=lambda p: p[0])
    merged = [list(intervals[0])]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    covered = sum(max(0.0, hi - lo) for lo, hi in merged)
    return max(0.0, min(1.0, covered / zone_h))


def _is_effective_text_block(block: "Block") -> bool:
    # 仅排除页眉/页脚/页码和纯图像可跨列类型；其余文本类都参与列置信评分。
    if block.block_type in SPAN_ELIGIBLE_TYPES:
        return False
    return block.block_type not in {
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.ABANDON,
    }


def _estimate_column_confidence(
    columns: List[List["Block"]],
    zone_blocks: List["Block"],
    image_width: int,
    image_height: int | None,
) -> float:
    """对候选列结构打分（0~1），用于多栏保留/回退。"""
    if len(columns) <= 1:
        return 1.0
    if not zone_blocks:
        return 0.0

    _zx1, zy1, _zx2, zy2 = _zone_bounds(zone_blocks)
    zone_h = max(zy2 - zy1, 1.0)
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), 1.0)

    # 1) 每列有效文本块数量
    eff_counts = [sum(1 for b in col if _is_effective_text_block(b)) for col in columns]
    avg_count = sum(eff_counts) / max(len(eff_counts), 1)
    min_count = min(eff_counts) if eff_counts else 0.0
    count_score = (
        0.55 * min(1.0, avg_count / 2.0)
        + 0.45 * min(1.0, min_count / 1.0)
    )

    # 2) 列间距归一化（间距过小通常是假分栏）
    centers = sorted(_column_center(col) for col in columns)
    if len(centers) >= 2:
        gaps = [max(0.0, centers[i + 1] - centers[i]) for i in range(len(centers) - 1)]
        gap_norm = (sum(gaps) / len(gaps)) / page_w
        gap_score = min(1.0, max(0.0, (gap_norm - 0.03) / 0.12))
    else:
        gap_score = 1.0

    # 3) 列内垂直覆盖
    coverages = [_y_coverage_ratio(col, zy1, zy2) for col in columns]
    coverage_score = min(coverages) if coverages else 0.0

    # 4) 列间覆盖平衡度
    max_cov = max(coverages) if coverages else 0.0
    balance_score = (min(coverages) / max_cov) if max_cov > 1e-6 else 0.0

    # 5) 区域高度占比（条带太薄时更倾向单栏）
    zone_ratio = zone_h / page_h if image_height else 1.0
    height_score = min(1.0, max(0.0, zone_ratio / 0.20))

    return (
        0.28 * count_score
        + 0.20 * gap_score
        + 0.22 * coverage_score
        + 0.20 * balance_score
        + 0.10 * height_score
    )


def _should_suppress_strip_multicol(
    zone_blocks: List["Block"],
    image_height: int | None,
    strip_ratio: float,
) -> bool:
    """顶部/底部窄条带中，页眉类区域默认不采用多栏。"""
    if not zone_blocks or not image_height or image_height <= 0:
        return False

    _zx1, zy1, _zx2, zy2 = _zone_bounds(zone_blocks)
    page_h = float(image_height)
    near_top = zy1 <= page_h * strip_ratio
    near_bottom = zy2 >= page_h * (1.0 - strip_ratio)
    if not (near_top or near_bottom):
        return False

    zone_h = max(0.0, zy2 - zy1)
    if zone_h > page_h * (strip_ratio * 1.8):
        return False

    strip_hits = sum(1 for b in zone_blocks if b.block_type in _STRIP_TYPES)
    strip_ratio_actual = strip_hits / max(len(zone_blocks), 1)
    return strip_ratio_actual >= 0.6


def _assign_single_column(blocks: List["Block"]) -> None:
    for blk in blocks:
        blk.col_count = 1
        blk.col_index = 0
        blk.spanned_cols = [0]


def _ordered_by_columns(blocks: List["Block"]) -> List["Block"]:
    col_map: dict[int, List["Block"]] = {}
    for blk in blocks:
        col_map.setdefault(blk.col_index, []).append(blk)
    ordered: List["Block"] = []
    for col_idx in sorted(col_map.keys()):
        ordered.extend(_sort_blocks_by_reading_row(col_map[col_idx]))
    return ordered


def _is_parallel_row_multicol_candidate(
    columns: List[List["Block"]],
    zone_blocks: List["Block"],
    image_width: int,
) -> bool:
    """并排短行场景保留多栏（如 TABLE I / TABLE II 同行）。"""
    if len(columns) < 2 or len(columns) > 4:
        return False
    if len(zone_blocks) < 2 or len(zone_blocks) > 6:
        return False

    # 页眉/页脚条带不走该特例
    strip_hits = sum(1 for b in zone_blocks if b.block_type in _STRIP_TYPES)
    if strip_hits / max(len(zone_blocks), 1) >= 0.5:
        return False

    types_ok = all(b.block_type in _PARALLEL_ROW_TYPES for b in zone_blocks)
    if not types_ok:
        return False

    reps: List["Block"] = []
    for col in columns:
        if not col:
            return False
        if len(col) > 2:
            return False
        reps.append(sorted(col, key=lambda b: b.bbox.y1)[0])

    # 同行重叠约束
    n = len(reps)
    for i in range(n):
        b1 = reps[i]
        h1 = max(1.0, float(b1.bbox.height))
        for j in range(i + 1, n):
            b2 = reps[j]
            h2 = max(1.0, float(b2.bbox.height))
            overlap = min(float(b1.bbox.y2), float(b2.bbox.y2)) - max(float(b1.bbox.y1), float(b2.bbox.y1))
            if overlap < min(h1, h2) * 0.30:
                return False

    centers = sorted((b.bbox.x1 + b.bbox.x2) * 0.5 for b in reps)
    if len(centers) >= 2:
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        mean_gap = sum(gaps) / len(gaps)
        if mean_gap < max(80.0, image_width * 0.08):
            return False

    return True


def sort_layout(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None = None,
    max_cols: int = MAX_COLS,
    cluster_thresh: float = COLUMN_CLUSTER_THRESH,
    column_confidence_min: float = 0.55,
    zone_strip_height_ratio: float = 0.12,
) -> List["Block"]:
    """将 *blocks* 按自然阅读顺序重新排列并填充列元数据。"""
    if not blocks:
        return []

    if len(blocks) <= 1:
        _assign_single_column(blocks)
        return list(blocks)

    zones = split_into_zones(blocks, image_width)
    ordered: List["Block"] = []

    for zone_blocks in zones:
        text_blocks: List["Block"] = []
        fig_blocks: List["Block"] = []
        for blk in zone_blocks:
            if blk.block_type in SPAN_ELIGIBLE_TYPES:
                fig_blocks.append(blk)
            else:
                text_blocks.append(blk)

        all_blocks = text_blocks + fig_blocks
        total_blocks = len(all_blocks)
        if total_blocks <= 1:
            _assign_single_column(all_blocks)
            ordered.extend(_sort_blocks_by_reading_row(all_blocks))
            continue

        # 候选分栏：文本不足时将所有区块纳入候选聚类。
        candidate_source = all_blocks if len(text_blocks) <= 1 else text_blocks
        columns, col_bounds = detect_columns(
            candidate_source,
            image_width,
            max_cols=max_cols,
            cluster_thresh=cluster_thresh,
        )
        col_count = len(columns)

        suppress_strip = _should_suppress_strip_multicol(
            zone_blocks=all_blocks,
            image_height=image_height,
            strip_ratio=zone_strip_height_ratio,
        )
        low_confidence = _estimate_column_confidence(
            columns=columns,
            zone_blocks=all_blocks,
            image_width=image_width,
            image_height=image_height,
        ) < column_confidence_min
        parallel_row_keep = (
            not suppress_strip
            and _is_parallel_row_multicol_candidate(columns, all_blocks, image_width)
        )

        should_force_single = (
            col_count <= 1
            or suppress_strip
            or (low_confidence and not parallel_row_keep)
        )

        if should_force_single:
            _assign_single_column(all_blocks)
            ordered.extend(_sort_blocks_by_reading_row(all_blocks))
            continue

        # 保留多栏：先标注候选分栏中的成员，再补充其余区块。
        for col_idx, col_members in enumerate(columns):
            for blk in col_members:
                blk.col_count = col_count
                blk.col_index = col_idx
                blk.spanned_cols = [col_idx]

        unassigned = [b for b in all_blocks if getattr(b, "col_count", 0) != col_count]
        if unassigned:
            detect_spanned_blocks(unassigned, col_bounds)
            for blk in unassigned:
                blk.col_count = col_count

        if fig_blocks:
            detect_spanned_blocks(fig_blocks, col_bounds)
            for blk in fig_blocks:
                blk.col_count = col_count

        ordered.extend(_ordered_by_columns(all_blocks))

    return ordered
