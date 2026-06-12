"""XY-Cut++ sorter following the original paper's 4-stage pipeline.

1. Cross-layout detection via beta * median(width).
2. Pre-mask title / figure / table / formula / caption / strip elements.
3. Coarse Y-band segmentation using wide masked barriers, then adaptive
   XY/YX-Cut recursive sorting within each band.
4. Semantic + geometry-aware restoration of masked elements.
5. Column metadata assignment for downstream renderers.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import median
from typing import Iterable, List, Optional, Sequence, TYPE_CHECKING

from docflow.layout.column_detector import detect_columns, detect_spanned_blocks
from docflow.model.base import BlockType
from docflow.utils.constants import COLUMN_CLUSTER_THRESH, MAX_COLS, SPAN_ELIGIBLE_TYPES

if TYPE_CHECKING:  # pragma: no cover
    from docflow.model.base import Block


# ---------------------------------------------------------------------------
# Type groups
# ---------------------------------------------------------------------------

_TEXTLIKE_TYPES = frozenset({
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.REFERENCE,
    BlockType.ABSTRACT,
    BlockType.CODE,
    BlockType.LIST,
    BlockType.FOOTNOTE,
    BlockType.FIGURE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.FORMULA_CAPTION,
})

_STRIP_TYPES = frozenset({
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
})

_CAPTION_TYPES = frozenset({
    BlockType.FIGURE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.FORMULA_CAPTION,
})

_VISUAL_TYPES = frozenset({
    BlockType.FIGURE,
    BlockType.TABLE,
    BlockType.FORMULA,
    BlockType.EQUATION,
})

_DYNAMIC_MASK_TYPES = frozenset({
    BlockType.TITLE,
    BlockType.FIGURE,
    BlockType.TABLE,
    BlockType.FORMULA,
    BlockType.EQUATION,
    BlockType.FIGURE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.FORMULA_CAPTION,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
})

_COLUMN_EXCLUDED_TYPES = _VISUAL_TYPES | _CAPTION_TYPES | _STRIP_TYPES
_COLUMN_ANCHOR_TYPES = frozenset({
    BlockType.TEXT,
    BlockType.REFERENCE,
    BlockType.ABSTRACT,
    BlockType.CODE,
    BlockType.LIST,
    BlockType.FOOTNOTE,
})


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Cut:
    axis: str
    position: float
    gap: float


def _attrs(block: "Block") -> dict:
    if block.attributes is None:
        block.attributes = {}
    return block.attributes.setdefault("xycutpp_proto", {})


def _mark(block: "Block", **values: object) -> None:
    _attrs(block).update(values)


def _marked(block: "Block", key: str, default: object = None) -> object:
    attrs = getattr(block, "attributes", None) or {}
    proto = attrs.get("xycutpp_proto", {}) if isinstance(attrs, dict) else {}
    return proto.get(key, default) if isinstance(proto, dict) else default


def _x1(block: "Block") -> float:
    return float(block.bbox.x1)


def _y1(block: "Block") -> float:
    return float(block.bbox.y1)


def _x2(block: "Block") -> float:
    return float(block.bbox.x2)


def _y2(block: "Block") -> float:
    return float(block.bbox.y2)


def _w(block: "Block") -> float:
    return max(0.0, float(block.bbox.width))


def _h(block: "Block") -> float:
    return max(0.0, float(block.bbox.height))


def _area(block: "Block") -> float:
    return max(0.0, float(block.bbox.area))


def _cx(block: "Block") -> float:
    return (_x1(block) + _x2(block)) * 0.5


def _cy(block: "Block") -> float:
    return (_y1(block) + _y2(block)) * 0.5


def _overlap_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _projection_overlap_ratio_x(a: "Block", b: "Block") -> float:
    overlap = _overlap_1d(_x1(a), _x2(a), _x1(b), _x2(b))
    return overlap / max(1.0, min(_w(a), _w(b)))


def _projection_overlap_ratio_y(a: "Block", b: "Block") -> float:
    overlap = _overlap_1d(_y1(a), _y2(a), _y1(b), _y2(b))
    return overlap / max(1.0, min(_h(a), _h(b)))


def _edge_gap(a: "Block", b: "Block") -> float:
    dx = max(0.0, max(_x1(a), _x1(b)) - min(_x2(a), _x2(b)))
    dy = max(0.0, max(_y1(a), _y1(b)) - min(_y2(a), _y2(b)))
    return (dx * dx + dy * dy) ** 0.5


def _sort_yx(blocks: Iterable["Block"]) -> List["Block"]:
    return sorted(blocks, key=lambda b: (_y1(b), _x1(b), _y2(b), _x2(b)))


def _sort_xy(blocks: Iterable["Block"]) -> List["Block"]:
    return sorted(blocks, key=lambda b: (_x1(b), _y1(b), _x2(b), _y2(b)))


def _block_id(block: "Block") -> str:
    raw = getattr(block, "block_id", "")
    return str(raw) if raw else f"@{id(block)}"


def _line_count(block: "Block") -> int:
    if hasattr(block, "count_lines"):
        try:
            return max(1, int(block.count_lines()))
        except Exception:
            return 1
    return 1


def _block_text(block: "Block") -> str:
    if hasattr(block, "full_text"):
        try:
            return str(block.full_text() or "")
        except Exception:
            return ""
    return ""


def _full_text(block: "Block") -> str:
    if hasattr(block, "full_text"):
        try:
            return str(block.full_text() or "")
        except Exception:
            return ""
    lines = getattr(block, "lines", None) or []
    return "".join(str(getattr(line, "text", "") or "") for line in lines)


# ---------------------------------------------------------------------------
# Helpers kept for postprocessing compatibility
# ---------------------------------------------------------------------------

def _merged_y_coverage(blocks: Sequence["Block"], page_h: float) -> float:
    if not blocks or page_h <= 0.0:
        return 0.0
    intervals = sorted((_y1(blk), _y2(blk)) for blk in blocks if _y2(blk) > _y1(blk))
    if not intervals:
        return 0.0
    merged: List[List[float]] = [[intervals[0][0], intervals[0][1]]]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    covered = sum(max(0.0, hi - lo) for lo, hi in merged)
    return max(0.0, min(1.0, covered / page_h))


def _intersects_region(block: "Block", region: tuple[float, float, float, float]) -> bool:
    rx1, ry1, rx2, ry2 = region
    return (
        _x1(block) < rx2
        and _x2(block) > rx1
        and _y1(block) < ry2
        and _y2(block) > ry1
    )


def _has_column_local_text_neighbor(
    block: "Block",
    blocks: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> bool:
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max(_y2(block), 1.0))
    max_gap = max(48.0, page_h * 0.10)
    for other in blocks:
        if other is block:
            continue
        if other.block_type not in _TEXTLIKE_TYPES or other.block_type in _STRIP_TYPES:
            continue
        if _w(other) > page_w * 0.60:
            continue
        if _projection_overlap_ratio_x(block, other) < 0.22:
            continue
        vertical_gap = min(
            max(0.0, _y1(other) - _y2(block)),
            max(0.0, _y1(block) - _y2(other)),
        )
        if vertical_gap <= max_gap:
            return True
    return False


def _has_locked_global_multicol_skeleton(
    blocks: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> bool:
    """Return true when the core has already established a stable 4-column skeleton.

    Postprocess rules may repair local family ordering, but once a stable
    4-column newspaper or magazine skeleton is present they should not
    rewrite column metadata or split the page into synthetic sub-structures.
    """
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(blk) for blk in blocks), default=1.0))
    bodylike = [
        blk for blk in blocks
        if blk.block_type in {
            BlockType.TEXT,
            BlockType.TITLE,
            BlockType.REFERENCE,
            BlockType.ABSTRACT,
        }
        and blk.block_type not in _STRIP_TYPES
        and blk.block_type not in _CAPTION_TYPES
        and int(getattr(blk, "col_count", 1) or 1) >= 4
        and len(getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]) == 1
        and _w(blk) <= page_w * 0.48
    ]
    if len(bodylike) < 8:
        return False

    by_col: dict[int, List["Block"]] = {}
    for blk in bodylike:
        col = int((getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)])[0])
        by_col.setdefault(col, []).append(blk)
    if len(by_col) < 3:
        return False

    substantial_cols = 0
    for members in by_col.values():
        if len(members) >= 2 and _merged_y_coverage(members, page_h) >= 0.08:
            substantial_cols += 1

    return substantial_cols >= 4


# ---------------------------------------------------------------------------
# Phase 1: cross-layout detection
# ---------------------------------------------------------------------------


def _candidate_widths_for_median(blocks: Sequence["Block"], image_width: int) -> List[float]:
    """Collect widths from narrow body-text blocks for a stable median.

    Wide visuals, captions, and strip blocks are excluded so they don't inflate
    the median on sparse pages.
    """
    page_w = max(float(image_width), 1.0)
    widths: List[float] = []
    for blk in blocks:
        if blk.block_type in _STRIP_TYPES:
            continue
        if blk.block_type in _VISUAL_TYPES and _w(blk) >= page_w * 0.55:
            continue
        if blk.block_type in _CAPTION_TYPES:
            continue
        if blk.block_type not in {BlockType.TEXT, BlockType.TITLE, BlockType.REFERENCE, BlockType.ABSTRACT}:
            continue
        widths.append(_w(blk))
    return widths or [_w(blk) for blk in blocks if blk.block_type not in _STRIP_TYPES and blk.block_type not in _VISUAL_TYPES]


def _detect_cross_layout_blocks(
    blocks: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int] = None,
    beta: float = 1.3,
    min_projection_overlap: int = 2,
    overlap_threshold: float = 0.10,
) -> set[int]:
    if len(blocks) < 3:
        return set()

    # Only detect cross-layout when there are blocks that don't overlap in X,
    # indicating actual column structure. Without columns the concept is meaningless.
    text_blocks = [b for b in blocks if b.block_type in _TEXTLIKE_TYPES and b.block_type not in _CAPTION_TYPES]
    has_columns = False
    for i, a in enumerate(text_blocks):
        for j in range(i + 1, len(text_blocks)):
            if _projection_overlap_ratio_x(a, text_blocks[j]) < overlap_threshold:
                has_columns = True
                break
        if has_columns:
            break
    if not has_columns:
        for blk in blocks:
            _mark(blk, cross_candidate=False, cross_threshold=0.0, cross_overlap_count=0)
        return set()

    widths = _candidate_widths_for_median(blocks, image_width)
    med_w = max(float(median(widths)), 1.0)
    threshold = beta * med_w
    page_w = max(float(image_width), 1.0)

    cross_ids: set[int] = set()
    for blk in blocks:
        if blk.block_type not in _TEXTLIKE_TYPES or blk.block_type in _CAPTION_TYPES:
            _mark(blk, cross_candidate=False, cross_threshold=round(threshold, 2))
            continue

        bw = _w(blk)
        length_hit = bw > threshold or bw >= page_w * 0.58
        if not length_hit:
            _mark(blk, cross_candidate=False, cross_threshold=round(threshold, 2))
            continue

        overlap_count = 0
        for other in blocks:
            if other is blk:
                continue
            if _projection_overlap_ratio_x(blk, other) >= overlap_threshold:
                overlap_count += 1
                if overlap_count >= min_projection_overlap:
                    break

        is_cross = overlap_count >= min_projection_overlap
        _mark(
            blk,
            cross_candidate=is_cross,
            cross_threshold=round(threshold, 2),
            cross_overlap_count=overlap_count,
        )
        if is_cross:
            cross_ids.add(id(blk))
    return cross_ids


# ---------------------------------------------------------------------------
# Phase 2: pre-mask and coarse segmentation
# ---------------------------------------------------------------------------


def _is_isolated_central_dynamic(
    block: "Block",
    blocks: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
    near_text_margin_ratio: float,
) -> bool:
    if block.block_type not in _DYNAMIC_MASK_TYPES:
        return False
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), 1.0)
    if page_h <= 1.0:
        return False

    ratio = _w(block) / max(_h(block), 1.0)
    normalizer = page_w if ratio < 3.0 else page_h
    dist = ((_cx(block) - page_w * 0.5) ** 2 + (_cy(block) - page_h * 0.5) ** 2) ** 0.5
    near_center = dist / max(normalizer, 1.0) <= 0.20
    if not near_center:
        return False

    margin = max(page_w, page_h) * near_text_margin_ratio
    min_text_gap = float("inf")
    for other in blocks:
        if other is block or other.block_type != BlockType.TEXT:
            continue
        min_text_gap = min(min_text_gap, _edge_gap(block, other))
    return min_text_gap > margin


def _is_top_attachment(
    block: "Block",
    blocks: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> bool:
    """Detect narrow, short text blocks immediately below a title (e.g. byline)."""
    if block.block_type not in _TEXTLIKE_TYPES or block.block_type in _DYNAMIC_MASK_TYPES:
        return False
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), _y2(block), 1.0)
    if _y1(block) > page_h * 0.15:
        return False
    if _w(block) > page_w * 0.55:
        return False
    if _line_count(block) > 1:
        return False
    if _h(block) > page_h * 0.04:
        return False
    for other in blocks:
        if other is block:
            continue
        if other.block_type != BlockType.TITLE:
            continue
        if _y2(other) > _y1(block):
            continue
        vertical_gap = _y1(block) - _y2(other)
        if vertical_gap > max(24.0, page_h * 0.02):
            continue
        if _projection_overlap_ratio_x(block, other) >= 0.25:
            return True
    return False


def _split_mask_sets(
    blocks: Sequence["Block"],
    *,
    cross_ids: set[int],
    image_width: int,
    image_height: Optional[int],
    near_text_margin_ratio: float,
) -> tuple[List["Block"], List["Block"]]:
    active: List["Block"] = []
    masked: List["Block"] = []
    for blk in blocks:
        is_cross = id(blk) in cross_ids
        is_top_att = _is_top_attachment(
            blk,
            blocks,
            image_width=image_width,
            image_height=image_height,
        )
        should_mask = (
            is_cross
            or blk.block_type in _DYNAMIC_MASK_TYPES
            or _is_isolated_central_dynamic(
                blk,
                blocks,
                image_width=image_width,
                image_height=image_height,
                near_text_margin_ratio=near_text_margin_ratio,
            )
            or is_top_att
        )
        if should_mask:
            masked.append(blk)
            reason = blk.block_type.value if blk.block_type in _DYNAMIC_MASK_TYPES else ("top_attachment" if is_top_att else "cross" if is_cross else blk.block_type.value)
            _mark(blk, phase="pre_mask", is_cross_layout=is_cross, mask_reason=reason)
        else:
            active.append(blk)
            _mark(blk, phase="anchor", is_cross_layout=False)
    return active, masked


def _wide_barriers(
    masked: Sequence["Block"],
    *,
    image_width: int,
    barrier_width_ratio: float,
) -> List["Block"]:
    page_w = max(float(image_width), 1.0)
    barriers: List["Block"] = []
    for blk in masked:
        if blk.block_type in _STRIP_TYPES:
            continue
        if _w(blk) >= page_w * barrier_width_ratio:
            barriers.append(blk)
    return _sort_yx(barriers)


def _coarse_y_bands(
    active: Sequence["Block"],
    masked: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
    barrier_width_ratio: float,
    min_band_gap: float,
) -> List[List["Block"]]:
    if not active:
        return []
    page_h = max(float(image_height or 0), max((_y2(b) for b in active), default=1.0))
    barriers = _wide_barriers(masked, image_width=image_width, barrier_width_ratio=barrier_width_ratio)
    if not barriers:
        return [list(active)]

    intervals: List[tuple[float, float]] = []
    cursor = 0.0
    for bar in barriers:
        top = _y1(bar)
        bottom = _y2(bar)
        if top - cursor >= min_band_gap:
            intervals.append((cursor, top))
        cursor = max(cursor, bottom)
    if page_h - cursor >= min_band_gap:
        intervals.append((cursor, page_h))

    groups: List[List["Block"]] = []
    used: set[int] = set()
    for y_top, y_bottom in intervals:
        group = [blk for blk in active if id(blk) not in used and _y1(blk) < y_bottom and _y2(blk) > y_top]
        if group:
            groups.append(group)
            used.update(id(blk) for blk in group)

    leftovers = [blk for blk in active if id(blk) not in used]
    if leftovers:
        groups.append(leftovers)
    return groups or [list(active)]


# ---------------------------------------------------------------------------
# Phase 3: adaptive recursive segmentation
# ---------------------------------------------------------------------------


def _split_by_cut(blocks: Sequence["Block"], cut: _Cut) -> tuple[List["Block"], List["Block"]] | None:
    first: List["Block"] = []
    second: List["Block"] = []
    if cut.axis == "y":
        for blk in blocks:
            (first if _cy(blk) < cut.position else second).append(blk)
    else:
        for blk in blocks:
            (first if _cx(blk) < cut.position else second).append(blk)
    if not first or not second:
        return None
    return first, second


def _best_gap_cut(blocks: Sequence["Block"], axis: str) -> _Cut:
    if len(blocks) <= 1:
        return _Cut(axis=axis, position=0.0, gap=0.0)

    if axis == "x":
        intervals = sorted((_x1(b), _x2(b)) for b in blocks)
    else:
        intervals = sorted((_y1(b), _y2(b)) for b in blocks)

    running_end = intervals[0][1]
    best_gap = 0.0
    best_pos = 0.0
    for start, end in intervals[1:]:
        if start > running_end:
            gap = start - running_end
            if gap > best_gap:
                best_gap = gap
                best_pos = (running_end + start) * 0.5
        running_end = max(running_end, end)
    return _Cut(axis=axis, position=best_pos, gap=best_gap)


def _density_tau(blocks: Sequence["Block"], cross_ids: set[int]) -> float:
    cross_area = sum(_area(b) for b in blocks if id(b) in cross_ids)
    single_area = sum(_area(b) for b in blocks if id(b) not in cross_ids)
    if single_area <= 0.0:
        return float("inf") if cross_area > 0.0 else 0.0
    return cross_area / single_area


def _fallback_sort_when_unsplittable(
    blocks: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int] = None,
) -> List["Block"]:
    """Stable fallback for tiny or highly overlapping regions.

    If columns are obvious, sort column-major; otherwise row-major.
    """
    if len(blocks) <= 2:
        return _sort_yx(blocks)
    columns, _ = detect_columns(
        [b for b in blocks if b.block_type not in _COLUMN_EXCLUDED_TYPES] or list(blocks),
        image_width=image_width,
        max_cols=MAX_COLS,
        cluster_thresh=min(COLUMN_CLUSTER_THRESH, 0.08),
    )
    if len(columns) >= 2 and len(blocks) >= 4:
        by_ids = {id(b) for col in columns for b in col}
        ordered: List["Block"] = []
        for col in columns:
            ordered.extend(_sort_yx(col))
        ordered.extend(_sort_yx([b for b in blocks if id(b) not in by_ids]))
        return ordered
    return _sort_yx(blocks)


def _recursive_adaptive_sort(
    blocks: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
    cross_ids: set[int],
    density_threshold: float,
    min_gap_px: float,
    depth: int = 0,
    max_depth: int = 64,
) -> List["Block"]:
    blocks = list(blocks)
    if len(blocks) <= 1:
        return blocks
    if depth >= max_depth:
        return _fallback_sort_when_unsplittable(blocks, image_width=image_width, image_height=image_height)

    tau = _density_tau(blocks, cross_ids)
    # Paper: XY-Cut (horizontal split = Y axis) when cross-layout dominates,
    # YX-Cut (vertical split = X axis) otherwise.
    primary_axis = "y" if tau > density_threshold else "x"
    secondary_axis = "x" if primary_axis == "y" else "y"

    cuts = {
        "x": _best_gap_cut(blocks, "x"),
        "y": _best_gap_cut(blocks, "y"),
    }
    chosen: Optional[_Cut] = None
    for axis in (primary_axis, secondary_axis):
        if cuts[axis].gap >= min_gap_px:
            chosen = cuts[axis]
            break

    if chosen is None:
        best = cuts["x"] if cuts["x"].gap >= cuts["y"].gap else cuts["y"]
        if len(blocks) >= 5 and best.gap >= max(3.0, min_gap_px * 0.45):
            chosen = best
        else:
            return _fallback_sort_when_unsplittable(blocks, image_width=image_width, image_height=image_height)

    split = _split_by_cut(blocks, chosen)
    if split is None:
        return _fallback_sort_when_unsplittable(blocks, image_width=image_width, image_height=image_height)

    first, second = split
    for blk in blocks:
        _mark(
            blk,
            last_cut_axis=chosen.axis,
            last_cut_pos=round(chosen.position, 2),
            last_cut_gap=round(chosen.gap, 2),
            last_density_tau=("inf" if tau == float("inf") else round(tau, 4)),
        )
    return (
        _recursive_adaptive_sort(
            first,
            image_width=image_width,
            image_height=image_height,
            cross_ids=cross_ids,
            density_threshold=density_threshold,
            min_gap_px=min_gap_px,
            depth=depth + 1,
            max_depth=max_depth,
        )
        + _recursive_adaptive_sort(
            second,
            image_width=image_width,
            image_height=image_height,
            cross_ids=cross_ids,
            density_threshold=density_threshold,
            min_gap_px=min_gap_px,
            depth=depth + 1,
            max_depth=max_depth,
        )
    )


def _sort_active_anchors(
    active: Sequence["Block"],
    masked: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
    cross_ids: set[int],
    density_threshold: float,
    min_gap_px: float,
    barrier_width_ratio: float,
) -> List["Block"]:
    bands = _coarse_y_bands(
        active,
        masked,
        image_width=image_width,
        image_height=image_height,
        barrier_width_ratio=barrier_width_ratio,
        min_band_gap=max(4.0, min_gap_px),
    )
    ordered: List["Block"] = []
    for band_idx, band in enumerate(bands):
        for blk in band:
            _mark(blk, coarse_band=band_idx)
        ordered.extend(
            _recursive_adaptive_sort(
                band,
                image_width=image_width,
                image_height=image_height,
                cross_ids=cross_ids,
                density_threshold=density_threshold,
                min_gap_px=min_gap_px,
            )
        )
    return ordered


# ---------------------------------------------------------------------------
# Phase 4: semantic + geometry restoration
# ---------------------------------------------------------------------------


def _priority(block: "Block", cross_ids: set[int]) -> int:
    if id(block) in cross_ids:
        return 0
    if block.block_type == BlockType.HEADER:
        return 0
    if block.block_type == BlockType.TITLE:
        return 1
    if block.block_type in _VISUAL_TYPES:
        return 2
    if block.block_type in _CAPTION_TYPES:
        return 3
    if block.block_type in {BlockType.PAGE_NUMBER, BlockType.FOOTER}:
        return 5
    return 4


def _direction(block: "Block") -> str:
    return "horizontal" if _w(block) >= _h(block) else "vertical"


def _projection_score(a: "Block", b: "Block") -> float:
    return max(_projection_overlap_ratio_x(a, b), _projection_overlap_ratio_y(a, b))


def _geometry_distance(
    pending: "Block",
    anchor: "Block",
    *,
    image_width: int,
    image_height: Optional[int],
) -> float:
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max(_y2(pending), _y2(anchor), 1.0))
    scale = max(page_w, page_h, 1.0)

    direction_conflict = _direction(pending) != _direction(anchor)
    low_projection = _projection_score(pending, anchor) < 0.20
    phi1 = 1.0 if direction_conflict and low_projection else 0.0

    phi2 = _edge_gap(pending, anchor) / scale

    if _y1(pending) >= _y2(anchor):
        vertical_gap = _y1(pending) - _y2(anchor)
    elif _y1(anchor) >= _y2(pending):
        vertical_gap = _y1(anchor) - _y2(pending)
    else:
        vertical_gap = abs(_cy(pending) - _cy(anchor)) * 0.20
    phi3 = vertical_gap / scale

    phi4 = abs(_cx(pending) - _cx(anchor)) / scale

    if pending.block_type == BlockType.TITLE:
        weights = (90.0, 8.0, 4.0, 0.8)
    elif pending.block_type in _VISUAL_TYPES:
        weights = (80.0, 10.0, 1.2, 0.8)
    elif pending.block_type in _CAPTION_TYPES:
        weights = (80.0, 12.0, 2.5, 0.6)
    else:
        weights = (90.0, 9.0, 2.0, 1.0)

    w1, w2, w3, w4 = weights
    return w1 * phi1 + w2 * phi2 + w3 * phi3 + w4 * phi4


def _nearest_anchor(
    pending: "Block",
    ranked: Sequence[tuple[float, "Block"]],
    *,
    image_width: int,
    image_height: Optional[int],
) -> tuple[float, "Block", float] | None:
    if not ranked:
        return None
    best: tuple[float, "Block", float] | None = None
    for rank, anchor in ranked:
        dist = _geometry_distance(
            pending,
            anchor,
            image_width=image_width,
            image_height=image_height,
        )
        if best is None or dist < best[2]:
            best = (rank, anchor, dist)
    return best


def _rank_by_vertical_band(
    pending: "Block",
    ranked: Sequence[tuple[float, "Block"]],
    *,
    prefer_before: bool,
) -> Optional[float]:
    # Restrict to same-column anchors so masked elements anchor within their
    # own column instead of leaking into adjacent columns.
    same_col = [(rank, blk) for rank, blk in ranked
                if _projection_overlap_ratio_x(pending, blk) >= 0.15]
    target = same_col if same_col else list(ranked)

    above = [(rank, blk) for rank, blk in target if _y2(blk) <= _y1(pending)]
    below = [(rank, blk) for rank, blk in target if _y1(blk) >= _y2(pending)]
    if prefer_before:
        if below:
            _, anchor = below[0]
            _mark(pending, restore_anchor_id=_block_id(anchor))
            return min(rank for rank, _ in below) - 0.45
        if target:
            return min(rank for rank, _ in target) - 0.45
    else:
        if above and below:
            above_max = max(rank for rank, _ in above)
            below_min = min(rank for rank, _ in below)
            return (above_max + below_min) * 0.5
        if above:
            return max(rank for rank, _ in above) + 0.45
        if below:
            return min(rank for rank, _ in below) - 0.20
    return None


def _choose_restore_rank(
    pending: "Block",
    ranked: Sequence[tuple[float, "Block"]],
    *,
    cross_ids: set[int],
    image_width: int,
    image_height: Optional[int],
    barrier_width_ratio: float,
) -> float:
    if not ranked:
        return float(_y1(pending) * 10000.0 + _x1(pending))

    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for _, b in ranked), default=1.0))
    wide = _w(pending) >= page_w * barrier_width_ratio or id(pending) in cross_ids

    if pending.block_type in {BlockType.HEADER, BlockType.TITLE}:
        # A section title within a column (has text above it) should sit
        # between the above and below content.  A page-top / spanning title
        # should sit before everything that follows.
        has_text_above = not wide and any(
            _projection_overlap_ratio_x(pending, blk) >= 0.15 and _y2(blk) <= _y1(pending)
            for _, blk in ranked
        )
        prefer_before = not has_text_above
        rank = _rank_by_vertical_band(pending, ranked, prefer_before=prefer_before)
        if rank is not None:
            return rank

    # Wide text-like blocks: place before content if in upper half (header-like),
    # after content if in lower half (footer-like).
    if wide and pending.block_type in _TEXTLIKE_TYPES:
        if _cy(pending) <= page_h * 0.45:
            rank = _rank_by_vertical_band(pending, ranked, prefer_before=True)
        else:
            rank = _rank_by_vertical_band(pending, ranked, prefer_before=False)
        if rank is not None:
            return rank

    if pending.block_type in {BlockType.FOOTER, BlockType.PAGE_NUMBER}:
        rank = _rank_by_vertical_band(pending, ranked, prefer_before=False)
        if rank is not None:
            return rank

    if pending.block_type in _VISUAL_TYPES or (wide and pending.block_type not in {BlockType.TITLE, BlockType.HEADER}):
        rank = _rank_by_vertical_band(pending, ranked, prefer_before=False)
        if rank is not None:
            return rank

    if pending.block_type in _CAPTION_TYPES:
        # Caption above a visual (e.g. table caption): place just before the visual.
        visual_below = [
            (rank, blk) for rank, blk in ranked
            if blk.block_type in _VISUAL_TYPES
            and _y1(blk) >= _y2(pending) - max(_h(pending), 20.0) * 0.8
            and _projection_overlap_ratio_x(pending, blk) >= 0.12
        ]
        if visual_below:
            return min(rank for rank, _ in visual_below) - 0.10
        # Caption below a visual (e.g. figure caption): place just after the visual.
        visual_above = [
            (rank, blk) for rank, blk in ranked
            if blk.block_type in _VISUAL_TYPES
            and _y2(blk) <= _y1(pending) + max(_h(pending), 20.0) * 0.5
            and _projection_overlap_ratio_x(pending, blk) >= 0.12
        ]
        if visual_above:
            return max(rank for rank, _ in visual_above) + 0.10

    nearest = _nearest_anchor(
        pending,
        ranked,
        image_width=image_width,
        image_height=image_height,
    )
    if nearest is None:
        return 0.0
    anchor_rank, anchor, dist = nearest
    _mark(pending, restore_anchor_id=_block_id(anchor), restore_distance=round(dist, 5))
    if _y2(pending) <= _y1(anchor):
        return anchor_rank - 0.20
    if _y1(pending) >= _y2(anchor):
        return anchor_rank + 0.20
    if _x1(pending) < _x1(anchor):
        return anchor_rank - 0.05
    return anchor_rank + 0.05


def _restore_masked_elements(
    anchors: Sequence["Block"],
    masked: Sequence["Block"],
    *,
    cross_ids: set[int],
    image_width: int,
    image_height: Optional[int],
    barrier_width_ratio: float,
) -> List["Block"]:
    ranked: List[tuple[float, "Block"]] = [(float(i), blk) for i, blk in enumerate(anchors, start=1)]

    if not ranked:
        ordered = _sort_yx(masked)
        for i, blk in enumerate(ordered):
            _mark(blk, final_order=i, phase="fallback_all_masked")
        return ordered

    for seq, blk in enumerate(sorted(masked, key=lambda b: (_priority(b, cross_ids), _y1(b), _x1(b)))):
        rank = _choose_restore_rank(
            blk,
            ranked,
            cross_ids=cross_ids,
            image_width=image_width,
            image_height=image_height,
            barrier_width_ratio=barrier_width_ratio,
        )
        rank += 0.001 * (_priority(blk, cross_ids) + seq / 1000.0)
        _mark(blk, restore_rank=round(rank, 6), restore_priority=_priority(blk, cross_ids))
        ranked.append((rank, blk))
        ranked.sort(key=lambda item: (item[0], _y1(item[1]), _x1(item[1])))

    ordered = [blk for _, blk in ranked]
    return ordered


# ---------------------------------------------------------------------------
# Column metadata assignment
# ---------------------------------------------------------------------------


def _assign_single_column(blocks: Sequence["Block"]) -> None:
    for blk in blocks:
        blk.col_count = 1
        blk.col_index = 0
        blk.spanned_cols = [0]


def _visual_column_fallback_source(
    ordered: Sequence["Block"],
    text_candidates: Sequence["Block"],
    *,
    page_w: float,
    page_h: float,
) -> List["Block"]:
    """Build a conservative column source for pages with a visual side column.

    Some textbook pages use a main text stream plus a side column made mostly of
    figures, captions, and short local headings. Body-text-only column detection
    sees only the main stream and collapses the page to one column. This fallback
    adds stable visual anchors while keeping wide text out of the column skeleton.
    """
    text_anchors = [
        block for block in text_candidates
        if block.block_type in _COLUMN_ANCHOR_TYPES
        and _w(block) <= page_w * 0.54
        and _h(block) >= max(18.0, page_h * 0.012)
    ]
    visual_anchors = [
        block for block in ordered
        if block.block_type in _VISUAL_TYPES
        and page_w * 0.10 <= _w(block) <= page_w * 0.48
        and _h(block) >= max(60.0, page_h * 0.045)
    ]
    if len(text_anchors) < 2 or not visual_anchors:
        return []

    text_center = median([_cx(block) for block in text_anchors])
    visual_center = median([_cx(block) for block in visual_anchors])
    if abs(visual_center - text_center) < page_w * 0.22:
        return []

    if visual_center > text_center:
        side_text = [block for block in text_anchors if _cx(block) <= visual_center - page_w * 0.10]
    else:
        side_text = [block for block in text_anchors if _cx(block) >= visual_center + page_w * 0.10]
    if len(side_text) < 2:
        return []

    side_text = sorted(side_text, key=lambda block: (-_area(block), _y1(block), _x1(block)))[:2]

    captions = [
        block for block in ordered
        if block.block_type in _CAPTION_TYPES
        and _w(block) <= page_w * 0.30
        and any(
            abs(_cx(block) - _cx(visual)) <= max(page_w * 0.13, _w(visual) * 0.70)
            and -max(24.0, page_h * 0.015) <= _y1(block) - _y2(visual) <= max(130.0, page_h * 0.08)
            for visual in visual_anchors
        )
    ]
    return side_text + visual_anchors + captions


def _visual_column_fallback_quality(
    columns: Sequence[Sequence["Block"]],
    *,
    page_w: float,
) -> bool:
    if len(columns) < 2:
        return False
    text_cols = 0
    visual_cols = 0
    centers: List[float] = []
    for column in columns:
        if not column:
            continue
        centers.append(median([_cx(block) for block in column]))
        if any(block.block_type in _COLUMN_ANCHOR_TYPES for block in column):
            text_cols += 1
        if any(block.block_type in _VISUAL_TYPES for block in column):
            visual_cols += 1
    if text_cols < 1 or visual_cols < 1 or len(centers) < 2:
        return False
    centers = sorted(centers)
    return max(b - a for a, b in zip(centers, centers[1:])) >= page_w * 0.22


def _assign_near_visual_titles_to_visual_columns(
    blocks: Sequence["Block"],
    *,
    page_w: float,
    page_h: float,
) -> None:
    visuals = [
        block for block in blocks
        if block.block_type in _VISUAL_TYPES
        and len(getattr(block, "spanned_cols", []) or []) == 1
    ]
    if not visuals:
        return

    for title in blocks:
        if title.block_type != BlockType.TITLE:
            continue
        title_text = re.sub(r"\s+", "", _block_text(title))
        if not title_text or len(title_text) > 12:
            continue
        if _w(title) > page_w * 0.22:
            continue

        best: tuple[float, "Block"] | None = None
        for visual in visuals:
            vertical_gap = _y1(visual) - _y2(title)
            if not (0.0 <= vertical_gap <= max(150.0, page_h * 0.09)):
                continue
            center_gap = abs(_cx(title) - _cx(visual))
            if center_gap > max(page_w * 0.26, _w(visual) * 0.85):
                continue
            score = vertical_gap + center_gap * 0.35
            if best is None or score < best[0]:
                best = (score, visual)

        if best is None:
            continue
        visual = best[1]
        title.spanned_cols = list(getattr(visual, "spanned_cols", []) or [getattr(visual, "col_index", 0)])
        title.col_index = int(title.spanned_cols[0])


def _sync_local_title_visual_columns(blocks: Sequence["Block"]) -> None:
    by_id = {_block_id(block): block for block in blocks if _block_id(block)}
    for block in blocks:
        if block.block_type != BlockType.TITLE:
            continue
        anchor_id = _marked(block, "local_title_visual_anchor_id")
        if not anchor_id:
            continue
        visual = by_id.get(str(anchor_id))
        if visual is None:
            continue
        visual_cols = list(getattr(visual, "spanned_cols", []) or [])
        if len(visual_cols) != 1:
            continue
        block.spanned_cols = visual_cols
        block.col_index = int(visual_cols[0])


def _assign_centered_section_starts_to_spanned_columns(
    blocks: Sequence["Block"],
    *,
    page_w: float,
    page_h: float,
    col_count: int,
) -> None:
    if col_count < 2:
        return
    has_spanned_text = any(
        block.block_type in _TEXTLIKE_TYPES
        and block.block_type != BlockType.TITLE
        and block.block_type not in _STRIP_TYPES
        and len(getattr(block, "spanned_cols", []) or []) > 1
        for block in blocks
    )
    if not has_spanned_text:
        return

    visuals = [
        block for block in blocks
        if block.block_type in _VISUAL_TYPES
        and len(getattr(block, "spanned_cols", []) or []) == 1
    ]
    page_center = page_w * 0.5
    center_tol = max(42.0, page_w * 0.055)
    all_cols = list(range(col_count))

    def _has_nearby_visual_anchor(title: "Block") -> bool:
        for visual in visuals:
            vertical_gap = _y1(visual) - _y2(title)
            if not (0.0 <= vertical_gap <= max(180.0, page_h * 0.12)):
                continue
            if abs(_cx(title) - _cx(visual)) <= max(page_w * 0.26, _w(visual) * 0.85):
                return True
        return False

    for title in blocks:
        if title.block_type != BlockType.TITLE:
            continue
        title_text = re.sub(r"\s+", "", _block_text(title))
        if not title_text or len(title_text) > 12:
            continue
        if _w(title) > page_w * 0.22:
            continue
        if abs(_cx(title) - page_center) > center_tol:
            continue
        if _has_nearby_visual_anchor(title):
            continue

        title.spanned_cols = all_cols
        title.col_index = 0
        _mark(title, centered_section_spanned=True)

        followers = [
            block for block in blocks
            if block.block_type in _TEXTLIKE_TYPES
            and block.block_type != BlockType.TITLE
            and block.block_type not in _STRIP_TYPES
            and _y1(block) >= _y2(title)
            and _y1(block) - _y2(title) <= max(70.0, page_h * 0.05)
            and _w(block) <= page_w * 0.50
            and _x1(block) <= page_w * 0.28
        ]
        if not followers:
            continue
        follower = min(followers, key=lambda block: (_y1(block), _x1(block)))
        follower.spanned_cols = all_cols
        follower.col_index = 0
        _mark(follower, centered_section_follower=True, centered_section_title_id=_block_id(title))


def _assign_column_metadata(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
    max_cols: int,
    cluster_thresh: float,
) -> None:
    ordered = list(ordered)
    if not ordered:
        return

    page_w = max(float(image_width), 1.0)
    # Exclude titles, visuals, captions, strips, and top-attachments from column
    # detection — they are structural elements, not column body text.
    candidates = [
        blk for blk in ordered
        if blk.block_type not in _COLUMN_EXCLUDED_TYPES
        and blk.block_type != BlockType.TITLE
        and _marked(blk, "mask_reason") != "top_attachment"
        and _w(blk) <= page_w * 0.60
    ]
    if len(candidates) < 2:
        candidates = [
            blk for blk in ordered
            if blk.block_type not in _COLUMN_EXCLUDED_TYPES
            and blk.block_type != BlockType.TITLE
            and _marked(blk, "mask_reason") != "top_attachment"
        ]
    if len(candidates) < 2:
        candidates = [blk for blk in ordered if blk.block_type not in _VISUAL_TYPES]
    if len(candidates) < 2:
        _assign_single_column(ordered)
        return

    columns, col_bounds = detect_columns(
        candidates,
        image_width,
        max_cols=max_cols,
        cluster_thresh=cluster_thresh,
    )
    if len(columns) <= 1:
        fallback_source = _visual_column_fallback_source(
            ordered,
            candidates,
            page_w=page_w,
            page_h=max(float(image_height or 0), max((_y2(block) for block in ordered), default=1.0)),
        )
        if fallback_source:
            fallback_columns, fallback_bounds = detect_columns(
                fallback_source,
                image_width,
                max_cols=max_cols,
                cluster_thresh=min(cluster_thresh, 0.08),
            )
            if _visual_column_fallback_quality(fallback_columns, page_w=page_w):
                columns, col_bounds = fallback_columns, fallback_bounds
    if len(columns) <= 1:
        _assign_single_column(ordered)
        return

    for col_idx, members in enumerate(columns):
        for blk in members:
            blk.col_count = len(columns)
            blk.col_index = col_idx
            blk.spanned_cols = [col_idx]

    unassigned = [blk for blk in ordered if int(getattr(blk, "col_count", 0) or 0) != len(columns)]
    if unassigned:
        detect_spanned_blocks(unassigned, col_bounds)
        for blk in unassigned:
            blk.col_count = len(columns)
            if _marked(blk, "mask_reason") == "top_attachment":
                blk.col_count = 1
                blk.col_index = 0
                blk.spanned_cols = [0]

    _assign_near_visual_titles_to_visual_columns(
        ordered,
        page_w=page_w,
        page_h=max(float(image_height or 0), max((_y2(block) for block in ordered), default=1.0)),
    )
    _sync_local_title_visual_columns(ordered)
    _assign_centered_section_starts_to_spanned_columns(
        ordered,
        page_w=page_w,
        page_h=max(float(image_height or 0), max((_y2(block) for block in ordered), default=1.0)),
        col_count=len(columns),
    )

    for blk in ordered:
        if _marked(blk, "mask_reason") != "top_attachment":
            blk.col_count = len(columns)
_INLINE_EQUATION_LABEL_RE = re.compile(r"^\(\s*\d+[a-zA-Z]?\s*\)$")
_NUMBERED_SECTION_RE = re.compile(r"^\s*\d+(?:\.\d+)*\b")


@dataclass(frozen=True)
class _LocalParallelRegion:
    region_id: str
    region_kind: str
    blocks: tuple["Block", ...]
    columns: tuple[tuple["Block", ...], ...]
    bounds: tuple[tuple[float, float], ...]
    top: float
    bottom: float


@dataclass(frozen=True)
class _SpanningArticleRegion:
    region_id: str
    region_kind: str
    title: "Block"
    subtitle: "Block"
    columns: tuple[tuple["Block", ...], ...]
    visuals: tuple["Block", ...]
    captions: tuple["Block", ...]


@dataclass(frozen=True)
class _RegionPlacement:
    first_idx: int
    last_idx: int
    prefix: tuple["Block", ...]


def _is_inline_equation_label(block: "Block") -> bool:
    if block.block_type not in _CAPTION_TYPES:
        return False
    text = _block_text(block).strip()
    if not text or not _INLINE_EQUATION_LABEL_RE.match(text):
        return False
    return _line_count(block) <= 2 and _w(block) <= 140.0


def _best_inline_equation_anchor(
    label: "Block",
    ordered: Sequence["Block"],
) -> "Block" | None:
    candidates: List[tuple[float, float, float, "Block"]] = []
    for blk in ordered:
        if blk.block_type not in _VISUAL_TYPES:
            continue
        if blk.block_type not in {BlockType.FORMULA, BlockType.EQUATION}:
            continue
        if _x2(blk) > _x1(label) + 40.0:
            continue
        y_overlap = _projection_overlap_ratio_y(label, blk)
        if y_overlap < 0.45:
            continue
        x_gap = max(0.0, _x1(label) - _x2(blk))
        center_dy = abs(_cy(label) - _cy(blk))
        candidates.append((x_gap, center_dy, -y_overlap, blk))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], -_x2(item[3])))
    return candidates[0][3]


def _enforce_inline_equation_label_adjacency(
    ordered: Sequence["Block"],
) -> List["Block"]:
    seq = list(ordered)
    if not seq:
        return seq

    moved = True
    while moved:
        moved = False
        for idx, blk in enumerate(list(seq)):
            if not _is_inline_equation_label(blk):
                continue
            anchor = _best_inline_equation_anchor(blk, seq)
            if anchor is None:
                continue
            anchor_idx = seq.index(anchor)
            curr_idx = seq.index(blk)
            target_idx = anchor_idx + 1
            if curr_idx == target_idx:
                continue
            seq.pop(curr_idx)
            if curr_idx < target_idx:
                target_idx -= 1
            seq.insert(target_idx, blk)
            moved = True
            break
    return seq


def _table_vertical_gap(a: "Block", b: "Block") -> float:
    if _overlap_1d(_y1(a), _y2(a), _y1(b), _y2(b)) > 0:
        return 0.0
    if _y2(a) <= _y1(b):
        return _y1(b) - _y2(a)
    return _y1(a) - _y2(b)


def _best_table_for_caption(
    caption: "Block",
    ordered: Sequence["Block"],
) -> "Block" | None:
    candidates: List[tuple[int, float, float, float, "Block"]] = []
    for blk in ordered:
        if blk.block_type != BlockType.TABLE:
            continue
        x_overlap = _projection_overlap_ratio_x(caption, blk)
        if x_overlap < 0.10 and _w(blk) < _w(caption) * 1.5:
            continue
        below_bias = 0 if _cy(blk) >= _cy(caption) - 4.0 else 1
        vertical_gap = _table_vertical_gap(caption, blk)
        center_dx = abs(_cx(caption) - _cx(blk))
        candidates.append((below_bias, vertical_gap, center_dx, -x_overlap, blk))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], _y1(item[4]), _x1(item[4])))
    return candidates[0][4]


def _best_table_for_footnote(
    footnote: "Block",
    ordered: Sequence["Block"],
) -> "Block" | None:
    candidates: List[tuple[int, float, float, float, "Block"]] = []
    for blk in ordered:
        if blk.block_type != BlockType.TABLE:
            continue
        x_overlap = _projection_overlap_ratio_x(footnote, blk)
        if x_overlap < 0.10 and _w(blk) < _w(footnote) * 1.5:
            continue
        above_bias = 0 if _cy(blk) <= _cy(footnote) + 4.0 else 1
        vertical_gap = _table_vertical_gap(footnote, blk)
        center_dx = abs(_cx(footnote) - _cx(blk))
        candidates.append((above_bias, vertical_gap, center_dx, -x_overlap, blk))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], -_y2(item[4]), _x1(item[4])))
    return candidates[0][4]


def _enforce_table_family_order(
    ordered: Sequence["Block"],
) -> List["Block"]:
    seq = list(ordered)
    if not seq:
        return seq

    captions = [blk for blk in seq if blk.block_type == BlockType.TABLE_CAPTION]
    captions.sort(key=lambda b: (_y1(b), _x1(b), _y2(b), _x2(b)))
    for caption in captions:
        table = _best_table_for_caption(caption, seq)
        if table is None:
            continue
        curr_idx = seq.index(caption)
        table_idx = seq.index(table)
        if curr_idx == table_idx - 1:
            continue
        seq.pop(curr_idx)
        if curr_idx < table_idx:
            table_idx -= 1
        seq.insert(table_idx, caption)
        _mark(caption, table_family_anchor_id=_block_id(table), table_family_role="caption")

    footnotes = [blk for blk in seq if blk.block_type == BlockType.TABLE_FOOTNOTE]
    footnotes.sort(key=lambda b: (_y1(b), _x1(b), _y2(b), _x2(b)))
    for footnote in footnotes:
        table = _best_table_for_footnote(footnote, seq)
        if table is None:
            continue
        curr_idx = seq.index(footnote)
        table_idx = seq.index(table)
        target_idx = table_idx + 1
        while target_idx < len(seq) and seq[target_idx].block_type == BlockType.TABLE_FOOTNOTE:
            target_idx += 1
        if curr_idx == target_idx:
            continue
        seq.pop(curr_idx)
        if curr_idx < target_idx:
            target_idx -= 1
        seq.insert(target_idx, footnote)
        _mark(footnote, table_family_anchor_id=_block_id(table), table_family_role="footnote")

    return seq


def _best_figure_for_caption(
    caption: "Block",
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> "Block" | None:
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in ordered), default=1.0))
    max_gap = max(72.0, page_h * 0.08)
    candidates: List[tuple[int, float, float, float, float, "Block"]] = []
    for figure in ordered:
        if figure.block_type != BlockType.FIGURE:
            continue
        x_overlap = _projection_overlap_ratio_x(caption, figure)
        center_inside = _x1(figure) - page_w * 0.03 <= _cx(caption) <= _x2(figure) + page_w * 0.03
        if x_overlap < 0.18 and not center_inside:
            continue
        if _y1(caption) >= _y2(figure) - 8.0:
            vertical_gap = max(0.0, _y1(caption) - _y2(figure))
            side_bias = 0
        elif _y2(caption) <= _y1(figure) + 8.0:
            vertical_gap = max(0.0, _y1(figure) - _y2(caption))
            side_bias = 1
        else:
            vertical_gap = 0.0
            side_bias = 0 if _cy(caption) >= _cy(figure) else 1
        if vertical_gap > max_gap:
            continue
        center_dx = abs(_cx(caption) - _cx(figure))
        candidates.append((side_bias, vertical_gap, center_dx, -x_overlap, _y1(figure), figure))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], _x1(item[5])))
    return candidates[0][5]


def _enforce_figure_family_order(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    seq = list(ordered)
    captions = [blk for blk in seq if blk.block_type == BlockType.FIGURE_CAPTION]
    if not captions:
        return seq

    captions.sort(key=lambda b: (_y1(b), _x1(b), _y2(b), _x2(b)))
    family_counter = 0
    for caption in captions:
        proto = (getattr(caption, "attributes", None) or {}).get("xycutpp_proto", {})
        if isinstance(proto, dict) and proto.get("figure_group_size"):
            continue
        figure = _best_figure_for_caption(
            caption,
            seq,
            image_width=image_width,
            image_height=image_height,
        )
        if figure is None:
            continue

        family_counter += 1
        family_id = f"figure_family_{family_counter}"
        below_figure = _cy(caption) >= _cy(figure) or _y1(caption) >= _y2(figure) - 8.0
        caption.col_count = int(getattr(figure, "col_count", 1) or 1)
        caption.col_index = int(getattr(figure, "col_index", 0) or 0)
        caption.spanned_cols = list(getattr(figure, "spanned_cols", []) or [caption.col_index])
        curr_idx = seq.index(caption)
        figure_idx = seq.index(figure)
        if below_figure:
            target_idx = figure_idx + 1
            while target_idx < len(seq):
                next_block = seq[target_idx]
                next_proto = (getattr(next_block, "attributes", None) or {}).get("xycutpp_proto", {})
                if (
                    next_block.block_type == BlockType.FIGURE_CAPTION
                    and isinstance(next_proto, dict)
                    and next_proto.get("figure_family_anchor_id") == _block_id(figure)
                ):
                    target_idx += 1
                    continue
                break
        else:
            target_idx = figure_idx

        if curr_idx == target_idx or (below_figure and curr_idx == target_idx - 1):
            _mark(
                figure,
                region_id=family_id,
                region_kind="figure_family",
                region_role="visual",
            )
            _mark(
                caption,
                figure_family_anchor_id=_block_id(figure),
                region_id=family_id,
                region_kind="figure_family",
                region_role="caption",
            )
            continue

        seq.pop(curr_idx)
        if curr_idx < target_idx:
            target_idx -= 1
        seq.insert(target_idx, caption)
        _mark(
            figure,
            region_id=family_id,
            region_kind="figure_family",
            region_role="visual",
        )
        _mark(
            caption,
            figure_family_anchor_id=_block_id(figure),
            region_id=family_id,
            region_kind="figure_family",
            region_role="caption",
        )

    return seq


def _same_row_figure_groups(figures: Sequence["Block"], page_h: float) -> List[List["Block"]]:
    groups: List[List["Block"]] = []
    remaining = sorted(figures, key=lambda b: (_y1(b), _x1(b)))
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        changed = True
        while changed:
            changed = False
            for cand in list(remaining):
                if any(
                    _projection_overlap_ratio_y(cand, member) >= 0.55
                    or abs(_cy(cand) - _cy(member)) <= max(36.0, page_h * 0.035)
                    for member in group
                ):
                    group.append(cand)
                    remaining.remove(cand)
                    changed = True
        group.sort(key=lambda b: (_x1(b), _y1(b)))
        groups.append(group)
    return groups


def _figure_group_captions(
    group: Sequence["Block"],
    ordered: Sequence["Block"],
    *,
    image_height: Optional[int],
) -> List["Block"]:
    page_h = max(float(image_height or 0), max((_y2(b) for b in ordered), default=1.0))
    group_x1 = min(_x1(b) for b in group)
    group_x2 = max(_x2(b) for b in group)
    group_y2 = max(_y2(b) for b in group)
    max_gap = max(48.0, page_h * 0.08)
    captions: List["Block"] = []
    for blk in ordered:
        if blk.block_type != BlockType.FIGURE_CAPTION:
            continue
        if not (group_y2 - 8.0 <= _y1(blk) <= group_y2 + max_gap):
            continue
        overlap = _overlap_1d(_x1(blk), _x2(blk), group_x1, group_x2)
        if overlap <= 0 and not (group_x1 <= _cx(blk) <= group_x2):
            continue
        captions.append(blk)
    captions.sort(key=lambda b: (_y1(b), _x1(b), _y2(b), _x2(b)))
    return captions


def _enforce_parallel_figure_group_order(
    ordered: Sequence["Block"],
    *,
    image_height: Optional[int],
) -> List["Block"]:
    seq = list(ordered)
    figures = [blk for blk in seq if blk.block_type == BlockType.FIGURE]
    if len(figures) < 2:
        return seq
    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))

    for group in _same_row_figure_groups(figures, page_h):
        if len(group) < 2:
            continue
        captions = _figure_group_captions(group, seq, image_height=image_height)
        moving = sorted(group, key=lambda b: (_x1(b), _y1(b))) + captions
        moving_ids = {id(b) for b in moving}
        indices = [seq.index(b) for b in moving if b in seq]
        if len(indices) < len(group):
            continue
        first_idx = min(indices)
        last_idx = max(indices)
        middle = [b for b in seq[first_idx:last_idx + 1] if id(b) not in moving_ids]
        desired = moving + middle
        if seq[first_idx:last_idx + 1] == desired:
            continue
        seq = seq[:first_idx] + desired + seq[last_idx + 1:]
        for order, blk in enumerate(moving):
            _mark(blk, figure_group_order=order, figure_group_size=len(moving))
    return seq


def _is_section_band_leader(
    block: "Block",
    *,
    image_width: int,
    image_height: Optional[int],
) -> bool:
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max(_y2(block), 1.0))
    if block.block_type == BlockType.TITLE:
        text = _block_text(block).strip()
        if text and _NUMBERED_SECTION_RE.match(text):
            return False
        return _h(block) <= page_h * 0.10
    if block.block_type != BlockType.TEXT:
        return False
    text = _block_text(block).strip()
    if not text or _line_count(block) > 2:
        return False
    if _y1(block) <= page_h * 0.12:
        return False
    if _w(block) < page_w * 0.28:
        return False
    if _h(block) > page_h * 0.08:
        return False
    return len(text) <= 220


def _best_section_band_anchor(
    leader: "Block",
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> "Block" | None:
    page_h = max(float(image_height or 0), max(_y2(leader), 1.0))
    max_gap = max(180.0, page_h * 0.12)
    candidates: List[tuple[int, float, float, float, float, float, "Block"]] = []
    for blk in ordered:
        if blk is leader:
            continue
        if blk.block_type in _CAPTION_TYPES or blk.block_type in _VISUAL_TYPES or blk.block_type in _STRIP_TYPES:
            continue
        if _y1(blk) < _y2(leader) - 4.0:
            continue
        x_overlap = _projection_overlap_ratio_x(leader, blk)
        if x_overlap < 0.12:
            continue
        vertical_gap = max(0.0, _y1(blk) - _y2(leader))
        if vertical_gap > max_gap:
            continue
        center_dx = abs(_cx(leader) - _cx(blk))
        gap_bucket = int(vertical_gap // 24.0)
        candidates.append((gap_bucket, -x_overlap, vertical_gap, center_dx, _y1(blk), _x1(blk), blk))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5]))
    return candidates[0][6]


def _enforce_section_band_leader_order(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    seq = list(ordered)
    if not seq:
        return seq

    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))

    moved = True
    while moved:
        moved = False
        leaders = [
            blk for blk in seq
            if _is_section_band_leader(blk, image_width=image_width, image_height=image_height)
        ]
        leaders.sort(key=lambda b: (_y1(b), _x1(b)), reverse=True)
        for leader in leaders:
            cross_column_overhang = 0
            for blk in seq:
                if blk is leader or blk.block_type != BlockType.TEXT:
                    continue
                if _y1(blk) >= _y1(leader):
                    continue
                if _y2(blk) <= _y1(leader) + page_h * 0.02:
                    continue
                if _projection_overlap_ratio_x(leader, blk) >= 0.10:
                    continue
                cross_column_overhang += 1
                if cross_column_overhang >= 1:
                    break
            if cross_column_overhang >= 1:
                continue
            upper_context = 0
            for blk in seq:
                if blk is leader:
                    continue
                if blk.block_type in _CAPTION_TYPES or blk.block_type in _VISUAL_TYPES or blk.block_type in _STRIP_TYPES:
                    continue
                if _y2(blk) > _y1(leader) + 4.0:
                    continue
                if _projection_overlap_ratio_x(leader, blk) < 0.18:
                    continue
                if _y1(leader) - _y2(blk) > page_h * 0.25:
                    continue
                upper_context += 1
                if upper_context > 1:
                    break
            if upper_context > 1:
                continue
            anchor = _best_section_band_anchor(
                leader,
                seq,
                image_width=image_width,
                image_height=image_height,
            )
            if anchor is None:
                continue
            curr_idx = seq.index(leader)
            anchor_idx = seq.index(anchor)
            if curr_idx < anchor_idx:
                continue
            seq.pop(curr_idx)
            anchor_idx = seq.index(anchor)
            seq.insert(anchor_idx, leader)
            _mark(leader, band_anchor_id=_block_id(anchor), band_anchor_kind="section_leader")
            moved = True
            break
    return seq


def _enforce_peripheral_sidebar_demote(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    seq = list(ordered)
    if len(seq) < 4:
        return seq
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))
    def _is_edge_text_block(block: "Block", side: str) -> bool:
        if block.block_type != BlockType.TEXT:
            return False
        if side == "left":
            return _x2(block) <= page_w * 0.30 or (_x1(block) <= page_w * 0.12 and _x2(block) <= page_w * 0.42)
        return _x1(block) >= page_w * 0.70 or (_x2(block) >= page_w * 0.88 and _x1(block) >= page_w * 0.58)

    wide_titles = [
        b for b in seq
        if b.block_type == BlockType.TITLE and _w(b) >= page_w * 0.22
    ]
    if not wide_titles:
        return seq

    for side in ("left", "right"):
        edge_blocks = [
            b for b in seq
            if _is_edge_text_block(b, side)
        ]
        if len(edge_blocks) < 2:
            continue
        y_span = max(_y2(b) for b in edge_blocks) - min(_y1(b) for b in edge_blocks)
        if y_span < page_h * 0.10:
            continue
        edge_ids = {id(b) for b in edge_blocks}
        edge_inner = max(_x2(b) for b in edge_blocks) if side == "left" else min(_x1(b) for b in edge_blocks)
        edge_top = min(_y1(b) for b in edge_blocks)
        main_title = None
        for title in wide_titles:
            if side == "left" and _x1(title) <= edge_inner:
                continue
            if side == "right" and _x2(title) >= edge_inner:
                continue
            if side == "left" and _x1(title) >= edge_inner + page_w * 0.04:
                main_title = title
                break
            if side == "right" and _x2(title) <= edge_inner - page_w * 0.04:
                main_title = title
                break
        if main_title is None:
            continue
        main_blocks = [
            b for b in seq
            if id(b) not in edge_ids
            and b.block_type in {BlockType.TEXT, BlockType.TITLE}
            and _y1(b) >= _y1(main_title) - 8.0
        ]
        if len(main_blocks) < 2:
            continue
        first_edge = min(seq.index(b) for b in edge_blocks)
        sidebar_bottom = max(_y2(b) for b in edge_blocks)
        early_main = [
            b for b in main_blocks
            if _y1(b) < sidebar_bottom + page_h * 0.04
        ]
        if not early_main:
            continue
        last_main = max(seq.index(b) for b in early_main)
        if first_edge > last_main:
            continue
        remain = [b for b in seq if id(b) not in edge_ids]
        insert_pos = 0
        last_main_block = seq[last_main]
        for i, blk in enumerate(remain):
            if blk is last_main_block:
                insert_pos = i + 1
                break
        for blk in edge_blocks:
            _mark(blk, peripheral_sidebar_demoted=True)
        seq = remain[:insert_pos] + edge_blocks + remain[insert_pos:]
    return seq


def _enforce_column_major_on_parallel_table_figures(
    ordered: Sequence["Block"],
) -> List["Block"]:
    seq = list(ordered)
    if len(seq) < 6:
        return seq

    table_caption_cols = {
        int(getattr(blk, "col_index", 0))
        for blk in seq
        if blk.block_type == BlockType.TABLE_CAPTION
        and len(getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]) == 1
    }
    if len(table_caption_cols) < 2:
        return seq

    figures = [
        blk for blk in seq
        if blk.block_type == BlockType.FIGURE
        and int(getattr(blk, "col_index", 0)) in table_caption_cols
    ]
    if len(figures) < 2:
        return seq
    by_col: dict[int, List["Block"]] = {}
    for blk in seq:
        cols = getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]
        if len(cols) == 1:
            by_col.setdefault(int(cols[0]), []).append(blk)
    if len(by_col) < 2:
        return seq

    col_major: List["Block"] = []
    used: set[int] = set()
    for col_idx in sorted(by_col.keys()):
        members = sorted(by_col[col_idx], key=lambda b: (_y1(b), _x1(b)))
        col_major.extend(members)
        used.update(id(b) for b in members)
    remainder = [blk for blk in seq if id(blk) not in used]
    if not remainder:
        return col_major

    ordered: List["Block"] = []
    inserted_remainder = False
    for blk in col_major:
        if not inserted_remainder and remainder and _y1(blk) > min(_y1(r) for r in remainder):
            ordered.extend(sorted(remainder, key=lambda b: (_y1(b), _x1(b))))
            inserted_remainder = True
        ordered.append(blk)
    if not inserted_remainder:
        ordered.extend(sorted(remainder, key=lambda b: (_y1(b), _x1(b))))
    return ordered


def _sort_same_column_text_runs_by_geometry(
    ordered: Sequence["Block"],
) -> List["Block"]:
    seq = list(ordered)
    if len(seq) < 3:
        return seq

    out: List["Block"] = []
    idx = 0
    while idx < len(seq):
        block = seq[idx]
        cols = getattr(block, "spanned_cols", []) or [getattr(block, "col_index", 0)]
        col_count = int(getattr(block, "col_count", 1) or 1)
        block_proto = (getattr(block, "attributes", None) or {}).get("xycutpp_proto", {})
        if block.block_type not in _TEXTLIKE_TYPES or len(cols) != 1 or col_count <= 1:
            out.append(block)
            idx += 1
            continue
        if block_proto.get("peripheral_sidebar_demoted"):
            out.append(block)
            idx += 1
            continue

        run = [block]
        idx += 1
        while idx < len(seq):
            cand = seq[idx]
            cand_cols = getattr(cand, "spanned_cols", []) or [getattr(cand, "col_index", 0)]
            cand_proto = (getattr(cand, "attributes", None) or {}).get("xycutpp_proto", {})
            if cand.block_type not in _TEXTLIKE_TYPES or len(cand_cols) != 1:
                break
            if cand_proto.get("peripheral_sidebar_demoted"):
                break
            if int(cand_cols[0]) != int(cols[0]) or getattr(cand, "col_count", 1) != getattr(block, "col_count", 1):
                break
            run.append(cand)
            idx += 1

        if len(run) >= 3:
            y_ordered = sorted(run, key=lambda b: (_y1(b), _x1(b)))
            inversions = sum(1 for a, b in zip(run, run[1:]) if _y1(a) > _y1(b) + 12.0)
            if inversions:
                out.extend(y_ordered)
                continue
        out.extend(run)
    return out


def _promote_upper_visual_family_before_lower_band(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    seq = list(ordered)
    if len(seq) < 5:
        return seq

    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))
    for figure in list(seq):
        if figure.block_type not in _VISUAL_TYPES or _w(figure) < page_w * 0.42:
            continue
        if _y1(figure) > page_h * 0.32:
            continue

        figure_idx = seq.index(figure)
        family = [figure]
        next_idx = figure_idx + 1
        while next_idx < len(seq):
            cand = seq[next_idx]
            proto = (getattr(cand, "attributes", None) or {}).get("xycutpp_proto", {})
            if (
                cand.block_type == BlockType.FIGURE_CAPTION
                and isinstance(proto, dict)
                and proto.get("figure_family_anchor_id") == _block_id(figure)
            ):
                family.append(cand)
                next_idx += 1
                continue
            break
        family_ids = {id(block) for block in family}
        family_bottom = max(_y2(block) for block in family)
        blockers = [
            block for block in seq
            if id(block) not in family_ids
            and block.block_type in _TEXTLIKE_TYPES
            and block.block_type not in _STRIP_TYPES
            and _y1(block) >= _y1(figure) + max(72.0, page_h * 0.05)
            and _y1(block) <= family_bottom + max(48.0, page_h * 0.04)
            and _projection_overlap_ratio_x(block, figure) >= 0.05
        ]
        if not blockers:
            continue

        first_blocker = min(blockers, key=lambda block: seq.index(block))
        blocker_idx = seq.index(first_blocker)
        family_first_idx = min(seq.index(block) for block in family)
        if family_first_idx <= blocker_idx:
            continue

        moving = [block for block in seq if id(block) in family_ids]
        remain = [block for block in seq if id(block) not in family_ids]
        insert_pos = remain.index(first_blocker)
        seq = remain[:insert_pos] + moving + remain[insert_pos:]
        _mark(figure, upper_visual_band_promoted=True, upper_visual_blocker_id=_block_id(first_blocker))
    return seq


def _reorder_side_visual_families_by_caption_anchor(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    seq = list(ordered)
    if len(seq) < 5:
        return seq

    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(block) for block in seq), default=1.0))
    has_spanned_text = any(
        block.block_type in _TEXTLIKE_TYPES
        and block.block_type != BlockType.TITLE
        and block.block_type not in _STRIP_TYPES
        and len(getattr(block, "spanned_cols", []) or []) > 1
        for block in seq
    )
    if not has_spanned_text:
        return seq

    figures = [
        block for block in seq
        if block.block_type in _VISUAL_TYPES
        and page_w * 0.10 <= _w(block) <= page_w * 0.48
        and len(getattr(block, "spanned_cols", []) or []) == 1
    ]
    for figure in figures:
        captions = [
            block for block in seq
            if block.block_type == BlockType.FIGURE_CAPTION
            and ((getattr(block, "attributes", None) or {}).get("xycutpp_proto", {}) or {}).get("figure_family_anchor_id") == _block_id(figure)
        ]
        if not captions:
            continue

        family = [figure] + captions
        family_ids = {id(block) for block in family}
        anchor_y = max(_y2(block) for block in family)
        family_first_idx = min(seq.index(block) for block in family)

        nearby_titles = [
            block for block in seq
            if id(block) not in family_ids
            and block.block_type == BlockType.TITLE
            and _w(block) <= page_w * 0.22
            and _y2(block) <= _y1(figure) + max(24.0, page_h * 0.015)
            and _y1(figure) - _y2(block) <= max(180.0, page_h * 0.12)
            and abs(_cx(block) - _cx(figure)) <= max(page_w * 0.22, _w(figure) * 0.85)
        ]
        if nearby_titles:
            title = max(nearby_titles, key=lambda block: (_y2(block), _x1(block)))
            family.insert(0, title)
            family_ids.add(id(title))
            title.spanned_cols = list(getattr(figure, "spanned_cols", []) or [getattr(figure, "col_index", 0)])
            title.col_index = int(title.spanned_cols[0])
            _mark(title, local_title_visual_anchor_id=_block_id(figure))

        if any(
            block.block_type in _TEXTLIKE_TYPES
            and len(getattr(block, "spanned_cols", []) or []) > 1
            and _overlap_1d(_y1(block), _y2(block), _y1(caption), _y2(caption))
            >= min(_h(block), _h(caption)) * 0.25
            for block in seq
            for caption in captions
        ):
            continue

        preceding = [
            block for block in seq
            if id(block) not in family_ids
            and block.block_type in _TEXTLIKE_TYPES
            and block.block_type not in _STRIP_TYPES
            and (
                (
                    len(getattr(block, "spanned_cols", []) or []) > 1
                    and _y2(block) <= anchor_y + max(10.0, page_h * 0.008)
                )
                or (
                    len(getattr(block, "spanned_cols", []) or []) <= 1
                    and _y1(block) <= anchor_y + max(10.0, page_h * 0.008)
                )
            )
            and _y1(block) >= _y1(figure) - max(260.0, page_h * 0.18)
            and (
                len(getattr(block, "spanned_cols", []) or []) > 1
                or int(getattr(block, "col_index", 0) or 0) != int(getattr(figure, "col_index", 0) or 0)
            )
        ]
        if not preceding:
            continue

        target_idx = max(seq.index(block) for block in preceding) + 1
        if abs(family_first_idx - target_idx) <= 1 and family_first_idx < target_idx:
            continue

        moving = [block for block in seq if id(block) in family_ids]
        remain = [block for block in seq if id(block) not in family_ids]
        target_block = seq[target_idx - 1] if target_idx > 0 else None
        insert_pos = remain.index(target_block) + 1 if target_block in remain else 0
        seq = remain[:insert_pos] + moving + remain[insert_pos:]
        _mark(figure, side_visual_caption_anchor_order=True, side_visual_anchor_y=round(anchor_y, 3))

    return seq


def _promote_textbook_intro_sidebar(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    seq = list(ordered)
    if len(seq) < 6:
        return seq
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))

    titles = [
        blk for blk in seq
        if blk.block_type == BlockType.TITLE
        and _x1(blk) <= page_w * 0.12
        and _w(blk) >= page_w * 0.38
        and _y1(blk) <= page_h * 0.18
    ]
    if not titles:
        return seq
    intro_title = min(titles, key=lambda b: (_y1(b), _x1(b)))
    visuals = [blk for blk in seq if blk.block_type in _VISUAL_TYPES]
    if not any(_y1(v) <= _y1(intro_title) + page_h * 0.02 and _x1(v) >= page_w * 0.45 for v in visuals):
        return seq
    if not any(_y1(v) >= page_h * 0.55 and _x1(v) <= page_w * 0.25 for v in visuals):
        return seq

    sidebar = [
        blk for blk in seq
        if blk.block_type == BlockType.TEXT
        and _x1(blk) <= page_w * 0.12
        and _x2(blk) <= page_w * 0.32
        and _y1(blk) > _y2(intro_title)
    ]
    if len(sidebar) < 2:
        return seq

    moving = [intro_title] + sorted(sidebar, key=lambda b: (_y1(b), _x1(b)))
    moving_ids = {id(b) for b in moving}
    remain = [blk for blk in seq if id(blk) not in moving_ids]
    insert_pos = min((seq.index(blk) for blk in moving), default=0)
    for blk in moving:
        _mark(blk, textbook_intro_sidebar=True)
    return remain[:insert_pos] + moving + remain[insert_pos:]


def _is_column_local_visual_block(
    block: "Block",
    seq: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> bool:
    if block.block_type != BlockType.FIGURE:
        return False
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))
    if _w(block) >= page_w * 0.42:
        return False
    if _w(block) < page_w * 0.12 or _h(block) < page_h * 0.06:
        return False
    return _has_column_local_text_neighbor(
        block,
        seq,
        image_width=image_width,
        image_height=image_height,
    )


def _enforce_column_local_visual_neighbors(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    seq = list(ordered)
    if len(seq) < 4:
        return seq

    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))
    max_gap = max(72.0, page_h * 0.08)

    for visual in list(seq):
        dbg = getattr(visual, "attributes", None) or {}
        proto = dbg.get("xycutpp_proto", {}) if isinstance(dbg, dict) else {}
        if proto.get("figure_group_size"):
            continue
        if not _is_column_local_visual_block(
            visual,
            seq,
            image_width=image_width,
            image_height=image_height,
        ):
            continue

        above = [
            blk for blk in seq
            if blk is not visual
            and blk.block_type in _TEXTLIKE_TYPES
            and blk.block_type not in _STRIP_TYPES
            and _y2(blk) <= _y1(visual) + 4.0
            and _projection_overlap_ratio_x(visual, blk) >= 0.22
            and _y1(visual) - _y2(blk) <= max_gap
        ]
        below = [
            blk for blk in seq
            if blk is not visual
            and blk.block_type in _TEXTLIKE_TYPES
            and blk.block_type not in _STRIP_TYPES
            and _y1(blk) >= _y2(visual) - 4.0
            and _projection_overlap_ratio_x(visual, blk) >= 0.22
            and _y1(blk) - _y2(visual) <= max_gap
        ]
        if not above and not below:
            continue

        anchor = None
        insert_after = True
        if above:
            anchor = max(above, key=lambda b: (_y2(b), _x1(b)))
            insert_after = True
        elif below:
            anchor = min(below, key=lambda b: (_y1(b), _x1(b)))
            insert_after = False
        if anchor is None:
            continue

        desired_idx = seq.index(anchor) + (1 if insert_after else 0)
        curr_idx = seq.index(visual)
        if desired_idx > curr_idx:
            desired_idx -= 1
        if curr_idx == desired_idx:
            continue

        seq.pop(curr_idx)
        seq.insert(desired_idx, visual)
        _mark(visual, local_visual_anchor_id=_block_id(anchor))

    return seq


def _enforce_local_title_before_side_visual(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    seq = list(ordered)
    if len(seq) < 4:
        return seq

    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))
    local_titles = [
        blk for blk in seq
        if blk.block_type == BlockType.TITLE
        and _w(blk) <= page_w * 0.22
        and abs(_cx(blk) - page_w * 0.5) <= page_w * 0.14
    ]
    if not local_titles:
        return seq

    for title in local_titles:
        title_idx = seq.index(title)
        candidates = [
            blk for blk in seq
            if blk.block_type in _VISUAL_TYPES
            and _x1(blk) >= page_w * 0.54
            and _w(blk) <= page_w * 0.38
            and _y1(blk) >= _y2(title) - 12.0
            and _y1(blk) <= _y2(title) + max(96.0, page_h * 0.06)
        ]
        if not candidates:
            continue
        visual = min(candidates, key=lambda b: (_y1(b), _x1(b)))
        visual_idx = seq.index(visual)
        if title_idx < visual_idx:
            continue
        seq.pop(title_idx)
        visual_idx = seq.index(visual)
        seq.insert(visual_idx, title)
        _mark(title, local_title_visual_anchor_id=_block_id(visual))

    return seq


def _defer_side_figure_family_until_body_continuation(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    seq = list(ordered)
    if len(seq) < 5:
        return seq

    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))
    below_slack = max(120.0, page_h * 0.08)
    overlap_slack = max(28.0, page_h * 0.025)

    for figure in list(seq):
        if figure.block_type not in _VISUAL_TYPES:
            continue
        if _x1(figure) < page_w * 0.54 or _w(figure) < page_w * 0.14 or _w(figure) > page_w * 0.38:
            continue

        figure_idx = seq.index(figure)
        family = [figure]
        next_idx = figure_idx + 1
        while next_idx < len(seq):
            cand = seq[next_idx]
            proto = (getattr(cand, "attributes", None) or {}).get("xycutpp_proto", {})
            if (
                cand.block_type == BlockType.FIGURE_CAPTION
                and isinstance(proto, dict)
                and proto.get("figure_family_anchor_id") == _block_id(figure)
            ):
                family.append(cand)
                next_idx += 1
                continue
            break
        family_ids = {id(block) for block in family}

        next_title_y = min(
            (
                _y1(block)
                for block in seq
                if id(block) not in family_ids
                and block.block_type == BlockType.TITLE
                and _y1(block) > _y1(figure) + 8.0
            ),
            default=page_h + 1.0,
        )
        continuation = [
            blk for blk in seq
            if id(blk) not in family_ids
            and blk.block_type in _TEXTLIKE_TYPES
            and blk.block_type not in _STRIP_TYPES
            and _x1(blk) <= page_w * 0.24
            and _w(blk) >= page_w * 0.60
            and _projection_overlap_ratio_x(figure, blk) >= 0.22
            and _y1(blk) <= min(next_title_y, _y2(figure) + below_slack)
            and _y2(blk) >= _y2(figure) - overlap_slack
        ]
        if not continuation:
            continue

        last_body = max(continuation, key=lambda blk: (_y2(blk), _x2(blk), seq.index(blk)))
        last_body_idx = seq.index(last_body)
        family_last_idx = max(seq.index(block) for block in family)
        if last_body_idx <= family_last_idx:
            continue

        moving = [block for block in seq if id(block) in family_ids]
        seq = [block for block in seq if id(block) not in family_ids]
        insert_pos = seq.index(last_body) + 1
        seq[insert_pos:insert_pos] = moving
        _mark(figure, body_continuation_anchor_id=_block_id(last_body), body_continuation_deferred=True)

    return seq


def _enforce_spanning_visual_after_covered_columns(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    seq = list(ordered)
    if len(seq) < 6:
        return seq

    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))

    for visual in list(seq):
        if visual.block_type not in _VISUAL_TYPES or _w(visual) < page_w * 0.42:
            continue
        if _y1(visual) < page_h * 0.55 or _h(visual) < page_h * 0.22:
            continue

        covered = [
            blk for blk in seq
            if blk is not visual
            and blk.block_type in _TEXTLIKE_TYPES
            and blk.block_type not in _STRIP_TYPES
            and _projection_overlap_ratio_x(visual, blk) >= 0.18
            and _y2(blk) <= _y1(visual) + 8.0
            and _y1(visual) - _y2(blk) <= max(96.0, page_h * 0.12)
        ]

        spanned_cols = [int(col) for col in (getattr(visual, "spanned_cols", []) or [getattr(visual, "col_index", 0)])]
        first_spanned_col = min(spanned_cols) if spanned_cols else int(getattr(visual, "col_index", 0))
        visual_col_count = int(getattr(visual, "col_count", 1) or 1)
        band_slack = max(48.0, page_h * 0.04)
        preceding_uncovered = [
            blk for blk in seq
            if len(spanned_cols) > 1
            and blk is not visual
            and blk.block_type in _TEXTLIKE_TYPES
            and blk.block_type not in _STRIP_TYPES
            and len(getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]) == 1
            and int((getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)])[0]) < first_spanned_col
            and int(getattr(blk, "col_count", visual_col_count) or visual_col_count) == visual_col_count
            and _y1(blk) >= _y1(visual) - band_slack
            and _y1(blk) <= _y2(visual) + band_slack
            and _y2(blk) > _y1(visual) - band_slack
        ]
        if len(covered) < 2 and not preceding_uncovered:
            continue

        if preceding_uncovered:
            prefix_by_col: dict[int, List["Block"]] = {}
            for block in preceding_uncovered:
                col = int((getattr(block, "spanned_cols", []) or [getattr(block, "col_index", 0)])[0])
                prefix_by_col.setdefault(col, []).append(block)

            moving_prefix_ids = {id(block) for block in preceding_uncovered}
            anchors: dict[int, "Block"] = {}
            for col, members in prefix_by_col.items():
                members.sort(key=lambda b: (_y1(b), _x1(b), _y2(b), _x2(b)))
                first_prefix_y = min(_y1(block) for block in members)
                same_col_predecessors = [
                    block for block in seq
                    if id(block) not in moving_prefix_ids
                    and block is not visual
                    and block.block_type in _TEXTLIKE_TYPES
                    and block.block_type not in _STRIP_TYPES
                    and len(getattr(block, "spanned_cols", []) or [getattr(block, "col_index", 0)]) == 1
                    and int((getattr(block, "spanned_cols", []) or [getattr(block, "col_index", 0)])[0]) == col
                    and _y1(block) <= first_prefix_y
                ]
                if same_col_predecessors:
                    anchors[col] = max(same_col_predecessors, key=lambda b: seq.index(b))

            if anchors:
                anchored_cols = {id(anchor): col for col, anchor in anchors.items()}
                rebuilt: List["Block"] = []
                inserted_cols: set[int] = set()
                for block in seq:
                    if id(block) in moving_prefix_ids:
                        continue
                    rebuilt.append(block)
                    anchored_col = anchored_cols.get(id(block))
                    if anchored_col is not None and anchored_col not in inserted_cols:
                        rebuilt.extend(prefix_by_col[anchored_col])
                        inserted_cols.add(anchored_col)
                for col in sorted(prefix_by_col):
                    if col not in inserted_cols:
                        rebuilt.extend(prefix_by_col[col])
                seq = rebuilt
                for block in preceding_uncovered:
                    _mark(
                        block,
                        spanning_visual_uncovered_prefix=True,
                        spanning_visual_anchor_id=_block_id(visual),
                    )

        visual_idx = seq.index(visual)
        blockers = covered + preceding_uncovered
        last_idx = max(seq.index(blk) for blk in blockers)
        if visual_idx > last_idx:
            _mark(
                visual,
                spanning_visual_after_columns=True,
                spanning_visual_waits_for_uncovered_prefix=bool(preceding_uncovered),
            )
            continue
        seq.pop(visual_idx)
        if visual_idx < last_idx:
            last_idx -= 1
        seq.insert(last_idx + 1, visual)
        _mark(
            visual,
            spanning_visual_after_columns=True,
            spanning_visual_waits_for_uncovered_prefix=bool(preceding_uncovered),
        )
    return seq


def _enforce_lower_section_wraparound_columns(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    """Recover lower newspaper wraparound continuations.

    XY-Cut++ can legitimately split a bottom article into a title/body column
    and a right continuation column.  This rule only fires when the title-body
    pair reaches the page bottom and the candidate right stack is the plausible
    continuation area; otherwise the global column skeleton is left untouched.
    """
    seq = list(ordered)
    if len(seq) < 6:
        return seq
    if _has_locked_global_multicol_skeleton(
        seq,
        image_width=image_width,
        image_height=image_height,
    ):
        return seq

    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))
    academic_cues = sum(
        1 for block in seq
        if block.block_type in {
            BlockType.TABLE,
            BlockType.TABLE_CAPTION,
            BlockType.FORMULA,
            BlockType.EQUATION,
            BlockType.FORMULA_CAPTION,
        }
    )
    if academic_cues >= 3:
        return seq
    if _looks_like_stable_academic_two_column_page(
        seq,
        image_width=image_width,
        image_height=image_height,
    ):
        return seq

    region_counter = 0
    for title in sorted([b for b in seq if b.block_type == BlockType.TITLE], key=lambda b: (_y1(b), _x1(b)), reverse=True):
        if _y1(title) < page_h * 0.58:
            continue
        if _w(title) < page_w * 0.16 or _w(title) > page_w * 0.42:
            continue

        left_body_candidates = [
            block for block in seq
            if block is not title
            and block.block_type in _TEXTLIKE_TYPES
            and block.block_type not in _STRIP_TYPES
            and _y1(block) >= _y2(title) - 8.0
            and _y1(block) - _y2(title) <= max(90.0, page_h * 0.08)
            and _projection_overlap_ratio_x(title, block) >= 0.18
        ]
        if not left_body_candidates:
            continue
        first_body = min(left_body_candidates, key=lambda b: (_y1(b), abs(_cx(b) - _cx(title))))
        if _y2(first_body) < page_h * 0.92:
            continue

        left_ids = {id(title), id(first_body)}
        right_stack = [
            block for block in seq
            if id(block) not in left_ids
            and block.block_type in (_TEXTLIKE_TYPES | {BlockType.TITLE})
            and block.block_type not in _STRIP_TYPES
            and _x1(block) >= max(_x2(title), _x2(first_body)) - page_w * 0.02
            and _x1(block) - _x1(first_body) >= page_w * 0.18
            and _y1(block) >= _y1(title) - max(460.0, page_h * 0.34)
            and _y2(block) <= max(_y2(first_body), _y2(title)) + max(120.0, page_h * 0.10)
        ]
        if len(right_stack) < 2:
            title_idx = seq.index(title)
            body_idx = seq.index(first_body)
            if title_idx > body_idx:
                seq.pop(title_idx)
                body_idx = seq.index(first_body)
                seq.insert(body_idx, title)
            _mark(
                title,
                lower_section_body_anchor_id=_block_id(first_body),
                lower_section_adjacency=True,
            )
            _mark(
                first_body,
                lower_section_title_anchor_id=_block_id(title),
                lower_section_adjacency=True,
            )
            continue

        right_stack = sorted(right_stack, key=lambda b: (_y1(b), _x1(b)))
        if min(_y1(block) for block in right_stack) > _y1(title) + page_h * 0.04:
            continue

        region_counter += 1
        region_id = f"wraparound_section_{region_counter}"
        left_column = [title, first_body]
        moving = left_column + right_stack
        moving_ids = {id(block) for block in moving}
        remain = [block for block in seq if id(block) not in moving_ids]

        preceding_context = [
            block for block in seq
            if id(block) not in moving_ids
            and _y1(block) < _y1(title)
            and (
                block.block_type in (_VISUAL_TYPES | _CAPTION_TYPES)
                or _x1(block) < _x1(title) - page_w * 0.03
                or (
                    block.block_type in _TEXTLIKE_TYPES
                    and _projection_overlap_ratio_x(title, block) >= 0.18
                )
            )
        ]
        if preceding_context:
            anchor_idx = max(seq.index(block) for block in preceding_context)
            anchor = seq[anchor_idx]
            insert_pos = 0
            for idx, block in enumerate(remain):
                if block is anchor:
                    insert_pos = idx + 1
                    break
            else:
                insert_pos = len(remain)
        else:
            anchor_idx = min(seq.index(block) for block in moving)
            insert_pos = 0
            for idx, block in enumerate(remain):
                if seq.index(block) > anchor_idx:
                    insert_pos = idx
                    break
            else:
                insert_pos = len(remain)

        title_idx = seq.index(title)
        body_idx = seq.index(first_body)
        if title_idx > body_idx:
            seq.pop(title_idx)
            body_idx = seq.index(first_body)
            seq.insert(body_idx, title)

        for block in left_column:
            block.col_count = 2
            block.col_index = 0
            block.spanned_cols = [0]
            _mark(
                block,
                region_id=region_id,
                region_kind="wraparound_section",
                region_role="left_column",
                lower_section_adjacency=True,
            )
        _mark(title, lower_section_body_anchor_id=_block_id(first_body))
        _mark(first_body, lower_section_title_anchor_id=_block_id(title))
        for block in right_stack:
            block.col_count = 2
            block.col_index = 1
            block.spanned_cols = [1]
            _mark(
                block,
                region_id=region_id,
                region_kind="wraparound_section",
                region_role="right_continuation",
                wraparound_continues_after_id=_block_id(first_body),
            )

        seq = remain[:insert_pos] + moving + remain[insert_pos:]

    return seq


def _delay_spanning_title_until_prior_overhang_resolves(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    seq = list(ordered)
    if len(seq) < 6:
        return seq
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))

    for title in list(seq):
        if title.block_type != BlockType.TITLE or _w(title) < page_w * 0.30:
            continue
        subtitle = None
        for cand in seq:
            if cand.block_type != BlockType.TEXT:
                continue
            if _w(cand) < page_w * 0.45:
                continue
            if not (_y2(title) - 4.0 <= _y1(cand) <= _y2(title) + page_h * 0.18):
                continue
            subtitle = cand
            break
        if subtitle is None:
            continue
        # Titles with an explicit wide subtitle are handled later by the
        # spanning-article continuity pass. Moving only the title/subtitle here
        # tends to separate them from their own article body and figure group.
        continue
        overhang = [
            blk for blk in seq
            if blk.block_type == BlockType.TEXT
            and _y1(blk) < _y1(title)
            and _y2(blk) > _y1(title) + page_h * 0.02
            and _projection_overlap_ratio_x(title, blk) < 0.10
        ]
        if not overhang:
            continue
        last_overhang = max(seq.index(b) for b in overhang)
        title_idx = seq.index(title)
        subtitle_idx = seq.index(subtitle)
        if title_idx > last_overhang and subtitle_idx > last_overhang:
            continue
        moving = [title, subtitle]
        remain = [b for b in seq if b not in moving]
        insert_pos = 0
        anchor_block = seq[last_overhang]
        for i, blk in enumerate(remain):
            if blk is anchor_block:
                insert_pos = i + 1
                break
        seq = remain[:insert_pos] + moving + remain[insert_pos:]
    return seq


def _find_spanning_article_regions(
    seq: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List[_SpanningArticleRegion]:
    if len(seq) < 6:
        return []
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))
    regions: List[_SpanningArticleRegion] = []
    region_counter = 0

    for blk in list(seq):
        if blk.block_type != BlockType.TITLE or _w(blk) < page_w * 0.30:
            continue
        subtitle = None
        for cand in seq:
            if cand is blk or cand.block_type != BlockType.TEXT:
                continue
            if _w(cand) < page_w * 0.45:
                continue
            if not (_y2(blk) - 4.0 <= _y1(cand) <= _y2(blk) + page_h * 0.18):
                continue
            subtitle = cand
            break
        if subtitle is None:
            continue

        visual_y = min((_y1(b) for b in seq if b.block_type in _VISUAL_TYPES and _y1(b) > _y2(subtitle)), default=page_h)
        band_blocks = [
            b for b in seq
            if b.block_type == BlockType.TEXT
            and b is not subtitle
            and _y1(b) >= _y1(subtitle) - page_h * 0.03
            and _y1(b) < visual_y
        ]
        if len(band_blocks) < 3:
            continue

        cols, _bounds = detect_columns(
            band_blocks,
            int(page_w),
            max_cols=4,
            cluster_thresh=min(COLUMN_CLUSTER_THRESH, 0.08),
        )
        if len(cols) < 2:
            continue

        reordered = [tuple(sorted(col, key=lambda b: (_y1(b), _x1(b)))) for col in cols]
        last_text_y2 = max(_y2(b) for col in reordered for b in col)
        follow_visuals = [
            b for b in seq
            if b.block_type in _VISUAL_TYPES
            and _y1(b) >= last_text_y2 - 4.0
            and _y1(b) <= last_text_y2 + page_h * 0.20
        ]
        visuals: List["Block"] = []
        captions: List["Block"] = []
        if follow_visuals:
            visual = sorted(follow_visuals, key=lambda b: (_y1(b), _x1(b)))[0]
            visuals.append(visual)
            follow_caps = [
                b for b in seq
                if b.block_type in _CAPTION_TYPES
                and _y1(b) >= _y2(visual) - 8.0
                and _y1(b) <= _y2(visual) + page_h * 0.06
            ]
            if follow_caps:
                captions.extend(sorted(follow_caps, key=lambda b: (_y1(b), _x1(b))))

        region_counter += 1
        regions.append(
            _SpanningArticleRegion(
                region_id=f"spanning_article_{region_counter}",
                region_kind="spanning_article_band",
                title=blk,
                subtitle=subtitle,
                columns=tuple(reordered),
                visuals=tuple(visuals),
                captions=tuple(captions),
            )
        )

    return regions


def _apply_spanning_article_region(
    seq: Sequence["Block"],
    *,
    region: _SpanningArticleRegion,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    seq = list(seq)
    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))
    body_blocks = [block for column in region.columns for block in column]
    moving = [region.title, region.subtitle] + body_blocks + list(region.visuals) + list(region.captions)

    _mark(region.title, region_id=region.region_id, region_kind=region.region_kind, region_role="title")
    _mark(region.subtitle, region_id=region.region_id, region_kind=region.region_kind, region_role="subtitle")
    for col_idx, column in enumerate(region.columns):
        for block in column:
            _mark(
                block,
                region_id=region.region_id,
                region_kind=region.region_kind,
                region_role="member",
                region_col_index=col_idx,
            )
    for block in region.visuals:
        _mark(block, region_id=region.region_id, region_kind=region.region_kind, region_role="visual")
    for block in region.captions:
        _mark(block, region_id=region.region_id, region_kind=region.region_kind, region_role="caption")

    moving_ids = {id(block) for block in moving}
    overhang = [
        block for block in seq
        if block.block_type == BlockType.TEXT
        and _y1(block) < _y1(region.title)
        and _y2(block) > _y1(region.title) + page_h * 0.02
        and _projection_overlap_ratio_x(region.title, block) < 0.10
    ]
    remain = [block for block in seq if id(block) not in moving_ids]
    if overhang:
        last_overhang = max(seq.index(block) for block in overhang)
        anchor_block = seq[last_overhang]
        insert_pos = 0
        for i, block in enumerate(remain):
            if block is anchor_block:
                insert_pos = i + 1
                break
        return remain[:insert_pos] + moving + remain[insert_pos:]

    first_idx = min(seq.index(region.title), seq.index(region.subtitle), *(seq.index(block) for block in body_blocks))
    last_idx = max(seq.index(block) for block in body_blocks)
    middle = [block for block in seq[first_idx:last_idx + 1] if id(block) not in moving_ids]
    return seq[:first_idx] + moving + middle + seq[last_idx + 1:]


def _enforce_spanning_article_column_continuity(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    seq = list(ordered)
    if _has_locked_global_multicol_skeleton(
        seq,
        image_width=image_width,
        image_height=image_height,
    ):
        return seq
    for region in _find_spanning_article_regions(
        seq,
        image_width=image_width,
        image_height=image_height,
    ):
        seq = _apply_spanning_article_region(
            seq,
            region=region,
            image_width=image_width,
            image_height=image_height,
        )
    return seq


def _is_local_parallel_region_member(
    block: "Block",
    *,
    page_w: float,
) -> bool:
    if block.block_type not in {BlockType.TEXT, BlockType.TITLE, BlockType.REFERENCE, BlockType.ABSTRACT}:
        return False
    if block.block_type in _STRIP_TYPES or block.block_type in _CAPTION_TYPES:
        return False
    if _w(block) > page_w * 0.38:
        return False
    text = re.sub(r"\s+", "", _block_text(block))
    if len(text) <= 2 and _line_count(block) <= 1:
        return False
    return True


def _vertical_groups_for_local_parallel_regions(
    blocks: Sequence["Block"],
    *,
    page_h: float,
) -> List[List["Block"]]:
    groups: List[List["Block"]] = []
    current: List["Block"] = []
    current_bottom = 0.0
    max_gap = max(90.0, page_h * 0.055)
    for block in sorted(blocks, key=lambda b: (_y1(b), _x1(b))):
        if not current or _y1(block) <= current_bottom + max_gap:
            current.append(block)
            current_bottom = max(current_bottom, _y2(block))
        else:
            groups.append(current)
            current = [block]
            current_bottom = _y2(block)
    if current:
        groups.append(current)
    return groups


def _column_quality_for_local_region(
    columns: Sequence[Sequence["Block"]],
    *,
    band_top: float,
    band_bottom: float,
    page_w: float,
    page_h: float,
) -> bool:
    if len(columns) < 3:
        return False
    if band_bottom - band_top > page_h * 0.55:
        return False

    total = sum(len(col) for col in columns)
    for col in columns:
        for block in col:
            if block.block_type != BlockType.TITLE:
                continue
            title_span = sum(
                1 for other_col in columns
                if any(_overlap_1d(_x1(block), _x2(block), _x1(other), _x2(other)) > 0 for other in other_col)
            )
            if 1 < title_span < len(columns):
                return False
    all_columns_are_tall_singletons = all(
        len(col) == 1
        and (
            _line_count(col[0]) >= 3
            or _h(col[0]) >= page_h * 0.06
        )
        for col in columns
        if col
    )
    if total < max(4, len(columns) + 2) and not all_columns_are_tall_singletons:
        return False

    filled_cols = 0
    tall_cols = 0
    col_tops: List[float] = []
    col_bottoms: List[float] = []
    for col in columns:
        if not col:
            continue
        filled_cols += 1
        col_top = min(_y1(block) for block in col)
        col_bottom = max(_y2(block) for block in col)
        col_tops.append(col_top)
        col_bottoms.append(col_bottom)
        col_height = max(0.0, col_bottom - col_top)
        has_long_member = any(_line_count(block) >= 3 or _h(block) >= page_h * 0.06 for block in col)
        if len(col) >= 2 or has_long_member:
            tall_cols += 1
        if _merged_y_coverage(col, max(band_bottom - band_top, 1.0)) < 0.04 and not has_long_member:
            return False

    if filled_cols < len(columns):
        return False
    if tall_cols < max(2, len(columns) - 1):
        return False
    if col_tops and col_bottoms:
        top_spread = max(col_tops) - min(col_tops)
        bottom_spread = max(col_bottoms) - min(col_bottoms)
        if top_spread > max(72.0, page_h * 0.045) and bottom_spread > max(180.0, page_h * 0.14):
            return False

    centers = [sum(_cx(block) for block in col) / max(len(col), 1) for col in columns if col]
    centers.sort()
    if len(centers) >= 2:
        min_gap = min(b - a for a, b in zip(centers, centers[1:]))
        if min_gap < page_w * 0.12:
            return False

    return True


def _find_local_parallel_text_regions(
    seq: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List[_LocalParallelRegion]:
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))
    candidates = [
        block for block in seq
        if _is_local_parallel_region_member(block, page_w=page_w)
    ]
    if len(candidates) < 3:
        return []

    regions: List[_LocalParallelRegion] = []
    region_counter = 0
    for group in _vertical_groups_for_local_parallel_regions(candidates, page_h=page_h):
        if len(group) < 3:
            continue
        band_top = min(_y1(block) for block in group)
        band_bottom = max(_y2(block) for block in group)
        if band_bottom - band_top < max(120.0, page_h * 0.08):
            continue

        columns, col_bounds = detect_columns(
            group,
            int(page_w),
            max_cols=4,
            cluster_thresh=min(COLUMN_CLUSTER_THRESH, 0.08),
        )
        partial_title = False
        for block in seq:
            if block.block_type != BlockType.TITLE:
                continue
            if _y2(block) < band_top - 8.0 or _y1(block) > band_bottom + 8.0:
                continue
            span_count = sum(
                1 for cx1, cx2 in col_bounds
                if _overlap_1d(_x1(block), _x2(block), float(cx1), float(cx2)) > 0
            )
            if 1 < span_count < len(columns):
                partial_title = True
                break
        if partial_title:
            continue
        if not _column_quality_for_local_region(
            columns,
            band_top=band_top,
            band_bottom=band_bottom,
            page_w=page_w,
            page_h=page_h,
        ):
            continue
        region_counter += 1
        regions.append(
            _LocalParallelRegion(
                region_id=f"local_parallel_{region_counter}",
                region_kind="local_parallel_text_band",
                blocks=tuple(group),
                columns=tuple(tuple(column) for column in columns),
                bounds=tuple((float(x1), float(x2)) for x1, x2 in col_bounds),
                top=float(band_top),
                bottom=float(band_bottom),
            )
        )

    return regions


def _select_region_prefix_blocks(
    seq: Sequence["Block"],
    *,
    region: _LocalParallelRegion,
    page_w: float,
    page_h: float,
) -> tuple["Block", ...]:
    group_ids = {id(block) for block in region.blocks}
    prefix_candidates: List[tuple[float, int, "Block"]] = []
    for block in seq:
        if id(block) in group_ids:
            continue
        block_text = re.sub(r"\s+", "", _block_text(block))
        centered = abs(_cx(block) - page_w * 0.5) <= page_w * 0.18
        vertical_gap = region.top - _y2(block)
        short_label = len(block_text) <= 4 and _line_count(block) <= 1
        if centered and short_label and 0.0 <= vertical_gap <= max(140.0, page_h * 0.08):
            prefix_candidates.append((vertical_gap, seq.index(block), block))
    if not prefix_candidates:
        return ()
    _gap, _idx, prefix_block = sorted(prefix_candidates)[0]
    return (prefix_block,)


def _region_placement(
    seq: Sequence["Block"],
    *,
    region: _LocalParallelRegion,
    prefix: Sequence["Block"],
) -> _RegionPlacement:
    first_idx = min(seq.index(block) for block in region.blocks)
    last_idx = max(seq.index(block) for block in region.blocks)
    return _RegionPlacement(
        first_idx=first_idx,
        last_idx=last_idx,
        prefix=tuple(prefix),
    )


def _apply_local_parallel_region(
    seq: Sequence["Block"],
    *,
    region: _LocalParallelRegion,
    page_w: float,
    page_h: float,
) -> List["Block"]:
    seq = list(seq)
    group = list(region.blocks)
    columns = [list(column) for column in region.columns]
    col_bounds = list(region.bounds)
    band_top = float(region.top)
    band_bottom = float(region.bottom)
    group_ids = {id(block) for block in group}
    prefix_ids: set[int] = set()

    prefix = list(_select_region_prefix_blocks(seq, region=region, page_w=page_w, page_h=page_h))
    prefix_ids = {id(block) for block in prefix}

    for prefix_block in prefix:
        prefix_block.col_count = len(columns)
        prefix_block.col_index = 0
        prefix_block.spanned_cols = list(range(len(columns)))
        _mark(prefix_block, region_id=region.region_id, region_kind=region.region_kind, region_role="prefix")

    reordered: List["Block"] = []
    for col_idx, column in enumerate(columns):
        members = sorted(column, key=lambda b: (_y1(b), _x1(b)))
        for block in members:
            block.col_count = len(columns)
            block.col_index = col_idx
            block.spanned_cols = [col_idx]
            _mark(
                block,
                region_id=region.region_id,
                region_kind=region.region_kind,
                region_role="member",
                region_col_index=col_idx,
            )
        reordered.extend(members)

    remain = [
        block for block in seq
        if id(block) not in group_ids and id(block) not in prefix_ids
    ]
    top_slack = max(24.0, page_h * 0.015)
    attached: List["Block"] = []
    before: List["Block"] = []
    after: List["Block"] = []
    for block in remain:
        if _intersects_region(block, (0.0, band_top, page_w, band_bottom)):
            detect_spanned_blocks([block], col_bounds)
            block.col_count = len(columns)
            _mark(block, region_id=region.region_id, region_kind=region.region_kind, region_role="attached")
            attached.append(block)
        elif _y2(block) <= band_top + top_slack:
            block.col_count = 1
            block.col_index = 0
            block.spanned_cols = [0]
            before.append(block)
        else:
            block.col_count = 1
            block.col_index = 0
            block.spanned_cols = [0]
            after.append(block)

    return before + prefix + reordered + attached + after


def _enforce_local_parallel_text_band_columns(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    """Recover local multi-column text bands inside otherwise single-column pages."""
    seq = list(ordered)
    if len(seq) < 3:
        return seq
    if _has_locked_global_multicol_skeleton(
        seq,
        image_width=image_width,
        image_height=image_height,
    ):
        return seq

    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in seq), default=1.0))
    for region in _find_local_parallel_text_regions(
        seq,
        image_width=image_width,
        image_height=image_height,
    ):
        seq = _apply_local_parallel_region(
            seq,
            region=region,
            page_w=page_w,
            page_h=page_h,
        )

    return seq


def _looks_like_stable_academic_two_column_page(
    blocks: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> bool:
    if len(blocks) < 10:
        return False
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(block) for block in blocks), default=1.0))

    textlike = [
        block for block in blocks
        if block.block_type in {BlockType.TEXT, BlockType.TITLE, BlockType.REFERENCE, BlockType.ABSTRACT}
        and block.block_type not in _STRIP_TYPES
        and _w(block) <= page_w * 0.48
    ]
    if len(textlike) < 8:
        return False

    left = [block for block in textlike if _cx(block) <= page_w * 0.46]
    right = [block for block in textlike if _cx(block) >= page_w * 0.54]
    if len(left) < 3 or len(right) < 3:
        return False

    if _merged_y_coverage(left, page_h) < 0.18 or _merged_y_coverage(right, page_h) < 0.18:
        return False

    table_count = sum(1 for block in blocks if block.block_type == BlockType.TABLE)
    equation_count = sum(1 for block in blocks if block.block_type in {BlockType.FORMULA, BlockType.EQUATION})
    academic_caption_count = sum(1 for block in blocks if block.block_type in {BlockType.TABLE_CAPTION, BlockType.FORMULA_CAPTION})
    inline_number_count = sum(
        1 for block in blocks
        if block.block_type in _CAPTION_TYPES
        and re.fullmatch(r"\(?\s*\d{1,3}[a-zA-Z]?\s*\)?", re.sub(r"\s+", "", _block_text(block)) or "")
    )
    title_count = sum(1 for block in blocks if block.block_type == BlockType.TITLE)
    cjk_chars = sum(1 for block in blocks for ch in _block_text(block) if "\u4e00" <= ch <= "\u9fff")
    all_chars = sum(1 for block in blocks for ch in _block_text(block) if not ch.isspace())
    cjk_ratio = cjk_chars / max(all_chars, 1)

    academic_cues = table_count + equation_count + academic_caption_count + inline_number_count
    if academic_cues < 3:
        return False
    if cjk_ratio > 0.20:
        return False
    if title_count > 10:
        return False
    return True


def _enforce_stable_academic_two_column_order(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    seq = list(ordered)
    if not _looks_like_stable_academic_two_column_page(
        seq,
        image_width=image_width,
        image_height=image_height,
    ):
        return seq

    single_col: dict[int, List["Block"]] = {}
    spanned: List["Block"] = []
    for block in seq:
        cols = list(getattr(block, "spanned_cols", []) or [getattr(block, "col_index", 0)])
        if len(cols) == 1:
            single_col.setdefault(int(cols[0]), []).append(block)
        else:
            spanned.append(block)

    if len(single_col) != 2:
        return seq

    all_single = [block for members in single_col.values() for block in members]
    top_y = min((_y1(block) for block in all_single), default=0.0)
    bottom_y = max((_y2(block) for block in all_single), default=0.0)
    head = [block for block in spanned if _y2(block) <= top_y + 48.0]
    tail = [block for block in spanned if _y1(block) >= bottom_y - 48.0]
    middle = [block for block in spanned if block not in head and block not in tail]

    result: List["Block"] = []
    result.extend(_sort_yx(head))
    for col_idx in sorted(single_col.keys()):
        members = sorted(single_col[col_idx], key=lambda b: (_y1(b), _x1(b), _y2(b), _x2(b)))
        member_ids = {id(block) for block in members}
        attached_middle = [
            block for block in middle
            if block.block_type in _VISUAL_TYPES
            and any(_overlap_1d(_x1(block), _x2(block), _x1(member), _x2(member)) > 0 for member in members)
        ]
        attached_tail = [
            block for block in tail
            if block.block_type in _CAPTION_TYPES
            and any(
                table.block_type == BlockType.TABLE
                and _y1(table) >= _y2(block) - 12.0
                and _overlap_1d(_x1(block), _x2(block), _x1(table), _x2(table)) > 0
                for table in attached_middle
            )
        ]
        family = _sort_yx(members + attached_tail + attached_middle)
        result.extend(family)
        for block in attached_middle + attached_tail:
            if block in middle:
                middle.remove(block)
            if block in tail:
                tail.remove(block)
    result.extend(_sort_yx(middle))
    result.extend(_sort_yx(tail))

    if len(result) != len(seq):
        return seq
    for idx, block in enumerate(result):
        _mark(block, academic_two_col_order=idx)
    return result


def postprocess_xycutpp_local_attachments(
    ordered: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> List["Block"]:
    """Apply thin local attachment rules after the paper-style core.

    These rules should only repair local family coherence and should not change
    the global page skeleton established by the XY-Cut++ core.
    """
    seq = list(ordered)
    seq = _enforce_inline_equation_label_adjacency(seq)
    seq = _enforce_parallel_figure_group_order(seq, image_height=image_height)
    seq = _enforce_table_family_order(seq)
    seq = _enforce_figure_family_order(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _reorder_side_visual_families_by_caption_anchor(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _promote_upper_visual_family_before_lower_band(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _enforce_local_title_before_side_visual(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _enforce_column_local_visual_neighbors(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _defer_side_figure_family_until_body_continuation(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _enforce_spanning_visual_after_covered_columns(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _delay_spanning_title_until_prior_overhang_resolves(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _enforce_section_band_leader_order(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _enforce_spanning_article_column_continuity(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _enforce_peripheral_sidebar_demote(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _enforce_column_major_on_parallel_table_figures(seq)
    seq = _sort_same_column_text_runs_by_geometry(seq)
    seq = _promote_textbook_intro_sidebar(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _enforce_local_parallel_text_band_columns(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _enforce_stable_academic_two_column_order(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _enforce_figure_family_order(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _reorder_side_visual_families_by_caption_anchor(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _enforce_local_title_before_side_visual(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _enforce_lower_section_wraparound_columns(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    seq = _enforce_inline_equation_label_adjacency(seq)
    _assign_centered_section_starts_to_spanned_columns(
        seq,
        page_w=max(float(image_width), 1.0),
        page_h=max(float(image_height or 0), max((_y2(block) for block in seq), default=1.0)),
        col_count=max((int(getattr(block, "col_count", 1) or 1) for block in seq), default=1),
    )
    _sync_local_title_visual_columns(seq)
    return seq


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _sort_layout_xycutpp_core(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None = None,
    max_cols: int = MAX_COLS,
    cluster_thresh: float = COLUMN_CLUSTER_THRESH,
    column_confidence_min: float = 0.55,
    zone_strip_height_ratio: float = 0.12,
    beta: float = 1.3,
    density_threshold: float = 0.9,
    min_gap_ratio: float = 0.015,
    min_projection_overlap: int = 2,
    title_width_ratio: float = 0.45,
    overlap_threshold: float = 0.10,
    barrier_width_ratio: float = 0.55,
    near_text_margin_ratio: float = 0.018,
) -> List["Block"]:
    """Sort blocks by the paper-style XY-Cut++ core pipeline.

    Legacy parameters are kept for call-site compatibility even if unused by
    the current pipeline.
    """
    del column_confidence_min, zone_strip_height_ratio, title_width_ratio

    valid = [blk for blk in blocks if blk is not None and getattr(blk, "bbox", None) is not None]
    if not valid:
        return []
    if len(valid) <= 1:
        _assign_single_column(valid)
        if valid:
            _mark(valid[0], final_order=0, strategy="xycutpp_proto")
        return valid

    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in valid), default=1.0))
    min_gap_px = max(5.0, min(page_w, page_h) * float(min_gap_ratio))

    cross_ids = _detect_cross_layout_blocks(
        valid,
        image_width=image_width,
        image_height=image_height,
        beta=beta,
        min_projection_overlap=min_projection_overlap,
        overlap_threshold=overlap_threshold,
    )
    active, masked = _split_mask_sets(
        valid,
        cross_ids=cross_ids,
        image_width=image_width,
        image_height=image_height,
        near_text_margin_ratio=near_text_margin_ratio,
    )

    anchors = _sort_active_anchors(
        active,
        masked,
        image_width=image_width,
        image_height=image_height,
        cross_ids=cross_ids,
        density_threshold=density_threshold,
        min_gap_px=min_gap_px,
        barrier_width_ratio=barrier_width_ratio,
    )
    ordered = _restore_masked_elements(
        anchors,
        masked,
        cross_ids=cross_ids,
        image_width=image_width,
        image_height=image_height,
        barrier_width_ratio=barrier_width_ratio,
    )

    _assign_column_metadata(
        ordered,
        image_width=image_width,
        image_height=image_height,
        max_cols=max_cols,
        cluster_thresh=cluster_thresh,
    )

    for idx, blk in enumerate(ordered):
        _mark(
            blk,
            strategy="xycutpp_proto",
            final_order=idx,
            final_col_count=int(getattr(blk, "col_count", 1) or 1),
            final_col_index=int(getattr(blk, "col_index", 0) or 0),
            final_spanned_cols=list(getattr(blk, "spanned_cols", [0]) or [0]),
        )
    return ordered


def sort_layout_xycutpp(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None = None,
    max_cols: int = MAX_COLS,
    cluster_thresh: float = COLUMN_CLUSTER_THRESH,
    column_confidence_min: float = 0.55,
    zone_strip_height_ratio: float = 0.12,
    beta: float = 1.3,
    density_threshold: float = 0.9,
    min_gap_ratio: float = 0.015,
    min_projection_overlap: int = 2,
    title_width_ratio: float = 0.45,
    overlap_threshold: float = 0.10,
    barrier_width_ratio: float = 0.55,
    near_text_margin_ratio: float = 0.018,
) -> List["Block"]:
    """Sort blocks by XY-Cut++ core, then apply thin local attachment rules."""
    ordered = _sort_layout_xycutpp_core(
        blocks,
        image_width=image_width,
        image_height=image_height,
        max_cols=max_cols,
        cluster_thresh=cluster_thresh,
        column_confidence_min=column_confidence_min,
        zone_strip_height_ratio=zone_strip_height_ratio,
        beta=beta,
        density_threshold=density_threshold,
        min_gap_ratio=min_gap_ratio,
        min_projection_overlap=min_projection_overlap,
        title_width_ratio=title_width_ratio,
        overlap_threshold=overlap_threshold,
        barrier_width_ratio=barrier_width_ratio,
        near_text_margin_ratio=near_text_margin_ratio,
    )
    ordered = postprocess_xycutpp_local_attachments(
        ordered,
        image_width=image_width,
        image_height=image_height,
    )
    for idx, blk in enumerate(ordered):
        _mark(blk, post_core_attachment_order=idx)
    return ordered


def sort_layout_xycutpp_paper_proto(*args, **kwargs):
    """Compatibility alias for the integrated prototype name."""
    return sort_layout_xycutpp(*args, **kwargs)
