"""Integrated XY-Cut++ sorter for DocFlow.

This module replaces the previous in-project XY-Cut++ implementation with the
prototype provided under ``Code/xycutpp_prototype_for_docflow``.

Properties:
- input: ``List[docflow.model.base.Block]``
- output: reordered blocks with ``col_count``, ``col_index`` and
  ``spanned_cols`` populated for downstream renderers
- debug: writes trace fields under ``block.attributes['xycutpp_proto']``

Pipeline:
1. Cross-layout detection via ``beta * median(width)``.
2. Pre-mask title / figure / table / formula / caption / strip elements.
3. Coarse Y-band segmentation using wide masked barriers.
4. Adaptive XY/YX recursive sorting on remaining anchors.
5. Semantic + geometry-aware restoration of masked elements.
6. Column metadata assignment for renderers.
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


def _text_isolation_gap(block: "Block", blocks: Sequence["Block"]) -> float:
    gap = float("inf")
    for other in blocks:
        if other is block or other.block_type != BlockType.TEXT:
            continue
        gap = min(gap, _edge_gap(block, other))
    return gap


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


def _is_top_centered_short_text(
    block: "Block",
    *,
    image_width: int,
    image_height: Optional[int],
) -> bool:
    if block.block_type not in _TEXTLIKE_TYPES or block.block_type in _STRIP_TYPES:
        return False
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max(_y2(block), 1.0))
    return (
        _line_count(block) <= 2
        and _w(block) <= page_w * 0.36
        and _y2(block) <= page_h * 0.18
        and abs(_cx(block) - page_w * 0.5) <= page_w * 0.22
    )


def _candidate_widths_for_median(blocks: Sequence["Block"], image_width: int) -> List[float]:
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
    return widths or [_w(blk) for blk in blocks]


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


def _has_stable_narrow_column_skeleton(
    bodylike_blocks: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> tuple[bool, List[List["Block"]], List[tuple[float, float]]]:
    columns, col_bounds = detect_columns(
        list(bodylike_blocks),
        image_width,
        max_cols=MAX_COLS,
        cluster_thresh=min(COLUMN_CLUSTER_THRESH, 0.08),
    )
    if len(columns) <= 1:
        return False, columns, col_bounds

    total_blocks = sum(len(col) for col in columns)
    if len(columns) == 2 and total_blocks == 2:
        left = columns[0][0]
        right = columns[1][0]
        page_h = max(float(image_height or 0), max(_y2(left), _y2(right), 1.0))
        if min(_y1(left), _y1(right)) >= page_h * 0.78:
            return False, columns, col_bounds
        y_overlap = _overlap_1d(_y1(left), _y2(left), _y1(right), _y2(right))
        min_h = max(1.0, min(_h(left), _h(right)))
        center_gap = abs(_cx(left) - _cx(right))
        page_w = max(float(image_width), 1.0)
        if y_overlap >= min_h * 0.35 and center_gap >= page_w * 0.18:
            return True, columns, col_bounds

    page_h = max(float(image_height or 0), max((_y2(blk) for blk in bodylike_blocks), default=1.0))
    substantial_cols = 0
    for col in columns:
        if len(col) >= 2 and _merged_y_coverage(col, page_h) >= 0.08:
            substantial_cols += 1

    stable = substantial_cols >= 2 and total_blocks >= max(4, len(columns) * 2)
    return stable, columns, col_bounds


def _has_single_column_wide_body_dominance(
    blocks: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> bool:
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(blk) for blk in blocks), default=1.0))
    wide_body = [
        blk for blk in blocks
        if blk.block_type in {BlockType.TEXT, BlockType.ABSTRACT, BlockType.REFERENCE}
        and blk.block_type not in _STRIP_TYPES
        and _w(blk) >= page_w * 0.60
        and (_line_count(blk) >= 2 or _h(blk) >= page_h * 0.025)
    ]
    if len(wide_body) < 2:
        return False

    wide_coverage = _merged_y_coverage(wide_body, page_h)
    if wide_coverage < 0.22:
        return False

    narrow_body = [
        blk for blk in blocks
        if blk.block_type == BlockType.TEXT
        and _w(blk) <= page_w * 0.48
        and _line_count(blk) >= 1
    ]
    stable_columns, _columns, _bounds = _has_stable_narrow_column_skeleton(
        narrow_body,
        image_width=image_width,
        image_height=image_height,
    )
    return not stable_columns


def _detect_cross_layout_blocks(
    blocks: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
    beta: float,
    min_projection_overlap: int,
    overlap_threshold: float,
) -> set[int]:
    if len(blocks) < 3:
        return set()

    if _has_single_column_wide_body_dominance(
        blocks,
        image_width=image_width,
        image_height=image_height,
    ):
        for blk in blocks:
            _mark(blk, cross_candidate=False, cross_threshold=0.0, cross_overlap_count=0)
        return set()

    text_candidates = [
        blk for blk in blocks
        if blk.block_type in _TEXTLIKE_TYPES and blk.block_type not in _STRIP_TYPES
    ]
    bodylike_probe = [
        blk for blk in text_candidates
        if blk.block_type == BlockType.TEXT
        and _w(blk) <= max(float(image_width), 1.0) * 0.48
        and not _is_top_centered_short_text(
            blk,
            image_width=image_width,
            image_height=image_height,
        )
    ]
    if len(bodylike_probe) < 2:
        for blk in blocks:
            _mark(blk, cross_candidate=False, cross_threshold=0.0, cross_overlap_count=0)
        return set()

    stable_columns, _columns, col_bounds = _has_stable_narrow_column_skeleton(
        bodylike_probe,
        image_width=image_width,
        image_height=image_height,
    )
    if not stable_columns:
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
            _mark(blk, cross_candidate=False, cross_threshold=round(threshold, 2), cross_overlap_count=0)
            continue
        bw = _w(blk)
        length_hit = bw > threshold or bw >= page_w * 0.58
        if not length_hit:
            _mark(blk, cross_candidate=False, cross_threshold=round(threshold, 2))
            continue

        spanned_col_count = sum(
            1 for cx1, cx2 in col_bounds
            if _overlap_1d(_x1(blk), _x2(blk), float(cx1), float(cx2)) > 0
        )

        overlap_count = 0
        for other in blocks:
            if other is blk:
                continue
            if _projection_overlap_ratio_x(blk, other) >= overlap_threshold:
                overlap_count += 1
                if overlap_count >= min_projection_overlap:
                    break

        is_cross = overlap_count >= min_projection_overlap and spanned_col_count >= 2
        _mark(
            blk,
            cross_candidate=is_cross,
            cross_threshold=round(threshold, 2),
            cross_overlap_count=overlap_count,
            cross_spanned_cols=spanned_col_count,
        )
        if is_cross:
            cross_ids.add(id(blk))
    return cross_ids


def _is_isolated_central_dynamic(
    block: "Block",
    blocks: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
    near_text_margin_ratio: float,
) -> bool:
    # Our current pre-cut implementation only produces full-width horizontal
    # bands. Visual islands work well with that approximation, but ordinary
    # section titles do not: using column-local titles as page-level barriers
    # breaks stable double-column pages such as academic articles. Keep title
    # restoration in CMM and reserve pre-cut for isolated visuals only.
    if block.block_type not in _VISUAL_TYPES:
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
    if _has_column_local_text_neighbor(
        block,
        blocks,
        image_width=image_width,
        image_height=image_height,
    ):
        return False

    margin = max(page_w, page_h) * near_text_margin_ratio
    min_text_gap = _text_isolation_gap(block, blocks)
    return min_text_gap > margin


def _is_top_centered_attachment(
    block: "Block",
    blocks: Sequence["Block"],
    *,
    cross_ids: set[int],
    image_width: int,
    image_height: Optional[int],
) -> bool:
    """Return true for short top lines that belong to a spanning page title."""
    if block.block_type != BlockType.TEXT or _line_count(block) != 1:
        return False
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in blocks), default=1.0))
    text = _block_text(block).strip()
    if not text or len(text) > 48:
        return False
    if not (page_w * 0.10 <= _w(block) <= page_w * 0.36):
        return False
    if _h(block) > max(page_h * 0.035, 40.0):
        return False
    if _y1(block) > page_h * 0.24:
        return False
    if abs(_cx(block) - page_w * 0.5) > page_w * 0.18:
        return False

    return any(
        other is not block
        and other.block_type == BlockType.TITLE
        and (id(other) in cross_ids or _w(other) >= page_w * 0.55)
        and _y2(other) <= _y1(block) + max(24.0, page_h * 0.025)
        and _y1(other) <= page_h * 0.16
        for other in blocks
    )


def _is_column_structural_visual(
    block: "Block",
    blocks: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
) -> bool:
    """Column-local figures should participate in XY-cut partitioning.

    Captions and floating/cross-column visuals are still restored semantically,
    but a large in-column figure is real column content. Masking it removes
    important projection mass and can make the opposite column appear first.
    """
    if block.block_type != BlockType.FIGURE:
        return False
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max(_y2(block), 1.0))
    if _w(block) > page_w * 0.42:
        return False
    if _w(block) < page_w * 0.16 or _h(block) < page_h * 0.10:
        return False
    if _area(block) < page_w * page_h * 0.025:
        return False
    return _has_column_local_text_neighbor(
        block,
        blocks,
        image_width=image_width,
        image_height=image_height,
    )


def _split_mask_sets(
    blocks: Sequence["Block"],
    *,
    cross_ids: set[int],
    image_width: int,
    image_height: Optional[int],
    near_text_margin_ratio: float,
) -> tuple[List["Block"], List["Block"], List["Block"]]:
    active: List["Block"] = []
    masked: List["Block"] = []
    precut_targets: List["Block"] = []
    for blk in blocks:
        is_cross = id(blk) in cross_ids
        text = _block_text(blk).strip()
        is_numbered_section_title = (
            blk.block_type == BlockType.TITLE
            and bool(text)
            and _NUMBERED_SECTION_RE.match(text) is not None
        )
        is_precut_target = _is_isolated_central_dynamic(
            blk,
            blocks,
            image_width=image_width,
            image_height=image_height,
            near_text_margin_ratio=near_text_margin_ratio,
        )
        is_top_attachment = _is_top_centered_attachment(
            blk,
            blocks,
            cross_ids=cross_ids,
            image_width=image_width,
            image_height=image_height,
        )
        is_structural_visual = (
            not is_cross
            and not is_precut_target
            and _is_column_structural_visual(
                blk,
                blocks,
                image_width=image_width,
                image_height=image_height,
            )
        )
        should_mask = (
            is_cross
            or is_top_attachment
            or (
                blk.block_type in _DYNAMIC_MASK_TYPES
                and not is_structural_visual
                and not is_numbered_section_title
            )
            or is_precut_target
        )
        if should_mask:
            masked.append(blk)
            if is_precut_target:
                precut_targets.append(blk)
            _mark(
                blk,
                phase="pre_mask",
                is_cross_layout=is_cross,
                is_precut_target=is_precut_target,
                mask_reason="cross" if is_cross else ("top_attachment" if is_top_attachment else blk.block_type.value),
            )
        else:
            active.append(blk)
            _mark(blk, phase="anchor", is_cross_layout=False)
    return active, masked, precut_targets


def _split_region_around_target(
    region: tuple[float, float, float, float],
    target: "Block",
    *,
    min_gap: float,
) -> List[tuple[float, float, float, float]]:
    rx1, ry1, rx2, ry2 = region
    tx1 = max(rx1, _x1(target))
    ty1 = max(ry1, _y1(target))
    tx2 = min(rx2, _x2(target))
    ty2 = min(ry2, _y2(target))
    if tx2 <= tx1 or ty2 <= ty1:
        return [region]

    pieces: List[tuple[float, float, float, float]] = []
    if ty1 - ry1 >= min_gap:
        pieces.append((rx1, ry1, rx2, ty1))
    if tx1 - rx1 >= min_gap and ty2 - ty1 >= min_gap:
        pieces.append((rx1, ty1, tx1, ty2))
    if rx2 - tx2 >= min_gap and ty2 - ty1 >= min_gap:
        pieces.append((tx2, ty1, rx2, ty2))
    if ry2 - ty2 >= min_gap:
        pieces.append((rx1, ty2, rx2, ry2))
    return pieces or [region]


def _coarse_precut_regions(
    active: Sequence["Block"],
    precut_targets: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
    min_band_gap: float,
) -> List[tuple[tuple[float, float, float, float], List["Block"]]]:
    if not active:
        return []
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max((_y2(b) for b in active), default=1.0))
    if not precut_targets:
        return [((0.0, 0.0, page_w, page_h), list(active))]

    regions: List[tuple[float, float, float, float]] = [(0.0, 0.0, page_w, page_h)]
    for target in _sort_yx(precut_targets):
        next_regions: List[tuple[float, float, float, float]] = []
        applied = False
        tcx, tcy = _cx(target), _cy(target)
        for region in regions:
            rx1, ry1, rx2, ry2 = region
            target_inside = (rx1 <= tcx <= rx2) and (ry1 <= tcy <= ry2)
            if not applied and target_inside:
                split_regions = _split_region_around_target(region, target, min_gap=min_band_gap)
                next_regions.extend(split_regions)
                applied = True
            else:
                next_regions.append(region)
        regions = next_regions

    memberships: List[tuple[tuple[float, float, float, float], List["Block"]]] = []
    assigned_ids: set[int] = set()
    for region in regions:
        members: List["Block"] = []
        for blk in active:
            if id(blk) in assigned_ids:
                continue
            inter = _region_intersection_area(blk, region)
            if inter <= 0.0:
                continue
            best_inter = inter
            is_best = True
            for other in regions:
                if other is region:
                    continue
                other_inter = _region_intersection_area(blk, other)
                if other_inter > best_inter:
                    is_best = False
                    break
            if is_best:
                members.append(blk)
                assigned_ids.add(id(blk))
        if members:
            memberships.append((region, sorted(members, key=lambda b: (_y1(b), _x1(b)))))

    leftovers = [blk for blk in active if id(blk) not in assigned_ids]
    if leftovers:
        memberships.append(
            (
                (
                    min(_x1(b) for b in leftovers),
                    min(_y1(b) for b in leftovers),
                    max(_x2(b) for b in leftovers),
                    max(_y2(b) for b in leftovers),
                ),
                sorted(leftovers, key=lambda b: (_y1(b), _x1(b))),
            )
        )

    return memberships or [((0.0, 0.0, page_w, page_h), list(active))]


def _region_intersection_area(block: "Block", region: tuple[float, float, float, float]) -> float:
    rx1, ry1, rx2, ry2 = region
    return _overlap_1d(_x1(block), _x2(block), rx1, rx2) * _overlap_1d(_y1(block), _y2(block), ry1, ry2)


def _intersects_region(block: "Block", region: tuple[float, float, float, float]) -> bool:
    return _region_intersection_area(block, region) > 0.0


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


def _best_projection_cut(
    blocks: Sequence["Block"],
    axis: str,
    region: tuple[float, float, float, float],
) -> _Cut:
    if len(blocks) <= 1:
        return _Cut(axis=axis, position=0.0, gap=0.0)

    if axis == "x":
        start = int(region[0])
        end = int(region[2])
        intervals = [(max(start, int(_x1(b))), min(end, int(_x2(b)))) for b in blocks]
    else:
        start = int(region[1])
        end = int(region[3])
        intervals = [(max(start, int(_y1(b))), min(end, int(_y2(b)))) for b in blocks]
    if end - start <= 1:
        return _Cut(axis=axis, position=float(start), gap=0.0)
    valid_intervals = [(lo, hi) for lo, hi in intervals if hi > lo]
    if len(valid_intervals) <= 1:
        return _Cut(axis=axis, position=0.0, gap=0.0)

    hist = [0] * (end - start + 1)
    for lo, hi in valid_intervals:
        for i in range(lo - start, hi - start):
            hist[i] += 1

    content_start = min(lo for lo, _ in valid_intervals) - start
    content_end = max(hi for _, hi in valid_intervals) - start
    if content_end - content_start <= 1:
        return _Cut(axis=axis, position=0.0, gap=0.0)

    search = hist[content_start:content_end]
    min_val = min(search)
    best_run_start = 0
    best_run_len = 0
    run_start = 0
    run_len = 0
    for idx, val in enumerate(search):
        if val == min_val:
            if run_len == 0:
                run_start = idx
            run_len += 1
            if run_len > best_run_len:
                best_run_len = run_len
                best_run_start = run_start
        else:
            run_len = 0

    if best_run_len <= 0:
        return _Cut(axis=axis, position=0.0, gap=0.0)
    best_run_start += content_start
    best_pos = float(start + best_run_start + best_run_len * 0.5)
    return _Cut(axis=axis, position=best_pos, gap=float(best_run_len))


def _density_tau(
    context_blocks: Sequence["Block"],
    cross_ids: set[int],
    region: tuple[float, float, float, float],
) -> float:
    cross_area = sum(_region_intersection_area(b, region) for b in context_blocks if id(b) in cross_ids)
    single_area = sum(_region_intersection_area(b, region) for b in context_blocks if id(b) not in cross_ids)
    if single_area <= 0.0:
        return float("inf") if cross_area > 0.0 else 0.0
    return cross_area / single_area


def _fallback_sort_when_unsplittable(blocks: Sequence["Block"], *, image_width: int) -> List["Block"]:
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
    region: tuple[float, float, float, float],
    context_blocks: Sequence["Block"],
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
        return _fallback_sort_when_unsplittable(blocks, image_width=image_width)

    tau = _density_tau(context_blocks, cross_ids, region)
    primary_axis = "y" if tau > density_threshold else "x"
    secondary_axis = "x" if primary_axis == "y" else "y"

    cuts = {
        "x": _best_projection_cut(blocks, "x", region),
        "y": _best_projection_cut(blocks, "y", region),
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
            return _fallback_sort_when_unsplittable(blocks, image_width=image_width)

    split = _split_by_cut(blocks, chosen)
    if split is None:
        return _fallback_sort_when_unsplittable(blocks, image_width=image_width)

    first, second = split
    if chosen.axis == "y":
        first_region = (region[0], region[1], region[2], chosen.position)
        second_region = (region[0], chosen.position, region[2], region[3])
    else:
        first_region = (region[0], region[1], chosen.position, region[3])
        second_region = (chosen.position, region[1], region[2], region[3])
    first_context = [blk for blk in context_blocks if _intersects_region(blk, first_region)]
    second_context = [blk for blk in context_blocks if _intersects_region(blk, second_region)]
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
            region=first_region,
            context_blocks=first_context,
            cross_ids=cross_ids,
            density_threshold=density_threshold,
            min_gap_px=min_gap_px,
            depth=depth + 1,
            max_depth=max_depth,
        )
        + _recursive_adaptive_sort(
            second,
            image_width=image_width,
            region=second_region,
            context_blocks=second_context,
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
    precut_targets: Sequence["Block"],
    *,
    image_width: int,
    image_height: Optional[int],
    cross_ids: set[int],
    density_threshold: float,
    min_gap_px: float,
) -> List["Block"]:
    bands = _coarse_precut_regions(
        active,
        precut_targets,
        image_width=image_width,
        image_height=image_height,
        min_band_gap=max(4.0, min_gap_px),
    )
    ordered: List["Block"] = []
    for band_idx, (region, band) in enumerate(bands):
        band_context = [blk for blk in list(band) + list(masked) if _intersects_region(blk, region)]
        for blk in band:
            _mark(blk, coarse_band=band_idx)
        ordered.extend(
            _recursive_adaptive_sort(
                band,
                image_width=image_width,
                region=region,
                context_blocks=band_context,
                cross_ids=cross_ids,
                density_threshold=density_threshold,
                min_gap_px=min_gap_px,
            )
        )
    return ordered


def _priority(block: "Block", cross_ids: set[int]) -> int:
    if id(block) in cross_ids:
        return 3
    if block.block_type == BlockType.TITLE:
        return 2
    if block.block_type in _VISUAL_TYPES:
        return 1
    return 0


def _direction(block: "Block") -> str:
    return "horizontal" if _w(block) >= _h(block) else "vertical"


def _projection_score(a: "Block", b: "Block") -> float:
    if _direction(a) == "horizontal":
        overlap = _overlap_1d(_x1(a), _x2(a), _x1(b), _x2(b))
        union = max(_x2(a), _x2(b)) - min(_x1(a), _x1(b))
    else:
        overlap = _overlap_1d(_y1(a), _y2(a), _y1(b), _y2(b))
        union = max(_y2(a), _y2(b)) - min(_y1(a), _y1(b))
    return overlap / max(1.0, union)


def _geometry_distance(
    pending: "Block",
    anchor: "Block",
    *,
    image_width: int,
    image_height: Optional[int],
    cross_ids: set[int],
) -> float:
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max(_y2(pending), _y2(anchor), 1.0))
    scale = max(page_w, page_h, 1.0)

    direction_conflict = _direction(pending) != _direction(anchor)
    low_projection = _projection_score(pending, anchor) < 0.30
    phi1 = 1.0 if direction_conflict and low_projection else 0.0

    dx = abs(_cx(pending) - _cx(anchor))
    dy = abs(_cy(pending) - _cy(anchor))
    axis_aligned = _overlap_1d(_x1(pending), _x2(pending), _x1(anchor), _x2(anchor)) > 0 or _overlap_1d(_y1(pending), _y2(pending), _y1(anchor), _y2(anchor)) > 0
    phi2 = min(dx, dy) if axis_aligned else dx + dy

    if id(pending) in cross_ids and _y1(pending) > _y2(anchor):
        phi3 = -_y2(anchor)
    else:
        phi3 = _y1(anchor)

    phi4 = _x1(anchor)

    base = [scale * scale, scale, 1.0, 1.0 / scale]
    if pending.block_type == BlockType.TITLE:
        semantic = [1.0, 0.1, 0.1, 1.0] if _direction(pending) == "horizontal" else [0.2, 0.1, 1.0, 1.0]
    elif id(pending) in cross_ids:
        semantic = [1.0, 1.0, 0.1, 1.0]
    else:
        semantic = [1.0, 1.0, 1.0, 0.1]

    w1, w2, w3, w4 = [a * b for a, b in zip(base, semantic)]
    return w1 * phi1 + w2 * phi2 + w3 * phi3 + w4 * phi4


def _semantic_candidate_entries(
    pending: "Block",
    ordered_entries: Sequence[tuple[float, int, "Block"]],
    *,
    image_width: int,
    image_height: Optional[int],
    cross_ids: set[int],
    barrier_width_ratio: float,
) -> List[tuple[float, int, "Block"]]:
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max(_y2(pending), 1.0))
    page_scale = max(page_w, page_h, 1.0)
    candidates: List[tuple[tuple[float, float, float, float, float, float], tuple[float, int, "Block"]]] = []
    for anchor_pos, anchor_priority, anchor in ordered_entries:
        if pending.block_type == BlockType.TITLE:
            if anchor.block_type not in {
                BlockType.TITLE,
                BlockType.TEXT,
                BlockType.REFERENCE,
                BlockType.ABSTRACT,
                BlockType.CODE,
                BlockType.LIST,
            }:
                continue
        elif pending.block_type in _VISUAL_TYPES:
            if anchor.block_type in _CAPTION_TYPES or anchor.block_type in _STRIP_TYPES:
                continue
        elif pending.block_type in _CAPTION_TYPES:
            if anchor.block_type not in _VISUAL_TYPES:
                continue
        x_overlap = _projection_overlap_ratio_x(pending, anchor)
        y_overlap = _projection_overlap_ratio_y(pending, anchor)
        below_gap = max(0.0, _y1(anchor) - _y2(pending))
        above_gap = max(0.0, _y1(pending) - _y2(anchor))
        nearest_gap = max(below_gap, above_gap)
        center_dx = abs(_cx(pending) - _cx(anchor))

        if pending.block_type == BlockType.TITLE:
            is_spanning_title = id(pending) in cross_ids or _w(pending) >= page_w * 0.38
            if x_overlap < 0.12:
                if not is_spanning_title:
                    continue
                if nearest_gap > page_scale * 0.06:
                    continue
            overlap_tolerance = max(24.0, page_h * 0.018)
            future_bias = 0.0 if _y1(anchor) >= _y2(pending) - overlap_tolerance else 1.0
            pref = (future_bias, nearest_gap, -x_overlap, center_dx, _y1(anchor), _x1(anchor))
        elif pending.block_type in _VISUAL_TYPES:
            if x_overlap < 0.10 and nearest_gap > page_scale * 0.10:
                continue
            if _w(pending) >= page_w * barrier_width_ratio:
                direction_bias = 0.0 if _y2(anchor) <= _y1(pending) + 4.0 else 1.0
            else:
                direction_bias = 0.0 if _y1(anchor) >= _y2(pending) - 4.0 else 1.0
            pref = (direction_bias, -max(x_overlap, y_overlap), nearest_gap, center_dx, _y1(anchor), _x1(anchor))
        elif pending.block_type in _CAPTION_TYPES:
            edge_gap = _edge_gap(pending, anchor)
            if x_overlap < 0.12 and y_overlap < 0.40 and edge_gap > max(18.0, page_scale * 0.04):
                continue
            vertical_gap = min(abs(_y1(pending) - _y2(anchor)), abs(_y2(pending) - _y1(anchor)))
            pref = (0.0, vertical_gap, center_dx, -max(x_overlap, y_overlap), _y1(anchor), _x1(anchor))
        elif id(pending) in cross_ids:
            pref = (0.0, nearest_gap, center_dx, -x_overlap, _y1(anchor), _x1(anchor))
        else:
            continue
        candidates.append((pref, (anchor_pos, anchor_priority, anchor)))

    candidates.sort(key=lambda item: item[0])
    return [entry for _, entry in candidates]


def _choose_best_anchor(
    pending: "Block",
    search_entries: Sequence[tuple[float, int, "Block"]],
    *,
    image_width: int,
    image_height: Optional[int],
    cross_ids: set[int],
) -> tuple[float, "Block", float] | None:
    best_anchor_idx = 0.0
    best_anchor: Optional["Block"] = None
    best_dist = float("inf")
    for anchor_pos, _anchor_priority, anchor in search_entries:
        dcurr = 0.0
        # Paper-style early termination over accumulated phi terms.
        full_dist = _geometry_distance(
            pending,
            anchor,
            image_width=image_width,
            image_height=image_height,
            cross_ids=cross_ids,
        )
        dcurr = full_dist
        if dcurr < best_dist:
            best_dist = dcurr
            best_anchor_idx = anchor_pos
            best_anchor = anchor
    if best_anchor is None:
        return None
    return (best_anchor_idx, best_anchor, best_dist)


def _preferred_below_anchor(
    pending: "Block",
    ordered_entries: Sequence[tuple[float, int, "Block"]],
    *,
    allowed_types: Sequence[BlockType] | set[BlockType] | frozenset[BlockType],
    min_x_overlap: float,
) -> tuple[float, int, "Block"] | None:
    candidates: List[tuple[float, float, float, int, "Block"]] = []
    for anchor_pos, anchor_priority, anchor in ordered_entries:
        if anchor.block_type not in allowed_types:
            continue
        if _y1(anchor) < _y2(pending) - 4.0:
            continue
        if _projection_overlap_ratio_x(pending, anchor) < min_x_overlap:
            continue
        vertical_gap = max(0.0, _y1(anchor) - _y2(pending))
        center_dx = abs(_cx(pending) - _cx(anchor))
        candidates.append((vertical_gap, center_dx, anchor_pos, anchor_priority, anchor))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], _y1(item[4]), _x1(item[4])))
    _, _, anchor_pos, anchor_priority, anchor = candidates[0]
    return (anchor_pos, anchor_priority, anchor)


def _preferred_above_anchor(
    pending: "Block",
    ordered_entries: Sequence[tuple[float, int, "Block"]],
    *,
    allowed_types: Sequence[BlockType] | set[BlockType] | frozenset[BlockType],
    min_x_overlap: float,
) -> tuple[float, int, "Block"] | None:
    candidates: List[tuple[float, float, float, int, "Block"]] = []
    for anchor_pos, anchor_priority, anchor in ordered_entries:
        if anchor.block_type not in allowed_types:
            continue
        if _y2(anchor) > _y1(pending) + 4.0:
            continue
        if _projection_overlap_ratio_x(pending, anchor) < min_x_overlap:
            continue
        vertical_gap = max(0.0, _y1(pending) - _y2(anchor))
        center_dx = abs(_cx(pending) - _cx(anchor))
        candidates.append((vertical_gap, center_dx, -anchor_pos, anchor_priority, anchor))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], -_y2(item[4]), _x1(item[4])))
    _, _, neg_anchor_pos, anchor_priority, anchor = candidates[0]
    return (-neg_anchor_pos, anchor_priority, anchor)


def _preferred_spanning_parent_above(
    pending: "Block",
    ordered_entries: Sequence[tuple[float, int, "Block"]],
    *,
    image_width: int,
    image_height: Optional[int],
    cross_ids: set[int],
) -> tuple[float, int, "Block"] | None:
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max(_y2(pending), 1.0))
    if not _is_top_centered_attachment(
        pending,
        [entry[2] for entry in ordered_entries] + [pending],
        cross_ids=cross_ids,
        image_width=image_width,
        image_height=image_height,
    ):
        return None
    candidates: List[tuple[float, float, float, int, "Block"]] = []
    for anchor_pos, anchor_priority, anchor in ordered_entries:
        if anchor.block_type != BlockType.TITLE or id(anchor) not in cross_ids:
            continue
        if _w(anchor) < page_w * 0.45 or _y1(anchor) > page_h * 0.16:
            continue
        if _y2(anchor) > _y1(pending) + max(24.0, page_h * 0.025):
            continue
        vertical_gap = max(0.0, _y1(pending) - _y2(anchor))
        center_dx = abs(_cx(pending) - _cx(anchor))
        candidates.append((vertical_gap, center_dx, -_w(anchor), anchor_priority, anchor))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:3])
    _, _, _, anchor_priority, anchor = candidates[0]
    anchor_pos = next(pos for pos, _, blk in ordered_entries if blk is anchor)
    return (anchor_pos, anchor_priority, anchor)


def _latest_above_anchor(
    pending: "Block",
    ordered_entries: Sequence[tuple[float, int, "Block"]],
    *,
    allowed_types: Sequence[BlockType] | set[BlockType] | frozenset[BlockType],
) -> tuple[float, int, "Block"] | None:
    candidates: List[tuple[float, float, float, int, "Block"]] = []
    for anchor_pos, anchor_priority, anchor in ordered_entries:
        if anchor.block_type not in allowed_types:
            continue
        if _y2(anchor) > _y1(pending) + 4.0:
            continue
        vertical_gap = max(0.0, _y1(pending) - _y2(anchor))
        candidates.append((vertical_gap, -anchor_pos, -_y2(anchor), anchor_priority, anchor))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], _x1(item[4])))
    _, neg_anchor_pos, _, anchor_priority, anchor = candidates[0]
    return (-neg_anchor_pos, anchor_priority, anchor)


def _preferred_visual_anchor_for_caption(
    pending: "Block",
    ordered_entries: Sequence[tuple[float, int, "Block"]],
    *,
    image_width: int,
    image_height: Optional[int],
) -> tuple[float, int, "Block"] | None:
    page_scale = max(float(image_width), float(image_height or 0), 1.0)
    max_gap = max(18.0, page_scale * 0.04)
    candidates: List[tuple[float, float, float, int, "Block"]] = []
    for anchor_pos, anchor_priority, anchor in ordered_entries:
        if anchor.block_type not in _VISUAL_TYPES:
            continue
        x_overlap = _projection_overlap_ratio_x(pending, anchor)
        y_overlap = _projection_overlap_ratio_y(pending, anchor)
        edge_gap = _edge_gap(pending, anchor)
        if x_overlap < 0.12 and y_overlap < 0.40 and edge_gap > max_gap:
            continue
        vertical_gap = min(abs(_y1(pending) - _y2(anchor)), abs(_y2(pending) - _y1(anchor)))
        center_dx = abs(_cx(pending) - _cx(anchor))
        candidates.append((vertical_gap, center_dx, anchor_pos, anchor_priority, anchor))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], _y1(item[4]), _x1(item[4])))
    _, _, anchor_pos, anchor_priority, anchor = candidates[0]
    return (anchor_pos, anchor_priority, anchor)


def _spanning_visual_floor_anchor(
    pending: "Block",
    ordered_entries: Sequence[tuple[float, int, "Block"]],
    *,
    image_width: int,
    image_height: Optional[int],
    barrier_width_ratio: float,
) -> tuple[float, int, "Block"] | None:
    if pending.block_type not in _VISUAL_TYPES:
        return None
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), max(_y2(pending), 1.0))
    if _y1(pending) > page_h * 0.40:
        return None

    below = [
        (pos, pri, blk)
        for pos, pri, blk in ordered_entries
        if blk.block_type in _COLUMN_ANCHOR_TYPES
        and _y1(blk) >= _y2(pending) - 8.0
        and _projection_overlap_ratio_x(pending, blk) >= 0.18
    ]
    if not below:
        return None
    centers = sorted(_cx(blk) for _, _, blk in below)
    min_center_gap = max(40.0, page_w * 0.08)
    covered_column_groups = 1
    last_center = centers[0]
    for center in centers[1:]:
        if center - last_center >= min_center_gap:
            covered_column_groups += 1
            last_center = center
    if _w(pending) < page_w * barrier_width_ratio and covered_column_groups < 2:
        return None

    candidates: List[tuple[float, float, float, float, int, "Block"]] = []
    for pos, pri, blk in below:
        candidates.append((
            pos,
            max(0.0, _y1(blk) - _y2(pending)),
            -_projection_overlap_ratio_x(pending, blk),
            _x1(blk),
            pri,
            blk,
        ))
    candidates.sort(key=lambda item: item[:4])
    pos, _, _, _, pri, blk = candidates[0]
    return (pos, pri, blk)


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
            cross_ids=set(),
        )
        if best is None or dist < best[2]:
            best = (rank, anchor, dist)
    return best


def _restore_masked_elements(
    anchors: Sequence["Block"],
    masked: Sequence["Block"],
    *,
    cross_ids: set[int],
    image_width: int,
    image_height: Optional[int],
    barrier_width_ratio: float,
) -> List["Block"]:
    ordered_entries: List[tuple[float, int, "Block"]] = [
        (float(i), _priority(blk, cross_ids), blk) for i, blk in enumerate(anchors)
    ]
    if not ordered_entries:
        ordered = _sort_yx(masked)
        for i, blk in enumerate(ordered):
            _mark(blk, final_order=i, phase="fallback_all_masked")
        return ordered

    stage_order = [
        lambda b: b.block_type == BlockType.TITLE and id(b) in cross_ids,
        lambda b: b.block_type == BlockType.TITLE,
        lambda b: id(b) in cross_ids,
        lambda b: b.block_type in _VISUAL_TYPES,
        lambda b: True,
    ]

    for stage_idx, predicate in enumerate(stage_order):
        pending_blocks = [b for b in masked if predicate(b)]
        masked = [b for b in masked if b not in pending_blocks]
        stage_entries: List[tuple[float, int, "Block"]] = []
        for pending in pending_blocks:
            top_anchor_pos = min((anchor_pos for anchor_pos, _, _ in ordered_entries), default=0.0)
            top_anchor_y1 = min((_y1(anchor) for _, _, anchor in ordered_entries), default=float("inf"))
            is_top_spanning_text = (
                (_w(pending) >= max(float(image_width), 1.0) * barrier_width_ratio)
                and (_y2(pending) <= top_anchor_y1 + 12.0)
                and (
                    pending.block_type == BlockType.TITLE
                    or (
                        id(pending) in cross_ids
                        and pending.block_type not in _VISUAL_TYPES
                    )
                )
            )
            if is_top_spanning_text:
                _mark(
                    pending,
                    restore_anchor_id="__page_top__",
                    restore_distance=0.0,
                    restore_priority=_priority(pending, cross_ids),
                )
                stage_entries.append((top_anchor_pos - 1.0, _priority(pending, cross_ids), pending))
                continue
            if pending.block_type == BlockType.TITLE or (
                id(pending) in cross_ids and pending.block_type not in _VISUAL_TYPES
            ):
                search_entries = _semantic_candidate_entries(
                    pending,
                    ordered_entries,
                    image_width=image_width,
                    image_height=image_height,
                    cross_ids=cross_ids,
                    barrier_width_ratio=barrier_width_ratio,
                ) or list(ordered_entries)
            elif pending.block_type in _VISUAL_TYPES:
                preferred = _spanning_visual_floor_anchor(
                    pending,
                    ordered_entries,
                    image_width=image_width,
                    image_height=image_height,
                    barrier_width_ratio=barrier_width_ratio,
                )
                if preferred is None:
                    preferred = _preferred_below_anchor(
                        pending,
                        ordered_entries,
                        allowed_types=_TEXTLIKE_TYPES | _VISUAL_TYPES,
                        min_x_overlap=0.18,
                    )
                if preferred is None and _w(pending) >= max(float(image_width), 1.0) * barrier_width_ratio:
                    preferred = _latest_above_anchor(
                        pending,
                        ordered_entries,
                        allowed_types=_TEXTLIKE_TYPES | _VISUAL_TYPES,
                    )
                if preferred is None:
                    preferred = _preferred_above_anchor(
                        pending,
                        ordered_entries,
                        allowed_types=_TEXTLIKE_TYPES | _VISUAL_TYPES,
                        min_x_overlap=0.18,
                    )
                search_entries = [preferred] if preferred is not None else list(ordered_entries)
            elif pending.block_type in _CAPTION_TYPES:
                preferred = _preferred_visual_anchor_for_caption(
                    pending,
                    ordered_entries,
                    image_width=image_width,
                    image_height=image_height,
                )
                search_entries = [preferred] if preferred is not None else list(ordered_entries)
            else:
                preferred = _preferred_spanning_parent_above(
                    pending,
                    ordered_entries,
                    image_width=image_width,
                    image_height=image_height,
                    cross_ids=cross_ids,
                )
                search_entries = [preferred] if preferred is not None else list(ordered_entries)
            semantic_locked = (
                pending.block_type == BlockType.TITLE
                or _is_top_centered_attachment(
                    pending,
                    [entry[2] for entry in ordered_entries] + [pending],
                    cross_ids=cross_ids,
                    image_width=image_width,
                    image_height=image_height,
                )
            )
            if semantic_locked and search_entries:
                best_anchor_idx, _best_anchor_priority, best_anchor = search_entries[0]
                best_dist = _geometry_distance(
                    pending,
                    best_anchor,
                    image_width=image_width,
                    image_height=image_height,
                    cross_ids=cross_ids,
                )
                best_match = (best_anchor_idx, best_anchor, best_dist)
            else:
                best_match = _choose_best_anchor(
                    pending,
                    search_entries,
                    image_width=image_width,
                    image_height=image_height,
                    cross_ids=cross_ids,
                )
            if best_match is not None:
                best_anchor_idx, best_anchor, best_dist = best_match
            else:
                best_anchor_idx, best_anchor, best_dist = 0.0, None, float("inf")
            if best_anchor is not None:
                _mark(
                    pending,
                    restore_anchor_id=_block_id(best_anchor),
                    restore_distance=round(best_dist, 5),
                    restore_priority=_priority(pending, cross_ids),
                )
                if pending.block_type in _CAPTION_TYPES and best_anchor.block_type in _VISUAL_TYPES and _y1(pending) >= _y2(best_anchor):
                    position = best_anchor_idx + 0.05
                elif _is_top_centered_attachment(
                    pending,
                    [entry[2] for entry in ordered_entries] + [pending],
                    cross_ids=cross_ids,
                    image_width=image_width,
                    image_height=image_height,
                ):
                    position = best_anchor_idx + 0.05
                elif id(pending) in cross_ids and best_anchor.block_type == BlockType.TITLE:
                    position = best_anchor_idx + 0.05
                elif pending.block_type == BlockType.TITLE and _cy(pending) <= _cy(best_anchor):
                    position = best_anchor_idx - 0.05
                elif _y2(pending) <= _y1(best_anchor):
                    position = best_anchor_idx - 0.25
                elif _y1(pending) >= _y2(best_anchor):
                    position = best_anchor_idx + 0.25
                elif _x1(pending) < _x1(best_anchor):
                    position = best_anchor_idx - 0.05
                else:
                    position = best_anchor_idx + 0.05
            else:
                position = float(len(ordered_entries))
            stage_entries.append((position, _priority(pending, cross_ids), pending))

        ordered_entries.extend(stage_entries)
        ordered_entries.sort(key=lambda item: (item[0], -item[1], _y1(item[2]), _x1(item[2])))

    return [blk for _, _, blk in ordered_entries]


def _assign_single_column(blocks: Sequence["Block"]) -> None:
    for blk in blocks:
        blk.col_count = 1
        blk.col_index = 0
        blk.spanned_cols = [0]


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
        block_proto = (getattr(block, "attributes", None) or {}).get("xycutpp_proto", {})
        if block.block_type not in _TEXTLIKE_TYPES or len(cols) != 1:
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
    for block in prefix:
        first_idx = min(first_idx, seq.index(block))
        last_idx = max(last_idx, seq.index(block))
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

    prefix = list(_select_region_prefix_blocks(seq, region=region, page_w=page_w, page_h=page_h))
    placement = _region_placement(seq, region=region, prefix=prefix)

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

    for block in seq[placement.first_idx:placement.last_idx + 1]:
        if id(block) in group_ids:
            continue
        if _intersects_region(block, (0.0, band_top, page_w, band_bottom)):
            detect_spanned_blocks([block], col_bounds)
            block.col_count = len(columns)
            _mark(block, region_id=region.region_id, region_kind=region.region_kind, region_role="attached")
        elif _w(block) >= page_w * 0.60:
            block.col_count = 1
            block.col_index = 0
            block.spanned_cols = [0]

    middle = [
        block for block in seq[placement.first_idx:placement.last_idx + 1]
        if id(block) not in group_ids and block not in prefix
    ]
    return seq[:placement.first_idx] + prefix + reordered + middle + seq[placement.last_idx + 1:]


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
    seq = _enforce_column_local_visual_neighbors(
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
    seq = _enforce_lower_section_wraparound_columns(
        seq,
        image_width=image_width,
        image_height=image_height,
    )
    return seq


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
    if _has_single_column_wide_body_dominance(
        ordered,
        image_width=image_width,
        image_height=image_height,
    ):
        _assign_single_column(ordered)
        return

    candidates = [
        blk for blk in ordered
        if blk.block_type in _COLUMN_ANCHOR_TYPES
        and _w(blk) <= page_w * 0.60
        and _marked(blk, "mask_reason") != "top_attachment"
    ]
    if len(candidates) < 2:
        candidates = [
            blk for blk in ordered
            if blk.block_type not in _COLUMN_EXCLUDED_TYPES
            and blk.block_type != BlockType.TITLE
            and _w(blk) <= page_w * 0.60
            and _marked(blk, "mask_reason") != "top_attachment"
        ]
    if len(candidates) < 2:
        candidates = [
            blk for blk in ordered
            if blk.block_type not in _COLUMN_EXCLUDED_TYPES
            and _w(blk) <= page_w * 0.60
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

    if len(columns) <= 1 and any(blk.block_type == BlockType.TITLE for blk in candidates):
        body_candidates = [
            blk for blk in candidates
            if blk.block_type != BlockType.TITLE
        ]
        if len(body_candidates) >= 2:
            columns, col_bounds = detect_columns(
                body_candidates,
                image_width,
                max_cols=max_cols,
                cluster_thresh=cluster_thresh,
            )

    if len(columns) <= 1:
        _assign_single_column(ordered)
        return

    candidate_ids = {id(blk) for col in columns for blk in col}
    for blk in ordered:
        blk.col_count = 0
        blk.col_index = 0
        blk.spanned_cols = []

    for col_idx, members in enumerate(columns):
        for blk in members:
            blk.col_count = len(columns)
            blk.col_index = col_idx
            blk.spanned_cols = [col_idx]

    unassigned = [blk for blk in ordered if id(blk) not in candidate_ids]
    if unassigned:
        detect_spanned_blocks(unassigned, col_bounds)
        for blk in unassigned:
            blk.col_count = len(columns)
            if _marked(blk, "mask_reason") == "top_attachment":
                blk.col_count = 1
                blk.col_index = 0
                blk.spanned_cols = [0]

    for blk in ordered:
        if _marked(blk, "mask_reason") != "top_attachment":
            blk.col_count = len(columns)

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

    Legacy parameters are kept for call-site compatibility even if not used by
    the underlying prototype.
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
    active, masked, precut_targets = _split_mask_sets(
        valid,
        cross_ids=cross_ids,
        image_width=image_width,
        image_height=image_height,
        near_text_margin_ratio=near_text_margin_ratio,
    )

    anchors = _sort_active_anchors(
        active,
        masked,
        precut_targets,
        image_width=image_width,
        image_height=image_height,
        cross_ids=cross_ids,
        density_threshold=density_threshold,
        min_gap_px=min_gap_px,
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
