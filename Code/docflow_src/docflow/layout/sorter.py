"""版面区块的顶层阅读顺序排序。

算法分三个阶段进行：

1. **区域拆分** -- 将页面划分为水平区域（zone）。
2. **候选分栏** -- 在每个区域内按 x 方向聚类得到候选列。
3. **置信回退** -- 若列结构置信不足则回退为单栏，避免错分栏。
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import re
from typing import List, TYPE_CHECKING, Tuple

from docflow.layout.zone_splitter import split_into_zones
from docflow.layout.column_detector import detect_columns, detect_spanned_blocks
from docflow.layout.xycutpp import postprocess_xycutpp_local_attachments, sort_layout_xycutpp
from docflow.model.base import BlockType
from docflow.utils.constants import (
    SPAN_ELIGIBLE_TYPES,
    MAX_COLS,
    COLUMN_CLUSTER_THRESH,
)

if TYPE_CHECKING:
    from docflow.model.base import Block
    from docflow.model.blocks.text_block import TextBlock


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
_ACADEMIC_SECTION_TITLE_RE = re.compile(
    r"^\s*(?:"
    r"[IVXLCM]+\."                        # III. CONCLUSION
    r"|[A-Z]\."                           # A. Dataset ...
    r"|\d+(?:\.\d+)*"                    # 2 / 2.1 / 3.1.4
    r")\s+\S",
    re.IGNORECASE,
)
_ACADEMIC_CUE_TYPES = frozenset({
    BlockType.FORMULA,
    BlockType.EQUATION,
    BlockType.TABLE,
    BlockType.TABLE_CAPTION,
    BlockType.FIGURE_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.FORMULA_CAPTION,
})
_SPANNING_CAPTION_TYPES = frozenset({
    BlockType.FIGURE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.FORMULA_CAPTION,
})


@dataclass
class _FlowSeed:
    flow_id: str
    kind: str
    col_ids: tuple[int, ...]
    anchor_y: float
    anchor_x: float
    title_priority: int


@dataclass
class _LayoutEvidence:
    col_count: int
    text_col_count: int
    stable_multicol: bool
    has_top_spanning_anchor: bool
    has_lower_wide_anchor: bool
    has_spanning_visual: bool
    has_spanning_caption: bool
    has_peripheral_sidebar: bool
    spanning_band_count: int
    centered_short_title_count: int


def _flow_id_of(block: "Block") -> str:
    attrs = getattr(block, "attributes", None) or {}
    return str(attrs.get("flow_id", ""))


def _set_flow_meta(block: "Block", flow_id: str, flow_kind: str) -> None:
    if block.attributes is None:
        block.attributes = {}
    block.attributes["flow_id"] = flow_id
    block.attributes["flow_kind"] = flow_kind


def _clear_flow_meta(blocks: List["Block"]) -> None:
    for block in blocks:
        attrs = getattr(block, "attributes", None)
        if not attrs:
            continue
        attrs.pop("flow_id", None)
        attrs.pop("flow_kind", None)


def _clone_blocks_for_evidence(blocks: List["Block"]) -> List["Block"]:
    return [copy.copy(block) for block in blocks]


def _block_cols(block: "Block") -> tuple[int, ...]:
    cols = getattr(block, "spanned_cols", None) or [getattr(block, "col_index", 0)]
    normalized = sorted({int(v) for v in cols})
    return tuple(normalized or [0])


def _is_textlike_block(block: "Block") -> bool:
    return block.block_type in {
        BlockType.TEXT,
        BlockType.TITLE,
        BlockType.REFERENCE,
        BlockType.ABSTRACT,
        BlockType.FIGURE_CAPTION,
        BlockType.TABLE_CAPTION,
        BlockType.TABLE_FOOTNOTE,
        BlockType.FORMULA_CAPTION,
        BlockType.CODE,
        BlockType.LIST,
        BlockType.FOOTNOTE,
    }


def _block_text(block: "Block") -> str:
    if not _is_textlike_block(block) or not hasattr(block, "full_text"):
        return ""
    try:
        return str(block.full_text() or "")
    except Exception:
        return ""


def _line_count(block: "Block") -> int:
    if hasattr(block, "count_lines"):
        try:
            return max(1, int(block.count_lines()))
        except Exception:
            return 1
    return 1


def _looks_like_academic_sectioned_page(blocks: List["Block"], image_width: int) -> bool:
    title_blocks = [blk for blk in blocks if blk.block_type == BlockType.TITLE]
    if len(title_blocks) < 3:
        return False

    page_w = max(float(image_width), 1.0)
    section_like = 0
    lower_section_like = 0
    narrow_titles = 0

    for blk in title_blocks:
        text = _block_text(blk).strip()
        if not text:
            continue
        is_narrow = float(blk.bbox.width) <= page_w * 0.62
        if is_narrow:
            narrow_titles += 1
        if not _ACADEMIC_SECTION_TITLE_RE.match(text):
            continue
        section_like += 1
        if float(blk.bbox.y1) >= 240.0:
            lower_section_like += 1

    if section_like < 3:
        return False
    if narrow_titles < max(2, len(title_blocks) // 2):
        return False
    return lower_section_like >= 2


def _has_stable_two_column_text_distribution(blocks: List["Block"], image_width: int) -> bool:
    page_w = max(float(image_width), 1.0)
    left = 0
    right = 0
    for blk in blocks:
        if blk.block_type in SPAN_ELIGIBLE_TYPES:
            continue
        if float(blk.bbox.width) >= page_w * 0.72:
            continue
        cx = (float(blk.bbox.x1) + float(blk.bbox.x2)) * 0.5
        if cx <= page_w * 0.46:
            left += 1
        elif cx >= page_w * 0.54:
            right += 1
    return left >= 3 and right >= 3


def _should_force_legacy_for_academic_page(blocks: List["Block"], image_width: int) -> bool:
    if not _has_stable_two_column_text_distribution(blocks, image_width):
        return False
    if _looks_like_academic_sectioned_page(blocks, image_width):
        return True

    cue_count = sum(1 for blk in blocks if blk.block_type in _ACADEMIC_CUE_TYPES)
    title_count = sum(1 for blk in blocks if blk.block_type == BlockType.TITLE)
    return cue_count >= 5 and title_count <= 4


def _looks_like_stable_multicol_spanning_page(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
) -> bool:
    evidence = _collect_layout_evidence(
        blocks,
        image_width=image_width,
        image_height=image_height,
    )
    return _has_stable_multicol_spanning_evidence(evidence)


def _has_stable_multicolumn_layout(blocks: List["Block"], image_width: int) -> bool:
    textish = [
        blk for blk in blocks
        if blk.block_type not in SPAN_ELIGIBLE_TYPES and blk.block_type not in _STRIP_TYPES
    ]
    if len(textish) < 6:
        return False
    col_count, _ = _global_assign_columns(
        list(blocks),
        image_width=image_width,
        max_cols=4,
        cluster_thresh=COLUMN_CLUSTER_THRESH,
    )
    return col_count >= 2


def _looks_like_peripheral_sidebar_layout(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
) -> bool:
    return _collect_layout_evidence(
        blocks,
        image_width=image_width,
        image_height=image_height,
    ).has_peripheral_sidebar


def _has_strong_multiflow_evidence(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
    evidence: _LayoutEvidence | None = None,
) -> bool:
    if not blocks or not image_height or image_height <= 0:
        return False

    evidence = evidence or _collect_layout_evidence(
        blocks,
        image_width=image_width,
        image_height=image_height,
    )
    if _looks_like_academic_sectioned_page(blocks, image_width):
        return False
    if evidence.has_peripheral_sidebar:
        return True
    if not evidence.has_lower_wide_anchor:
        return False
    if not evidence.has_spanning_caption:
        return False

    top_guard = float(image_height) * 0.16
    text_blocks = [blk for blk in blocks if blk.block_type == BlockType.TEXT]
    title_blocks = [blk for blk in blocks if blk.block_type == BlockType.TITLE and float(blk.bbox.y1) > top_guard]
    if not title_blocks:
        return False

    for title in title_blocks:
        wide_title = (
            float(title.bbox.width) >= float(image_width) * 0.30
            or len(getattr(title, "spanned_cols", []) or [getattr(title, "col_index", 0)]) > 1
        )
        if not wide_title:
            continue

        text_above = [
            blk for blk in text_blocks
            if float(blk.bbox.y2) <= float(title.bbox.y1)
            and float(title.bbox.y1) - float(blk.bbox.y2) <= max(float(image_height) * 0.22, 900.0)
        ]
        cols_above = {
            getattr(blk, "col_index", 0)
            for blk in text_above
            if float(blk.bbox.width) <= float(image_width) * 0.40
        }
        text_below = [
            blk for blk in text_blocks
            if float(blk.bbox.y1) >= float(title.bbox.y2)
            and float(blk.bbox.y1) - float(title.bbox.y2) <= max(float(image_height) * 0.25, 1100.0)
        ]
        kicker_above = any(
            _line_count(blk) <= 2
            and len(_block_text(blk).strip()) <= 120
            and 0.0 <= float(title.bbox.y1) - float(blk.bbox.y2) <= 220.0
            and max(
                0.0,
                min(float(title.bbox.x2), float(blk.bbox.x2)) - max(float(title.bbox.x1), float(blk.bbox.x1)),
            ) >= min(float(title.bbox.width), float(blk.bbox.width)) * 0.20
            for blk in text_above
        )

        if len(cols_above) >= 2 and len(text_below) >= 2:
            return True
        if kicker_above and len(text_above) >= 2 and len(text_below) >= 2:
            return True
    return False


def _head_block_sort_key(block: "Block") -> tuple[int, float, float]:
    is_visual = block.block_type in SPAN_ELIGIBLE_TYPES
    if block.block_type == BlockType.TITLE:
        priority = 0
    elif not is_visual:
        priority = 1
    else:
        priority = 2
    return priority, float(block.bbox.y1), float(block.bbox.x1)


def _sort_generic_multicolumn_column_major(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
    max_cols: int,
    cluster_thresh: float,
) -> List["Block"]:
    if not blocks:
        return []

    _global_assign_columns(
        blocks,
        image_width=image_width,
        max_cols=max_cols,
        cluster_thresh=cluster_thresh,
    )
    col_count = max((int(getattr(blk, "col_count", 1) or 1) for blk in blocks), default=1)
    if col_count <= 1:
        return _sort_single_column_blocks(blocks)

    single_col_blocks = [
        blk for blk in blocks
        if len(getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]) == 1
    ]
    if not single_col_blocks:
        return sorted(blocks, key=lambda b: (b.bbox.y1, b.bbox.x1))

    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), 1.0)
    widths = sorted(float(blk.bbox.width) for blk in single_col_blocks)
    median_width = widths[len(widths) // 2] if widths else page_w / max(col_count, 1)
    wide_thresh = max(median_width * 1.25, page_w * 0.42)

    body_text = [blk for blk in single_col_blocks if blk.block_type not in {BlockType.TITLE}]
    first_body_y = min((float(blk.bbox.y1) for blk in body_text), default=float("inf"))
    last_body_y = max((float(blk.bbox.y2) for blk in body_text), default=0.0)
    top_band_limit = min(first_body_y + 120.0, page_h * 0.18) if first_body_y != float("inf") else page_h * 0.18
    bottom_band_start = max(last_body_y - 48.0, page_h * 0.82)
    lead_titles = [
        blk for blk in blocks
        if blk.block_type == BlockType.TITLE and float(blk.bbox.y1) <= top_band_limit + max(220.0, page_h * 0.08)
    ]
    lead_title_cols = {col for blk in lead_titles for col in _block_cols(blk)}
    lead_title_bottom = max((float(blk.bbox.y2) for blk in lead_titles), default=float("-inf"))

    head_blocks: List["Block"] = []
    column_blocks: dict[int, List["Block"]] = {}
    tail_blocks: List["Block"] = []

    for blk in sorted(blocks, key=lambda b: (b.bbox.y1, b.bbox.x1)):
        cols = getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]
        is_spanned = len(cols) > 1 or float(blk.bbox.width) >= wide_thresh
        is_caption = blk.block_type in _SPANNING_CAPTION_TYPES
        is_visual = blk.block_type in SPAN_ELIGIBLE_TYPES
        is_topish = float(blk.bbox.y1) <= top_band_limit
        is_bottomish = float(blk.bbox.y1) >= bottom_band_start
        is_title_attached_visual = (
            is_visual
            and bool(lead_titles)
            and bool(set(cols).intersection(lead_title_cols))
            and float(blk.bbox.y2) <= lead_title_bottom + max(160.0, page_h * 0.06)
        )
        is_short_text = (
            blk.block_type == BlockType.TEXT
            and _line_count(blk) <= 2
            and len(_block_text(blk).strip()) <= 100
            and float(blk.bbox.y1) <= page_h * 0.14
        )

        if is_title_attached_visual:
            head_blocks.append(blk)
            continue
        if is_spanned and is_topish:
            head_blocks.append(blk)
            continue
        if is_short_text and not is_visual and not is_caption:
            head_blocks.append(blk)
            continue
        if is_spanned and (is_visual or is_caption or is_bottomish):
            tail_blocks.append(blk)
            continue

        col_idx = int(getattr(blk, "col_index", 0))
        column_blocks.setdefault(col_idx, []).append(blk)

    ordered: List["Block"] = []
    ordered.extend(sorted(head_blocks, key=_head_block_sort_key))
    for col_idx in sorted(column_blocks.keys()):
        ordered.extend(sorted(column_blocks[col_idx], key=lambda b: (b.bbox.y1, b.bbox.x1)))
    ordered.extend(sorted(tail_blocks, key=lambda b: (b.bbox.y1, b.bbox.x1)))
    return ordered


def _sort_stable_multicol_column_major(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
    max_cols: int,
    cluster_thresh: float,
) -> List["Block"]:
    if not blocks:
        return []

    _global_assign_columns(
        blocks,
        image_width=image_width,
        max_cols=max_cols,
        cluster_thresh=cluster_thresh,
    )
    col_count = max((int(getattr(blk, "col_count", 1) or 1) for blk in blocks), default=1)
    if col_count <= 1:
        return _sort_single_column_blocks(blocks)

    single_col_blocks = [
        blk for blk in blocks
        if len(getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]) == 1
    ]
    if not single_col_blocks:
        return sorted(blocks, key=lambda b: (b.bbox.y1, b.bbox.x1))

    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), 1.0)
    widths = sorted(float(blk.bbox.width) for blk in single_col_blocks)
    median_width = widths[len(widths) // 2] if widths else page_w / max(col_count, 1)
    wide_thresh = max(median_width * 1.35, page_w * 0.45)

    body_text = [
        blk for blk in single_col_blocks
        if blk.block_type not in {BlockType.TITLE}
    ]
    first_body_y = min((float(blk.bbox.y1) for blk in body_text), default=float("inf"))
    top_band_limit = min(first_body_y + 120.0, page_h * 0.18) if first_body_y != float("inf") else page_h * 0.18
    lead_titles = [
        blk for blk in blocks
        if blk.block_type == BlockType.TITLE and float(blk.bbox.y1) <= top_band_limit + max(220.0, page_h * 0.08)
    ]
    lead_title_cols = {col for blk in lead_titles for col in _block_cols(blk)}
    lead_title_bottom = max((float(blk.bbox.y2) for blk in lead_titles), default=float("-inf"))

    head_blocks: List["Block"] = []
    column_blocks: dict[int, List["Block"]] = {}
    tail_blocks: List["Block"] = []

    for blk in sorted(blocks, key=lambda b: (b.bbox.y1, b.bbox.x1)):
        cols = getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]
        is_spanned = len(cols) > 1 or float(blk.bbox.width) >= wide_thresh
        is_caption = blk.block_type in _SPANNING_CAPTION_TYPES
        is_visual = blk.block_type in SPAN_ELIGIBLE_TYPES
        is_topish = float(blk.bbox.y1) <= top_band_limit
        is_title_attached_visual = (
            is_visual
            and bool(lead_titles)
            and bool(set(cols).intersection(lead_title_cols))
            and float(blk.bbox.y2) <= lead_title_bottom + max(160.0, page_h * 0.06)
        )
        is_short_text = (
            blk.block_type == BlockType.TEXT
            and _line_count(blk) <= 2
            and len(_block_text(blk).strip()) <= 80
            and float(blk.bbox.y1) <= page_h * 0.14
        )

        if is_title_attached_visual:
            head_blocks.append(blk)
            continue
        if is_spanned and is_topish:
            head_blocks.append(blk)
            continue
        if is_short_text and not is_visual and not is_caption:
            head_blocks.append(blk)
            continue
        if is_spanned and (is_visual or is_caption):
            tail_blocks.append(blk)
            continue

        col_idx = int(getattr(blk, "col_index", 0))
        column_blocks.setdefault(col_idx, []).append(blk)

    ordered: List["Block"] = []
    ordered.extend(sorted(head_blocks, key=_head_block_sort_key))
    for col_idx in sorted(column_blocks.keys()):
        ordered.extend(sorted(column_blocks[col_idx], key=lambda b: (b.bbox.y1, b.bbox.x1)))
    ordered.extend(sorted(tail_blocks, key=lambda b: (b.bbox.y1, b.bbox.x1)))
    return ordered


def _sort_academic_column_major(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
    max_cols: int,
    cluster_thresh: float,
) -> List["Block"]:
    if not blocks:
        return []

    _global_assign_columns(
        blocks,
        image_width=image_width,
        max_cols=max_cols,
        cluster_thresh=cluster_thresh,
    )
    col_count = max((int(getattr(blk, "col_count", 1) or 1) for blk in blocks), default=1)
    if col_count <= 1:
        return _sort_single_column_blocks(blocks)

    single_col_blocks = [
        blk for blk in blocks
        if len(getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]) == 1
    ]
    if not single_col_blocks:
        return sorted(blocks, key=lambda b: (b.bbox.y1, b.bbox.x1))

    min_body_y = min(float(blk.bbox.y1) for blk in single_col_blocks)
    max_body_y = max(float(blk.bbox.y2) for blk in single_col_blocks)

    head_blocks: List["Block"] = []
    tail_blocks: List["Block"] = []
    middle_blocks: List["Block"] = []
    by_col: dict[int, List["Block"]] = {}

    for blk in blocks:
        cols = getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]
        is_spanned = len(cols) > 1
        if not is_spanned:
            by_col.setdefault(int(cols[0]), []).append(blk)
            continue
        if float(blk.bbox.y2) <= min_body_y + 48.0:
            head_blocks.append(blk)
        elif float(blk.bbox.y1) >= max_body_y - 48.0:
            tail_blocks.append(blk)
        else:
            middle_blocks.append(blk)

    ordered: List["Block"] = []
    ordered.extend(sorted(head_blocks, key=lambda b: (b.bbox.y1, b.bbox.x1)))
    for col_idx in sorted(by_col.keys()):
        ordered.extend(sorted(by_col[col_idx], key=lambda b: (b.bbox.y1, b.bbox.x1)))
    ordered.extend(sorted(middle_blocks, key=lambda b: (b.bbox.y1, b.bbox.x1)))
    ordered.extend(sorted(tail_blocks, key=lambda b: (b.bbox.y1, b.bbox.x1)))
    return ordered


def _looks_like_sentence_continuation(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return False
    starts_lower = normalized[:1].islower()
    starts_punct = normalized[:1] in {",", ".", ";", ":", ")", "]", "”", '"', "'"}
    starts_hyphen_tail = normalized[:1] in {"-", "—"}
    common_tail = normalized[:16].lower().startswith((
        "and ", "or ", "but ", "if ", "the ", "a ", "an ",
    ))
    return starts_lower or starts_punct or starts_hyphen_tail or common_tail


def _is_candidate_title_anchor(
    block: "Block",
    image_width: int,
    image_height: int | None,
) -> bool:
    if block.block_type != BlockType.TITLE:
        return False
    if image_height and image_height > 0 and float(block.bbox.y1) <= float(image_height) * 0.14:
        cols = _block_cols(block)
        wide_enough = (
            float(block.bbox.width) >= max(float(image_width) * 0.42, 240.0)
            or len(cols) > 1
        )
        if not wide_enough:
            return False
    return True


def _is_candidate_kicker_anchor(
    block: "Block",
    image_width: int,
    image_height: int | None,
    col_width: float,
) -> bool:
    if block.block_type != BlockType.TEXT:
        return False
    if image_height and image_height > 0 and float(block.bbox.y1) <= float(image_height) * 0.14:
        return False
    if _line_count(block) > 2:
        return False
    text = _block_text(block).strip()
    if not text or len(text) > 140:
        return False
    if float(block.bbox.width) < max(col_width * 1.15, float(image_width) * 0.12):
        return False
    if float(block.bbox.width) > max(col_width * 1.9, float(image_width) * 0.45):
        return False
    if float(block.bbox.height) > max(110.0, float(image_width) * 0.03):
        return False
    if _looks_like_sentence_continuation(text):
        return False
    # 像 magazine 的 section kicker，通常很短且位于独立标题带中。
    return True


def _is_attached_to_title_seed(
    block: "Block",
    title_seeds: List[_FlowSeed],
    *,
    max_above_gap: float = 260.0,
    max_below_gap: float = 800.0,
) -> bool:
    cols = set(_block_cols(block))
    for seed in title_seeds:
        if not cols.intersection(seed.col_ids):
            continue
        if 0.0 <= seed.anchor_y - float(block.bbox.y2) <= max_above_gap:
            return True
        if 0.0 <= float(block.bbox.y1) - seed.anchor_y <= max_below_gap:
            return True
    return False


def _global_assign_columns(
    blocks: List["Block"],
    image_width: int,
    max_cols: int,
    cluster_thresh: float,
) -> tuple[int, List[Tuple[float, float]]]:
    textish = [blk for blk in blocks if blk.block_type not in SPAN_ELIGIBLE_TYPES and blk.block_type not in _STRIP_TYPES]
    bodyish = [
        blk for blk in textish
        if blk.block_type != BlockType.TITLE
        and float(blk.bbox.width) <= float(image_width) * 0.33
    ]
    candidate_source = bodyish if len(bodyish) >= 2 else (textish if len(textish) >= 2 else list(blocks))
    effective_thresh = min(cluster_thresh, 0.07)
    columns, col_bounds = detect_columns(
        candidate_source,
        image_width,
        max_cols=max_cols,
        cluster_thresh=effective_thresh,
    )
    col_count = len(columns)
    if col_count <= 1:
        _assign_single_column(blocks)
        return 1, [(0.0, float(image_width))]

    for col_idx, col_members in enumerate(columns):
        for blk in col_members:
            blk.col_count = col_count
            blk.col_index = col_idx
            blk.spanned_cols = [col_idx]

    unassigned = [blk for blk in blocks if getattr(blk, "col_count", 0) != col_count]
    if unassigned:
        detect_spanned_blocks(unassigned, col_bounds)
    for blk in blocks:
        blk.col_count = col_count
    return col_count, col_bounds


def _has_top_spanning_anchor(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
) -> bool:
    if not blocks:
        return False
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), 1.0)
    top_limit = page_h * 0.18 if image_height else 220.0
    for blk in blocks:
        if blk.block_type != BlockType.TITLE:
            continue
        cols = _block_cols(blk)
        if float(blk.bbox.y1) > top_limit:
            continue
        if float(blk.bbox.width) >= page_w * 0.55 or len(cols) > 1:
            return True
    return False


def _has_lower_wide_anchor(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
) -> bool:
    if not blocks or not image_height or image_height <= 0:
        return False
    page_w = max(float(image_width), 1.0)
    top_guard = float(image_height) * 0.16
    for blk in blocks:
        if blk.block_type != BlockType.TITLE:
            continue
        cols = _block_cols(blk)
        if float(blk.bbox.y1) <= top_guard:
            continue
        if float(blk.bbox.width) >= page_w * 0.30 or len(cols) > 1:
            return True
    return False


def _has_stable_multicol_spanning_evidence(evidence: _LayoutEvidence) -> bool:
    if not evidence.stable_multicol:
        return False
    if evidence.has_peripheral_sidebar:
        return False
    if evidence.text_col_count < 3:
        return False
    return (
        evidence.has_top_spanning_anchor
        and evidence.has_spanning_visual
        and (evidence.has_spanning_caption or evidence.has_lower_wide_anchor)
    )


def _has_banded_mixed_layout_evidence(evidence: _LayoutEvidence) -> bool:
    if evidence.has_peripheral_sidebar:
        return False
    if evidence.has_top_spanning_anchor:
        return False
    if evidence.text_col_count > 2:
        return False
    return evidence.spanning_band_count >= 3 and evidence.centered_short_title_count >= 2


def _collect_layout_evidence(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
    *,
    max_cols: int = 4,
    cluster_thresh: float = COLUMN_CLUSTER_THRESH,
) -> _LayoutEvidence:
    if not blocks:
        return _LayoutEvidence(1, 0, False, False, False, False, False, False, 0, 0)

    col_count, col_bounds = _global_assign_columns(
        list(blocks),
        image_width=image_width,
        max_cols=max_cols,
        cluster_thresh=cluster_thresh,
    )
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), 1.0)
    top_guard = page_h * 0.16
    textlike_blocks = [
        blk
        for blk in blocks
        if blk.block_type not in SPAN_ELIGIBLE_TYPES
        and blk.block_type not in _STRIP_TYPES
        and len(getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]) == 1
    ]
    stable_multicol = col_count >= 2 and len(textlike_blocks) >= 6
    has_top_spanning_anchor = _has_top_spanning_anchor(blocks, image_width, image_height)
    has_lower_wide_anchor = _has_lower_wide_anchor(blocks, image_width, image_height)
    has_spanning_visual = any(
        blk.block_type in SPAN_ELIGIBLE_TYPES and len(_block_cols(blk)) > 1
        for blk in blocks
    )
    has_spanning_caption = any(
        blk.block_type in _SPANNING_CAPTION_TYPES and len(_block_cols(blk)) > 1
        for blk in blocks
    )
    spanning_band_count = sum(
        1
        for blk in blocks
        if blk.block_type not in SPAN_ELIGIBLE_TYPES
        and (
            float(blk.bbox.width) >= page_w * 0.55
            or len(_block_cols(blk)) > 1
        )
    )
    centered_short_title_count = sum(
        1
        for blk in blocks
        if blk.block_type == BlockType.TITLE
        and float(blk.bbox.width) <= page_w * 0.28
        and abs(((float(blk.bbox.x1) + float(blk.bbox.x2)) * 0.5) - page_w * 0.5) <= page_w * 0.16
    )
    if len(textlike_blocks) < 6:
        return _LayoutEvidence(
            col_count,
            0,
            stable_multicol,
            has_top_spanning_anchor,
            has_lower_wide_anchor,
            has_spanning_visual,
            has_spanning_caption,
            False,
            spanning_band_count,
            centered_short_title_count,
        )

    by_col: dict[int, List["Block"]] = {}
    for blk in textlike_blocks:
        by_col.setdefault(int(getattr(blk, "col_index", 0)), []).append(blk)
    text_col_count = len(by_col)

    middle_cols = [idx for idx in range(1, col_count - 1)]
    if not middle_cols:
        return _LayoutEvidence(
            col_count,
            text_col_count,
            stable_multicol,
            has_top_spanning_anchor,
            has_lower_wide_anchor,
            has_spanning_visual,
            has_spanning_caption,
            False,
            spanning_band_count,
            centered_short_title_count,
        )

    middle_blocks = [blk for col_id in middle_cols for blk in by_col.get(col_id, [])]
    if len(middle_blocks) < 4:
        return _LayoutEvidence(
            col_count,
            text_col_count,
            stable_multicol,
            has_top_spanning_anchor,
            has_lower_wide_anchor,
            has_spanning_visual,
            has_spanning_caption,
            False,
            spanning_band_count,
            centered_short_title_count,
        )

    central_titles = sum(
        1
        for blk in blocks
        if blk.block_type == BlockType.TITLE
        and float(blk.bbox.y1) > top_guard
        and set(_block_cols(blk)).intersection(middle_cols)
    )
    middle_widths = sorted(float(blk.bbox.width) for blk in middle_blocks)
    middle_width_ref = middle_widths[len(middle_widths) // 2] if middle_widths else page_w / max(col_count, 1)
    has_peripheral_sidebar = False

    for edge_col in (0, col_count - 1):
        members = by_col.get(edge_col, [])
        if len(members) < 2 or central_titles < 1:
            continue

        widths = sorted(float(blk.bbox.width) for blk in members)
        median_width = widths[len(widths) // 2] if widths else 0.0
        edge_center = (float(col_bounds[edge_col][0]) + float(col_bounds[edge_col][1])) * 0.5
        near_edge = edge_center <= page_w * 0.28 or edge_center >= page_w * 0.72
        narrow_enough = median_width <= min(page_w * 0.32, middle_width_ref * 0.86)
        coverage = _y_coverage_ratio(members, 0.0, page_h)
        local_titles = sum(1 for blk in members if blk.block_type == BlockType.TITLE)
        local_short_blocks = sum(
            1
            for blk in members
            if _line_count(blk) <= 3 and len(_block_text(blk).strip()) <= 180
        )
        if not (near_edge and narrow_enough and coverage >= 0.16):
            continue
        if local_titles + local_short_blocks < 2:
            continue

        adjacent_col = 1 if edge_col == 0 else col_count - 2
        gap = max(0.0, float(col_bounds[adjacent_col][0]) - float(col_bounds[edge_col][1]))
        if edge_col == col_count - 1:
            gap = max(0.0, float(col_bounds[edge_col][0]) - float(col_bounds[adjacent_col][1]))
        if gap < page_w * 0.035:
            continue
        has_peripheral_sidebar = True
        break

    return _LayoutEvidence(
        col_count,
        text_col_count,
        stable_multicol,
        has_top_spanning_anchor,
        has_lower_wide_anchor,
        has_spanning_visual,
        has_spanning_caption,
        has_peripheral_sidebar,
        spanning_band_count,
        centered_short_title_count,
    )


def _build_top_continuation_seeds(
    blocks: List["Block"],
    first_anchor_y: float,
    image_height: int | None,
    title_seeds: List[_FlowSeed],
) -> List[_FlowSeed]:
    by_col: dict[int, List["Block"]] = {}
    for blk in blocks:
        if float(blk.bbox.y1) >= first_anchor_y:
            continue
        if blk.block_type in _STRIP_TYPES:
            continue
        if blk.block_type in SPAN_ELIGIBLE_TYPES and _is_attached_to_title_seed(
            blk,
            title_seeds,
            max_above_gap=max(float(image_height or 0) * 0.24, 720.0),
            max_below_gap=0.0,
        ):
            continue
        if _is_candidate_kicker_anchor(
            blk,
            image_width=max(int(blk.bbox.x2), 1),
            image_height=image_height,
            col_width=max(float(blk.bbox.width), 1.0),
        ) and _is_attached_to_title_seed(blk, title_seeds, max_above_gap=320.0, max_below_gap=0.0):
            continue
        for col_id in _block_cols(blk):
            by_col.setdefault(col_id, []).append(blk)

    if not by_col:
        return []

    col_stats: List[tuple[int, float, float, float]] = []
    for col_id, members in sorted(by_col.items()):
        top = min(float(b.bbox.y1) for b in members)
        bottom = max(float(b.bbox.y2) for b in members)
        left = min(float(b.bbox.x1) for b in members)
        col_stats.append((col_id, top, bottom, left))

    seeds: List[_FlowSeed] = []
    current_cols: List[int] = [col_stats[0][0]]
    current_top = col_stats[0][1]
    current_bottom = col_stats[0][2]
    current_left = col_stats[0][3]
    threshold = max(140.0, float(image_height or 0) * 0.04)

    for col_id, top, bottom, left in col_stats[1:]:
        current_h = max(current_bottom - current_top, 1.0)
        other_h = max(bottom - top, 1.0)
        overlap = min(current_bottom, bottom) - max(current_top, top)
        overlap_ratio = overlap / min(current_h, other_h)
        adjacent_col = col_id <= max(current_cols) + 1
        similar_bottom = abs(bottom - current_bottom) <= threshold
        current_crosses_anchor = current_bottom > first_anchor_y + 220.0
        next_crosses_anchor = bottom > first_anchor_y + 220.0
        crosses_anchor_consistent = current_crosses_anchor == next_crosses_anchor
        if adjacent_col and crosses_anchor_consistent and (similar_bottom or overlap_ratio >= 0.72):
            current_cols.append(col_id)
            current_top = min(current_top, top)
            current_bottom = max(current_bottom, bottom)
            current_left = min(current_left, left)
        else:
            seeds.append(
                _FlowSeed(
                    flow_id=f"flow_top_{len(seeds)}",
                    kind="continuation",
                    col_ids=tuple(current_cols),
                    anchor_y=current_top,
                    anchor_x=current_left,
                    title_priority=1,
                )
            )
            current_cols = [col_id]
            current_top = top
            current_bottom = bottom
            current_left = left

    seeds.append(
        _FlowSeed(
            flow_id=f"flow_top_{len(seeds)}",
            kind="continuation",
            col_ids=tuple(current_cols),
            anchor_y=current_top,
            anchor_x=current_left,
            title_priority=1,
        )
    )
    return seeds


def _collect_peripheral_flow_seeds(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
    col_count: int,
) -> List[_FlowSeed]:
    if col_count < 3 or not image_height or image_height <= 0:
        return []

    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height), 1.0)
    textlike_blocks = [
        blk
        for blk in blocks
        if blk.block_type not in SPAN_ELIGIBLE_TYPES
        and blk.block_type not in _STRIP_TYPES
        and len(getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]) == 1
    ]
    if len(textlike_blocks) < 6:
        return []

    by_col: dict[int, List["Block"]] = {}
    for blk in textlike_blocks:
        by_col.setdefault(int(getattr(blk, "col_index", 0)), []).append(blk)

    middle_cols = [idx for idx in range(1, col_count - 1)]
    middle_blocks = [blk for col_id in middle_cols for blk in by_col.get(col_id, [])]
    if len(middle_blocks) < 4:
        return []

    middle_widths = sorted(float(blk.bbox.width) for blk in middle_blocks)
    middle_width_ref = middle_widths[len(middle_widths) // 2] if middle_widths else page_w / max(col_count, 1)

    seeds: List[_FlowSeed] = []
    for edge_col in (0, col_count - 1):
        members = sorted(by_col.get(edge_col, []), key=lambda b: (b.bbox.y1, b.bbox.x1))
        if len(members) < 2:
            continue

        widths = sorted(float(blk.bbox.width) for blk in members)
        median_width = widths[len(widths) // 2] if widths else 0.0
        col_center = sum((float(blk.bbox.x1) + float(blk.bbox.x2)) * 0.5 for blk in members) / max(len(members), 1)
        near_edge = col_center <= page_w * 0.28 or col_center >= page_w * 0.72
        narrow_enough = median_width <= min(page_w * 0.32, middle_width_ref * 0.86)
        coverage = _y_coverage_ratio(members, 0.0, page_h)
        short_text_count = sum(
            1
            for blk in members
            if _line_count(blk) <= 4 and len(_block_text(blk).strip()) <= 220
        )
        title_count = sum(1 for blk in members if blk.block_type == BlockType.TITLE)
        if not (near_edge and narrow_enough and coverage >= 0.16):
            continue
        if short_text_count + title_count < 2:
            continue

        anchor = next(
            (
                blk for blk in members
                if blk.block_type == BlockType.TITLE or (_line_count(blk) <= 4 and len(_block_text(blk).strip()) <= 220)
            ),
            members[0],
        )
        seeds.append(
            _FlowSeed(
                flow_id=f"flow_side_{len(seeds)}",
                kind="peripheral",
                col_ids=(edge_col,),
                anchor_y=float(anchor.bbox.y1),
                anchor_x=float(anchor.bbox.x1),
                title_priority=0,
            )
        )

    return seeds


def _collect_flow_seeds(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
    col_count: int,
) -> List[_FlowSeed]:
    col_width = max(float(image_width) / max(col_count, 1), 1.0)
    title_seeds: List[_FlowSeed] = []
    ordered_blocks = sorted(blocks, key=lambda b: (b.bbox.y1, b.bbox.x1))

    for blk in ordered_blocks:
        if _is_candidate_title_anchor(blk, image_width, image_height):
            cols = _block_cols(blk)
            title_seeds.append(
                _FlowSeed(
                    flow_id=f"flow_title_{len(title_seeds)}",
                    kind="title",
                    col_ids=cols,
                    anchor_y=float(blk.bbox.y1),
                    anchor_x=float(blk.bbox.x1),
                    title_priority=0,
                )
            )

    for blk in ordered_blocks:
        if not _is_candidate_kicker_anchor(
            blk,
            image_width=image_width,
            image_height=image_height,
            col_width=col_width,
        ):
            continue
        if _is_attached_to_title_seed(blk, title_seeds):
            continue
        title_seeds.append(
            _FlowSeed(
                flow_id=f"flow_title_{len(title_seeds)}",
                kind="kicker",
                col_ids=_block_cols(blk),
                anchor_y=float(blk.bbox.y1),
                anchor_x=float(blk.bbox.x1),
                title_priority=0,
            )
        )

    peripheral_seeds = _collect_peripheral_flow_seeds(
        blocks,
        image_width=image_width,
        image_height=image_height,
        col_count=col_count,
    )

    if title_seeds:
        first_anchor_y = min(seed.anchor_y for seed in title_seeds)
        top_seeds = _build_top_continuation_seeds(blocks, first_anchor_y, image_height, title_seeds)
    else:
        all_cols = tuple(sorted({col for blk in blocks for col in _block_cols(blk)}))
        top_seeds = [
            _FlowSeed(
                flow_id="flow_top_0",
                kind="continuation",
                col_ids=all_cols or (0,),
                anchor_y=min(float(blk.bbox.y1) for blk in blocks),
                anchor_x=min(float(blk.bbox.x1) for blk in blocks),
                title_priority=1,
            )
        ]
    return top_seeds + peripheral_seeds + title_seeds


def _seed_overlap_score(seed: _FlowSeed, block: "Block") -> tuple[int, float]:
    block_cols = set(_block_cols(block))
    overlap = len(block_cols.intersection(seed.col_ids))
    if overlap <= 0:
        return 0, float("inf")
    vertical_distance = max(0.0, float(block.bbox.y1) - seed.anchor_y)
    return overlap, vertical_distance


def _assign_blocks_to_flows(
    blocks: List["Block"],
    seeds: List[_FlowSeed],
) -> dict[str, List["Block"]]:
    flows: dict[str, List["Block"]] = {seed.flow_id: [] for seed in seeds}
    ordered_seeds = sorted(seeds, key=lambda s: (s.anchor_y, s.anchor_x, s.title_priority))

    for blk in sorted(blocks, key=lambda b: (b.bbox.y1, b.bbox.x1)):
        matching = [seed for seed in ordered_seeds if set(_block_cols(blk)).intersection(seed.col_ids)]
        if not matching:
            matching = ordered_seeds

        below_or_touching = [seed for seed in matching if seed.anchor_y <= float(blk.bbox.y1) + 36.0]
        chosen_pool = below_or_touching or matching
        chosen = max(
            chosen_pool,
            key=lambda seed: (
                _seed_overlap_score(seed, blk)[0],
                -abs(_seed_overlap_score(seed, blk)[1]),
                -seed.title_priority,
                -seed.anchor_y,
            ),
        )
        _set_flow_meta(blk, chosen.flow_id, chosen.kind)
        flows.setdefault(chosen.flow_id, []).append(blk)

    return flows


def _merge_overhanging_continuation_flows(
    flow_blocks: dict[str, List["Block"]],
    seeds_by_id: dict[str, _FlowSeed],
    *,
    image_height: int | None,
) -> dict[str, List["Block"]]:
    """Attach side-column overhangs to the lower title flow they continue."""
    if len(flow_blocks) < 2:
        return flow_blocks

    page_h = max(float(image_height or 0), 1.0)
    result = {flow_id: list(blocks) for flow_id, blocks in flow_blocks.items()}
    title_seeds = [
        seed for seed in seeds_by_id.values()
        if seed.kind in {"title", "kicker"}
    ]
    continuation_seeds = [
        seed for seed in seeds_by_id.values()
        if seed.kind == "continuation" and len(seed.col_ids) == 1
    ]

    for cont_seed in continuation_seeds:
        members = result.get(cont_seed.flow_id, [])
        if not members or any(blk.block_type == BlockType.TITLE for blk in members):
            continue

        top = min(float(blk.bbox.y1) for blk in members)
        bottom = max(float(blk.bbox.y2) for blk in members)
        first_block = min(members, key=lambda b: (b.bbox.y1, b.bbox.x1))
        first_text = _block_text(first_block).strip()

        candidates: List[tuple[float, float, _FlowSeed]] = []
        for title_seed in title_seeds:
            if cont_seed.flow_id == title_seed.flow_id:
                continue
            if min(cont_seed.col_ids) <= max(title_seed.col_ids):
                continue
            if min(cont_seed.col_ids) - max(title_seed.col_ids) > 1:
                continue
            crosses_title_band = top < title_seed.anchor_y and bottom > title_seed.anchor_y + page_h * 0.08
            sentence_continuation = _looks_like_sentence_continuation(first_text)
            if not (crosses_title_band or sentence_continuation):
                continue
            candidates.append((abs(float(cont_seed.anchor_x) - float(title_seed.anchor_x)), title_seed.anchor_y, title_seed))

        if not candidates:
            continue

        candidates.sort(key=lambda item: (item[0], -item[1]))
        target_seed = candidates[0][2]
        for blk in members:
            _set_flow_meta(blk, target_seed.flow_id, target_seed.kind)
        result[target_seed.flow_id] = sorted(
            result.get(target_seed.flow_id, []) + members,
            key=lambda b: (b.bbox.y1, b.bbox.x1),
        )
        result.pop(cont_seed.flow_id, None)

    return {flow_id: blocks for flow_id, blocks in result.items() if blocks}


def _reattach_pretitle_fragments(flow_blocks: dict[str, List["Block"]]) -> dict[str, List["Block"]]:
    """把紧贴标题上方的短 kicker/section label 挂回标题 flow。"""
    result = {flow_id: list(blocks) for flow_id, blocks in flow_blocks.items()}

    title_targets: List[tuple[str, "Block"]] = []
    for flow_id, blocks in result.items():
        for blk in blocks:
            if blk.block_type == BlockType.TITLE:
                title_targets.append((flow_id, blk))
                break

    for target_flow_id, title_block in title_targets:
        for flow_id, blocks in list(result.items()):
            if flow_id == target_flow_id:
                continue
            remain: List["Block"] = []
            moved: List["Block"] = []
            for blk in blocks:
                if not _is_textlike_block(blk) or blk.block_type == BlockType.TITLE:
                    remain.append(blk)
                    continue
                if _line_count(blk) > 2:
                    remain.append(blk)
                    continue
                text = _block_text(blk).strip()
                if not text or len(text) > 80:
                    remain.append(blk)
                    continue
                vertical_gap = float(title_block.bbox.y1) - float(blk.bbox.y2)
                horizontal_overlap = max(0.0, min(float(title_block.bbox.x2), float(blk.bbox.x2)) - max(float(title_block.bbox.x1), float(blk.bbox.x1)))
                overlap_ratio = horizontal_overlap / max(1.0, min(float(title_block.bbox.width), float(blk.bbox.width)))
                if 0.0 <= vertical_gap <= 180.0 and overlap_ratio >= 0.20:
                    moved.append(blk)
                else:
                    remain.append(blk)
            if moved:
                result[flow_id] = remain
                result[target_flow_id] = sorted(result[target_flow_id] + moved, key=lambda b: (b.bbox.y1, b.bbox.x1))

    return {flow_id: blocks for flow_id, blocks in result.items() if blocks}


def _flow_bbox(blocks: List["Block"]) -> tuple[float, float, float, float]:
    return _zone_bounds(blocks)


def _reorder_flows(
    ordered_flow_ids: List[str],
    flow_blocks: dict[str, List["Block"]],
    seeds_by_id: dict[str, _FlowSeed],
) -> List[str]:
    ranked = sorted(
        ordered_flow_ids,
        key=lambda flow_id: (
            seeds_by_id[flow_id].anchor_y,
            seeds_by_id[flow_id].anchor_x,
            seeds_by_id[flow_id].title_priority,
        ),
    )

    changed = True
    while changed:
        changed = False
        for idx, flow_id in enumerate(list(ranked)):
            seed = seeds_by_id[flow_id]
            if seed.kind != "continuation" or len(seed.col_ids) != 1:
                continue
            fb = _flow_bbox(flow_blocks.get(flow_id, []))
            for later_id in ranked[idx + 1:]:
                later_seed = seeds_by_id[later_id]
                if later_seed.title_priority > seed.title_priority:
                    continue
                if later_seed.anchor_x >= seed.anchor_x:
                    continue
                lb = _flow_bbox(flow_blocks.get(later_id, []))
                vertical_overlap = min(fb[3], lb[3]) - max(fb[1], lb[1])
                if vertical_overlap <= 0:
                    continue
                ranked.remove(flow_id)
                insert_at = ranked.index(later_id) + 1
                ranked.insert(insert_at, flow_id)
                changed = True
                break
            if changed:
                break
    return ranked


def _choose_flow_strategy(blocks: List["Block"], image_width: int) -> str:
    if any(blk.block_type == BlockType.TITLE for blk in blocks):
        return "xycutpp"
    if any(len(getattr(blk, "spanned_cols", []) or []) > 1 for blk in blocks):
        return "xycutpp"
    if any(blk.block_type in SPAN_ELIGIBLE_TYPES for blk in blocks) and len(blocks) >= 4:
        return "xycutpp"
    if _looks_complex_for_xycutpp(blocks, image_width):
        return "xycutpp"
    return "legacy"


def _sort_flow_anchor_blocks(blocks: List["Block"]) -> List["Block"]:
    if not blocks:
        return []
    titles = [blk for blk in blocks if blk.block_type == BlockType.TITLE]
    if not titles:
        return sorted(blocks, key=_head_block_sort_key)

    first_title_y = min(float(blk.bbox.y1) for blk in titles)
    pretitle: List["Block"] = []
    rest: List["Block"] = []
    for blk in blocks:
        if (
            blk.block_type == BlockType.TEXT
            and _line_count(blk) <= 2
            and len(_block_text(blk).strip()) <= 160
            and float(blk.bbox.y2) <= first_title_y + 8.0
        ):
            pretitle.append(blk)
        else:
            rest.append(blk)
    return sorted(pretitle, key=lambda b: (b.bbox.y1, b.bbox.x1)) + sorted(rest, key=_head_block_sort_key)


def _sort_same_column_runs_by_y(blocks: List["Block"]) -> List["Block"]:
    ordered: List["Block"] = []
    idx = 0
    while idx < len(blocks):
        block = blocks[idx]
        cols = getattr(block, "spanned_cols", []) or [getattr(block, "col_index", 0)]
        if not _is_textlike_block(block) or len(cols) != 1:
            ordered.append(block)
            idx += 1
            continue

        run = [block]
        idx += 1
        while idx < len(blocks):
            cand = blocks[idx]
            cand_cols = getattr(cand, "spanned_cols", []) or [getattr(cand, "col_index", 0)]
            if not _is_textlike_block(cand) or len(cand_cols) != 1:
                break
            if int(cand_cols[0]) != int(cols[0]) or getattr(cand, "col_count", 1) != getattr(block, "col_count", 1):
                break
            run.append(cand)
            idx += 1

        if len(run) >= 3 and any(float(a.bbox.y1) > float(b.bbox.y1) + 12.0 for a, b in zip(run, run[1:])):
            ordered.extend(sorted(run, key=lambda b: (b.bbox.y1, b.bbox.x1)))
        else:
            ordered.extend(run)
    return ordered


def _sort_blocks_in_flow(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
    max_cols: int,
    cluster_thresh: float,
    column_confidence_min: float,
    zone_strip_height_ratio: float,
    xycutpp_beta: float,
    xycutpp_density_threshold: float,
    xycutpp_min_gap_ratio: float,
    xycutpp_title_width_ratio: float,
) -> List["Block"]:
    fx1, fy1, fx2, fy2 = _flow_bbox(blocks)
    local_width = max(int(round(fx2 - fx1)), max_cols * 120)
    local_height = max(int(round(fy2 - fy1)), 1)

    flow_cols = {tuple(sorted(getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)])) for blk in blocks}
    single_col_ids = {cols[0] for cols in flow_cols if len(cols) == 1}
    if len(single_col_ids) == 1 and all(len(cols) == 1 for cols in flow_cols):
        _assign_single_column(blocks)
        return _sort_single_column_blocks(blocks)
    if len(single_col_ids) >= 2:
        widths = sorted(float(blk.bbox.width) for blk in blocks if len(getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]) == 1)
        median_width = widths[len(widths) // 2] if widths else max((fx2 - fx1) / max(len(single_col_ids), 1), 1.0)
        wide_thresh = max(median_width * 1.2, float(local_width) * 0.32)

        title_blocks = [blk for blk in blocks if blk.block_type == BlockType.TITLE]
        title_top = min((float(blk.bbox.y1) for blk in title_blocks), default=float("inf"))
        title_bottom = max((float(blk.bbox.y2) for blk in title_blocks), default=float("-inf"))

        anchor_blocks: List["Block"] = []
        body_blocks: List["Block"] = []
        tail_blocks: List["Block"] = []
        min_body_y = min(
            (
                float(blk.bbox.y1)
                for blk in blocks
                if len(getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]) == 1
                and blk.block_type not in {BlockType.TITLE}
                and float(blk.bbox.width) < wide_thresh
            ),
            default=float("inf"),
        )

        for blk in sorted(blocks, key=lambda b: (b.bbox.y1, b.bbox.x1)):
            cols = getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]
            is_wide = len(cols) > 1 or float(blk.bbox.width) >= wide_thresh
            is_caption = blk.block_type in {
                BlockType.FIGURE_CAPTION,
                BlockType.TABLE_CAPTION,
                BlockType.TABLE_FOOTNOTE,
                BlockType.FORMULA_CAPTION,
            }
            is_visual = blk.block_type in SPAN_ELIGIBLE_TYPES
            is_top_visual = (
                is_visual
                and title_blocks
                and float(blk.bbox.y2) <= title_top + max(140.0, float(local_height) * 0.12)
            )
            near_title_band = (
                title_blocks
                and float(blk.bbox.y2) >= title_top - 180.0
                and float(blk.bbox.y1) <= title_bottom + 520.0
                and blk.block_type in {BlockType.TEXT, BlockType.TITLE}
                and _line_count(blk) <= 2
            )
            if (
                blk.block_type == BlockType.TITLE
                or near_title_band
                or is_top_visual
                or (is_wide and float(blk.bbox.y1) <= min_body_y + 140.0 and not is_visual)
            ):
                anchor_blocks.append(blk)
            elif is_visual or is_caption or is_wide:
                tail_blocks.append(blk)
            else:
                body_blocks.append(blk)

        ordered_body: List["Block"] = []
        by_col: dict[int, List["Block"]] = {}
        for blk in body_blocks:
            by_col.setdefault(int(blk.col_index), []).append(blk)
        for col_idx in sorted(by_col.keys()):
            ordered_body.extend(_sort_blocks_by_reading_row(by_col[col_idx]))

        ordered = _sort_flow_anchor_blocks(anchor_blocks) + ordered_body + sorted(
            tail_blocks,
            key=lambda b: (b.bbox.y1, b.bbox.x1),
        )

        present_cols = sorted(single_col_ids)
        col_map = {col_id: idx for idx, col_id in enumerate(present_cols)}
        for blk in ordered:
            cols = getattr(blk, "spanned_cols", []) or [getattr(blk, "col_index", 0)]
            normalized = [col_map.get(int(col), 0) for col in cols]
            blk.col_count = len(present_cols)
            blk.col_index = min(normalized) if normalized else 0
            blk.spanned_cols = sorted(set(normalized or [0]))
        return _sort_same_column_runs_by_y(ordered)

    strategy = _choose_flow_strategy(blocks, image_width)
    if strategy == "xycutpp":
        return sort_layout_xycutpp(
            blocks,
            image_width=local_width,
            image_height=local_height,
            max_cols=max_cols,
            cluster_thresh=cluster_thresh,
            column_confidence_min=column_confidence_min,
            zone_strip_height_ratio=zone_strip_height_ratio,
            beta=xycutpp_beta,
            density_threshold=xycutpp_density_threshold,
            min_gap_ratio=xycutpp_min_gap_ratio,
            title_width_ratio=xycutpp_title_width_ratio,
        )
    return sort_layout_legacy(
        blocks,
        image_width=local_width,
        image_height=local_height,
        max_cols=max_cols,
        cluster_thresh=cluster_thresh,
        column_confidence_min=column_confidence_min,
        zone_strip_height_ratio=zone_strip_height_ratio,
    )


def _should_use_article_flow_segmentation(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
) -> bool:
    if len(blocks) < 8:
        return False
    title_count = sum(1 for blk in blocks if blk.block_type == BlockType.TITLE)
    figure_count = sum(1 for blk in blocks if blk.block_type in SPAN_ELIGIBLE_TYPES)
    if title_count == 0:
        return False
    if _looks_like_academic_sectioned_page(blocks, image_width):
        return False
    if not _looks_complex_for_xycutpp(blocks, image_width):
        return False
    if image_height and image_height > 0:
        lower_titles = sum(1 for blk in blocks if blk.block_type == BlockType.TITLE and float(blk.bbox.y1) > float(image_height) * 0.28)
        if lower_titles >= 1:
            return True
    return figure_count >= 1 and title_count >= 1


def _sort_layout_with_article_flows(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
    max_cols: int,
    cluster_thresh: float,
    column_confidence_min: float,
    zone_strip_height_ratio: float,
    xycutpp_beta: float,
    xycutpp_density_threshold: float,
    xycutpp_min_gap_ratio: float,
    xycutpp_title_width_ratio: float,
) -> List["Block"]:
    if not blocks:
        return []

    _global_assign_columns(blocks, image_width=image_width, max_cols=max_cols, cluster_thresh=cluster_thresh)
    col_count = max((blk.col_count for blk in blocks), default=1)
    if col_count <= 1:
        return sort_layout_xycutpp(
            blocks,
            image_width=image_width,
            image_height=image_height,
            max_cols=max_cols,
            cluster_thresh=cluster_thresh,
            column_confidence_min=column_confidence_min,
            zone_strip_height_ratio=zone_strip_height_ratio,
            beta=xycutpp_beta,
            density_threshold=xycutpp_density_threshold,
            min_gap_ratio=xycutpp_min_gap_ratio,
            title_width_ratio=xycutpp_title_width_ratio,
        )

    seeds = _collect_flow_seeds(blocks, image_width=image_width, image_height=image_height, col_count=col_count)
    if len(seeds) <= 1:
        return sort_layout_xycutpp(
            blocks,
            image_width=image_width,
            image_height=image_height,
            max_cols=max_cols,
            cluster_thresh=cluster_thresh,
            column_confidence_min=column_confidence_min,
            zone_strip_height_ratio=zone_strip_height_ratio,
            beta=xycutpp_beta,
            density_threshold=xycutpp_density_threshold,
            min_gap_ratio=xycutpp_min_gap_ratio,
            title_width_ratio=xycutpp_title_width_ratio,
        )

    seeds_by_id = {seed.flow_id: seed for seed in seeds}
    flow_blocks = _assign_blocks_to_flows(blocks, seeds)
    flow_blocks = _merge_overhanging_continuation_flows(
        flow_blocks,
        seeds_by_id,
        image_height=image_height,
    )
    flow_blocks = _reattach_pretitle_fragments(flow_blocks)
    ordered_flow_ids = _reorder_flows(list(flow_blocks.keys()), flow_blocks, seeds_by_id)

    ordered: List["Block"] = []
    for flow_id in ordered_flow_ids:
        members = flow_blocks.get(flow_id, [])
        if not members:
            continue
        flow_kind = seeds_by_id[flow_id].kind
        local_sorted = _sort_blocks_in_flow(
            members,
            image_width=image_width,
            image_height=image_height,
            max_cols=max_cols,
            cluster_thresh=cluster_thresh,
            column_confidence_min=column_confidence_min,
            zone_strip_height_ratio=zone_strip_height_ratio,
            xycutpp_beta=xycutpp_beta,
            xycutpp_density_threshold=xycutpp_density_threshold,
            xycutpp_min_gap_ratio=xycutpp_min_gap_ratio,
            xycutpp_title_width_ratio=xycutpp_title_width_ratio,
        )
        for blk in local_sorted:
            _set_flow_meta(blk, flow_id, flow_kind)
        ordered.extend(local_sorted)

    _clear_flow_meta(ordered)
    return postprocess_xycutpp_local_attachments(
        ordered,
        image_width=image_width,
        image_height=image_height,
    )


def _sort_single_column_blocks(blocks: List["Block"]) -> List["Block"]:
    return sorted(blocks, key=lambda b: (b.bbox.y1, b.bbox.x1))


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


def _is_small_decorative_visual(
    block: "Block",
    image_width: int,
    image_height: int | None,
) -> bool:
    if block.block_type not in SPAN_ELIGIBLE_TYPES:
        return False
    page_w = max(float(image_width), 1.0)
    page_h = max(float(image_height or 0), 1.0)
    return (
        float(block.bbox.width) <= page_w * 0.10
        and float(block.bbox.height) <= max(page_h * 0.08, 120.0)
    )


def _is_title_like_zone_block(
    block: "Block",
    image_width: int,
    image_height: int | None,
) -> bool:
    return block.block_type == BlockType.TITLE or _is_small_decorative_visual(block, image_width, image_height)


def _merge_title_only_zones(
    zones: List[List["Block"]],
    image_width: int,
    image_height: int | None,
) -> List[List["Block"]]:
    if len(zones) < 2:
        return zones

    page_h = max(float(image_height or 0), 1.0)
    merged: List[List["Block"]] = []
    idx = 0
    while idx < len(zones):
        zone = list(zones[idx])
        if not (
            zone
            and len(zone) <= 3
            and all(_is_title_like_zone_block(blk, image_width, image_height) for blk in zone)
        ):
            merged.append(zone)
            idx += 1
            continue

        cluster = list(zone)
        j = idx + 1
        while j < len(zones):
            next_zone = list(zones[j])
            current_bottom = max(float(blk.bbox.y2) for blk in cluster)
            next_top = min(float(blk.bbox.y1) for blk in next_zone)
            vertical_gap = max(0.0, next_top - current_bottom)
            zone_center = sum((float(blk.bbox.x1) + float(blk.bbox.x2)) * 0.5 for blk in cluster) / max(len(cluster), 1)
            next_x1, _next_y1, next_x2, _next_y2 = _zone_bounds(next_zone)
            overlaps_next = any(
                max(0.0, min(float(blk.bbox.x2), next_x2) - max(float(blk.bbox.x1), next_x1)) >= max(24.0, float(blk.bbox.width) * 0.10)
                for blk in cluster
            )
            center_within_next = next_x1 - image_width * 0.06 <= zone_center <= next_x2 + image_width * 0.06
            if vertical_gap > max(96.0, page_h * 0.06) or not (overlaps_next or center_within_next):
                break

            cluster.extend(next_zone)
            j += 1
            if not (len(next_zone) <= 3 and all(_is_title_like_zone_block(blk, image_width, image_height) for blk in next_zone)):
                break

        if len(cluster) > len(zone):
            merged.append(sorted(cluster, key=lambda b: (b.bbox.y1, b.bbox.x1)))
            idx = j
            continue

        merged.append(zone)
        idx += 1
    return merged


def _extract_zone_title_band(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
) -> List["Block"]:
    titles = [blk for blk in blocks if blk.block_type == BlockType.TITLE]
    if not titles:
        return []

    top_title_y = min(float(blk.bbox.y1) for blk in titles)
    top_title_height = max(float(blk.bbox.height) for blk in titles if float(blk.bbox.y1) == top_title_y)
    # 标题带只吸收真正贴着 section header 的同层标题，避免把下一行的左右子标题一起提前。
    band_limit = top_title_y + max(28.0, min(48.0, top_title_height * 0.80))
    anchor_titles = [
        blk for blk in titles
        if float(blk.bbox.y1) <= band_limit
    ]
    if not anchor_titles:
        return []

    max_title_bottom = max(float(blk.bbox.y2) for blk in anchor_titles)
    title_band = list(anchor_titles)
    for blk in blocks:
        if blk in title_band:
            continue
        if not _is_small_decorative_visual(blk, image_width, image_height):
            continue
        if float(blk.bbox.y1) > max_title_bottom + 48.0:
            continue
        if any(
            max(0.0, min(float(blk.bbox.x2), float(title.bbox.x2)) - max(float(blk.bbox.x1), float(title.bbox.x1)))
            >= min(float(blk.bbox.width), float(title.bbox.width)) * 0.08
            for title in anchor_titles
        ):
            title_band.append(blk)
    return title_band


def _sort_zone_blocks_with_title_band(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None,
    *,
    by_columns: bool,
) -> List["Block"]:
    title_band = _extract_zone_title_band(blocks, image_width, image_height)
    if not title_band:
        return _ordered_by_columns(blocks) if by_columns else _sort_blocks_by_reading_row(blocks)

    title_ids = {id(blk) for blk in title_band}
    ordered_head = _sort_blocks_by_reading_row(title_band)
    remainder = [blk for blk in blocks if id(blk) not in title_ids]
    if not remainder:
        return ordered_head
    ordered_tail = _ordered_by_columns(remainder) if by_columns else _sort_blocks_by_reading_row(remainder)
    return ordered_head + ordered_tail


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


def _looks_complex_for_xycutpp(blocks: List["Block"], image_width: int) -> bool:
    if len(blocks) < 4:
        return False

    wide_threshold = max(float(image_width) * 0.45, 1.0)
    wide_text_like = sum(
        1
        for blk in blocks
        if blk.block_type in {
            BlockType.TEXT,
            BlockType.TITLE,
            BlockType.REFERENCE,
            BlockType.ABSTRACT,
            BlockType.FIGURE_CAPTION,
            BlockType.TABLE_CAPTION,
            BlockType.TABLE_FOOTNOTE,
            BlockType.FORMULA_CAPTION,
        }
        and float(blk.bbox.width) >= wide_threshold
    )
    has_dynamic_visual = any(blk.block_type in SPAN_ELIGIBLE_TYPES for blk in blocks)
    has_titles = any(blk.block_type == BlockType.TITLE for blk in blocks)
    has_many_columns_hint = len({
        int((blk.bbox.x1 + blk.bbox.x2) * 0.5 // max(float(image_width) / 3.0, 1.0))
        for blk in blocks
        if blk.block_type not in SPAN_ELIGIBLE_TYPES and blk.block_type not in _STRIP_TYPES
    }) >= 2

    return wide_text_like >= 1 or (has_dynamic_visual and has_titles) or has_many_columns_hint


def sort_layout_legacy(
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

    zones = _merge_title_only_zones(
        split_into_zones(blocks, image_width),
        image_width=image_width,
        image_height=image_height,
    )
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
            ordered.extend(
                _sort_zone_blocks_with_title_band(
                    all_blocks,
                    image_width=image_width,
                    image_height=image_height,
                    by_columns=False,
                )
            )
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

        ordered.extend(
            _sort_zone_blocks_with_title_band(
                all_blocks,
                image_width=image_width,
                image_height=image_height,
                by_columns=True,
            )
        )

    return ordered


def sort_layout(
    blocks: List["Block"],
    image_width: int,
    image_height: int | None = None,
    max_cols: int = MAX_COLS,
    cluster_thresh: float = COLUMN_CLUSTER_THRESH,
    column_confidence_min: float = 0.55,
    zone_strip_height_ratio: float = 0.12,
    strategy: str = "auto",
    xycutpp_beta: float = 1.3,
    xycutpp_density_threshold: float = 0.9,
    xycutpp_min_gap_ratio: float = 0.015,
    xycutpp_title_width_ratio: float = 0.45,
) -> List["Block"]:
    """将 *blocks* 按自然阅读顺序重新排列并填充列元数据。

    Parameters
    ----------
    strategy:
        ``"legacy"`` 使用原有分区/分栏排序器；
        其余 ``"xycutpp"`` / ``"xycutpp_hybrid"`` / ``"xycutpp_paper"`` /
        ``"newspaper_hybrid"`` / ``"auto"`` 全部统一走 XY-Cut++ 内核。
    """
    mode = (strategy or "auto").strip().lower()
    if mode == "xycutpp":
        mode = "xycutpp_hybrid"
    if mode not in {"legacy", "xycutpp_paper", "xycutpp_hybrid", "newspaper_hybrid", "auto"}:
        mode = "auto"
    if mode == "legacy":
        return sort_layout_legacy(
            blocks,
            image_width=image_width,
            image_height=image_height,
            max_cols=max_cols,
            cluster_thresh=cluster_thresh,
            column_confidence_min=column_confidence_min,
            zone_strip_height_ratio=zone_strip_height_ratio,
        )
    evidence_blocks = _clone_blocks_for_evidence(blocks)
    evidence = _collect_layout_evidence(
        evidence_blocks,
        image_width=image_width,
        image_height=image_height,
        max_cols=max_cols,
        cluster_thresh=cluster_thresh,
    )
    if (
        mode in {"xycutpp_hybrid", "newspaper_hybrid", "auto"}
        and _has_strong_multiflow_evidence(
            evidence_blocks,
            image_width=image_width,
            image_height=image_height,
            evidence=evidence,
        )
    ):
        return _sort_layout_with_article_flows(
            blocks,
            image_width=image_width,
            image_height=image_height,
            max_cols=max_cols,
            cluster_thresh=cluster_thresh,
            column_confidence_min=column_confidence_min,
            zone_strip_height_ratio=zone_strip_height_ratio,
            xycutpp_beta=xycutpp_beta,
            xycutpp_density_threshold=xycutpp_density_threshold,
            xycutpp_min_gap_ratio=xycutpp_min_gap_ratio,
            xycutpp_title_width_ratio=xycutpp_title_width_ratio,
        )
    return sort_layout_xycutpp(
        blocks,
        image_width=image_width,
        image_height=image_height,
        max_cols=max_cols,
        cluster_thresh=cluster_thresh,
        column_confidence_min=column_confidence_min,
        zone_strip_height_ratio=zone_strip_height_ratio,
        beta=xycutpp_beta,
        density_threshold=xycutpp_density_threshold,
        min_gap_ratio=xycutpp_min_gap_ratio,
        title_width_ratio=xycutpp_title_width_ratio,
    )
