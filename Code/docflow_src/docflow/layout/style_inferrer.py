"""版面样式推断器。

从 bbox、category 和文本行位置推断 block.style 中缺失的字段，
并计算多段落区块各段的首行缩进。

职责边界
--------
- **输入**：已完成版面分析（col_index 已设置）、字号已估算的 Block 列表
- **输出**：block.style.* 全部字段填充完毕；multi-para 时 para.first_line_indent_px 已设置
- **不做**：任何 DOCX / Markdown 渲染，不产生任何输出文件

优先级原则
----------
JSON 中已明确提供的字段（值不为 None）一律不覆盖；
只在字段为 None 时才写入推断值。
"""
from __future__ import annotations

import re
from typing import List, Optional, TYPE_CHECKING, Tuple

from docflow.model.base import BlockType
from docflow.model.blocks.text_block import TextBlock
from docflow.schema.models import BlockStyle
from docflow.layout.paragraph_detector import detect_first_line_indent

if TYPE_CHECKING:
    from docflow.model.zone import Zone
    from docflow.model.page import CoordMapper


# caption 类区块集合（与 renderer 保持一致）
_CAPTION_TYPES: frozenset = frozenset({
    BlockType.TABLE_CAPTION,
    BlockType.FIGURE_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.FORMULA_CAPTION,
})
_NUMBERED_TITLE_LEVEL_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:[\.、])?\s*\S")


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

def infer_block_styles(
    zones: List["Zone"],
    mapper: "CoordMapper",
    justify_min_lines: int = 3,
    page_width_px: float = 0.0,
) -> None:
    """推断并填充所有 TextBlock 的 block.style。

    遍历 *zones*，为每个 zone 计算各列的像素边界（col_px），
    再对该 zone 内每个 TextBlock 调用 :func:`_infer_text_block`。

    Parameters
    ----------
    zones:
        已完成 col_index 分配的 Zone 列表（来自 pipeline 的 _blocks_to_zones）。
    mapper:
        坐标映射器，用于 px → pt 转换。
    """
    all_text_blocks: List[TextBlock] = []

    for zone in zones:
        # 列边界优先使用文本块；若该列无文本块，再回退到全区块边界。
        col_px_text: dict = {}
        col_px_all: dict = {}
        text_blocks_in_zone: List[TextBlock] = []
        for b in zone.blocks:
            ci = b.col_index
            if ci not in col_px_all:
                col_px_all[ci] = [b.bbox.x1, b.bbox.x2]
            else:
                col_px_all[ci][0] = min(col_px_all[ci][0], b.bbox.x1)
                col_px_all[ci][1] = max(col_px_all[ci][1], b.bbox.x2)

            if isinstance(b, TextBlock):
                text_blocks_in_zone.append(b)
                if ci not in col_px_text:
                    col_px_text[ci] = [b.bbox.x1, b.bbox.x2]
                else:
                    col_px_text[ci][0] = min(col_px_text[ci][0], b.bbox.x1)
                    col_px_text[ci][1] = max(col_px_text[ci][1], b.bbox.x2)

        if (
            page_width_px > 0
            and zone.col_count == 1
            and zone.rendering_strategy == "single_col"
            and len(text_blocks_in_zone) == 1
        ):
            ci = text_blocks_in_zone[0].col_index
            col_px_text[ci] = [0.0, float(page_width_px)]
            col_px_all[ci] = [0.0, float(page_width_px)]

        for block in zone.blocks:
            if isinstance(block, TextBlock):
                _infer_text_block(
                    block=block,
                    col_px_text=col_px_text,
                    col_px_all=col_px_all,
                    mapper=mapper,
                    justify_min_lines=justify_min_lines,
                )
                all_text_blocks.append(block)

    # 对同类别区块的字号离群值归一化
    _normalize_category_styles(all_text_blocks)

    # 使用推断出的行距修正字号（设计文档 §4.1：字号应在源页面坐标系中推断）
    # 初始估算假设 line_spacing ≈ 1.15，现用实际推断值修正
    _refine_font_size_from_line_spacing(all_text_blocks, mapper)


# ---------------------------------------------------------------------------
# 单区块推断
# ---------------------------------------------------------------------------

def _infer_text_block(
    block: TextBlock,
    col_px_text: dict,
    col_px_all: dict,
    mapper: "CoordMapper",
    justify_min_lines: int,
) -> None:
    """对单个 TextBlock 推断并填充缺失的 style 字段。

    只写入值为 None 的字段，JSON 中已有的值保持不变。
    """
    if block.style is None:
        block.style = BlockStyle()
    bs = block.style

    rtype = block.block_type
    is_title = rtype == BlockType.TITLE
    is_caption = rtype in _CAPTION_TYPES

    # ── 字号 ──────────────────────────────────────────────────────────
    if bs.font_size_pt is None:
        bs.font_size_pt = block.estimated_font_size_pt
    font_size = bs.font_size_pt or 10.5   # 保证后续计算有值

    # ── bold / italic ─────────────────────────────────────────────────
    # OCR 引擎无法可靠检测粗/斜体；此处仅填 False 作为中性默认值，
    # 待后续专用样式分类模型的结果写入 JSON block.style 后优先生效。
    if bs.bold is None:
        bs.bold = False
    if bs.italic is None:
        bs.italic = False

    # ── 对齐方式 ──────────────────────────────────────────────────────
    if bs.alignment is None:
        ci = block.col_index
        fallback = [block.bbox.x1, block.bbox.x2]
        col_left, col_right = col_px_text.get(ci, col_px_all.get(ci, fallback))
        bs.alignment = _detect_alignment(
            block=block,
            col_left=float(col_left),
            col_right=float(col_right),
            is_title=is_title,
            is_caption=is_caption,
            justify_min_lines=justify_min_lines,
        )

    # ── 行距（乘数） ──────────────────────────────────────────────────
    if bs.line_spacing is None:
        ls_mul = _estimate_line_spacing(block, font_size, mapper)
        if ls_mul is not None:
            bs.line_spacing = ls_mul

    # ── 段后间距默认值 ────────────────────────────────────────────────
    if bs.space_after_pt is None:
        bs.space_after_pt = 4.0 if is_title else 1.0

    # ── 首行缩进（单段落情形） ─────────────────────────────────────────
    if (bs.first_line_indent_pt is None
            and not is_title
            and not is_caption
            and block.paragraphs
            and len(block.paragraphs) == 1):
        indent = detect_first_line_indent(block.lines, mapper)
        bs.first_line_indent_pt = indent if indent > 0 else None

    # ── 多段落各自的首行缩进 ──────────────────────────────────────────
    if block.paragraphs and len(block.paragraphs) > 1:
        _infer_paragraph_indents(block)


# ---------------------------------------------------------------------------
# 多段落首行缩进
# ---------------------------------------------------------------------------

def _infer_paragraph_indents(block: TextBlock) -> None:
    """为多段落区块的每个段落推断首行缩进（px）。

    以所有段落中各段第 2 行及以后 x1 的最小值为"基线"，
    段落首行 x1 超出基线的偏移量即为该段落的首行缩进（像素）。
    若无非首行，退回到所有段落首行 x1 的最小值。
    结果写入 ``para.first_line_indent_px``。
    """
    paras = block.paragraphs

    # 收集基线 x：所有段落中各段第 2 行及以后的 x1
    non_first_xs: List[float] = []
    for para in paras:
        for ln in para.lines[1:]:
            if ln.x1 is not None:
                non_first_xs.append(ln.x1)

    if non_first_xs:
        baseline_x = min(non_first_xs)
    else:
        first_xs = [
            para.lines[0].x1
            for para in paras
            if para.lines and para.lines[0].x1 is not None
        ]
        baseline_x = min(first_xs) if first_xs else None

    if baseline_x is None:
        return

    for para in paras:
        if not para.lines:
            continue
        fx = para.lines[0].x1
        if fx is None:
            continue
        para.first_line_indent_px = max(0.0, fx - baseline_x)


# ---------------------------------------------------------------------------
# 对齐方式推断
# ---------------------------------------------------------------------------

def _extract_line_edges(block: TextBlock) -> List[Tuple[float, float]]:
    line_edges: List[Tuple[float, float]] = []
    for ln in block.lines:
        if ln.x1 is None or ln.x2 is None:
            continue
        line_edges.append((float(ln.x1), float(ln.x2)))
    return line_edges


def _center_evidence(
    line_edges: List[Tuple[float, float]],
    col_left: float,
    col_right: float,
) -> float:
    if not line_edges:
        return 0.0
    col_center = (col_left + col_right) * 0.5
    col_w = max(col_right - col_left, 1.0)
    offsets = [abs(((x1 + x2) * 0.5) - col_center) / col_w for x1, x2 in line_edges]
    mean_offset = sum(offsets) / len(offsets)
    return max(0.0, 1.0 - min(1.0, mean_offset / 0.22))


def _edge_gaps(
    line_edges: List[Tuple[float, float]],
    col_left: float,
    col_right: float,
) -> Tuple[float, float]:
    if not line_edges:
        return 0.0, 0.0
    left = sum(max(0.0, x1 - col_left) for x1, _ in line_edges) / len(line_edges)
    right = sum(max(0.0, col_right - x2) for _, x2 in line_edges) / len(line_edges)
    return left, right


def _detect_alignment(
    block: TextBlock,
    col_left: float,
    col_right: float,
    is_title: bool,
    is_caption: bool,
    justify_min_lines: int = 3,
) -> str:
    """按文本行几何分布推断对齐方式（行级主导）。"""
    def _title_level() -> Optional[int]:
        text = (block.full_text() or "").strip()
        match = _NUMBERED_TITLE_LEVEL_RE.match(text)
        if not match:
            return None
        return match.group(1).count(".") + 1

    if is_caption:
        return "center"

    col_w = col_right - col_left
    if col_w <= 0:
        return "left"

    line_edges = _extract_line_edges(block)
    if not line_edges:
        # 几何缺失时回退到 bbox。
        d_left = abs(block.bbox.x1 - col_left)
        d_right = abs(col_right - block.bbox.x2)
        if d_left <= col_w * 0.12 and d_right <= col_w * 0.12 and not is_title:
            return "justify"
        if abs(d_left - d_right) <= col_w * 0.1:
            return "center" if is_title else "left"
        return "left" if d_left <= d_right else "right"

    thresh = col_w * 0.10
    left_hit_ratio = sum(1 for x1, _ in line_edges if abs(x1 - col_left) <= thresh) / len(line_edges)
    right_hit_ratio = sum(1 for _, x2 in line_edges if abs(col_right - x2) <= thresh) / len(line_edges)
    body_lines = line_edges[:-1] if len(line_edges) >= 3 else line_edges
    ragged_right_ratio = (
        sum(1 for _, x2 in body_lines if (col_right - x2) > thresh) / max(len(body_lines), 1)
    )
    center_score = _center_evidence(line_edges, col_left, col_right)
    avg_left_gap, avg_right_gap = _edge_gaps(line_edges, col_left, col_right)
    is_short_block = len(line_edges) < max(1, justify_min_lines)
    non_justify_type = block.block_type in {
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.REFERENCE,
    }

    if is_title:
        if _title_level() is not None:
            return "left"
        title_text = (block.full_text() or "").strip()
        if title_text[:1] in {"“", "\"", "‘", "'"} and len(title_text) <= 42:
            return "center"
        if center_score >= 0.52:
            return "center"
        if left_hit_ratio >= 0.58 and right_hit_ratio >= 0.58:
            return "center"
        if len(line_edges) <= 2 and center_score >= 0.40 and left_hit_ratio >= 0.40 and right_hit_ratio >= 0.40:
            return "center"
        if abs(avg_left_gap - avg_right_gap) <= col_w * 0.08:
            return "center"
        return "left"

    if non_justify_type:
        if center_score >= 0.72 and left_hit_ratio < 0.45 and right_hit_ratio < 0.45:
            return "center"
        if right_hit_ratio >= 0.85 and left_hit_ratio < 0.35:
            return "right"
        return "left"

    if not is_short_block:
        if left_hit_ratio >= 0.72 and right_hit_ratio >= 0.72 and ragged_right_ratio <= 0.35:
            return "justify"

    if center_score >= 0.72 and left_hit_ratio < 0.45 and right_hit_ratio < 0.45:
        return "center"
    if right_hit_ratio >= 0.82 and left_hit_ratio < 0.45:
        return "right"
    return "left"


# ---------------------------------------------------------------------------
# 行距推断
# ---------------------------------------------------------------------------

def _estimate_line_spacing(
    block: TextBlock,
    font_size_pt: float,
    mapper: "CoordMapper",
) -> Optional[float]:
    """从文本行几何分布估算行距乘数。

    优先使用相邻文本行之间的平均间距来估算行距，
    比 bbox 高度法更准确——bbox 高度会混入段间距，
    导致行距乘数被高估。

    行距乘数定义为 ``每行平均高度（pt） / font_size_pt``，
    与 python-docx ``paragraph_format.line_spacing = float`` 语义一致
    （即 MULTIPLE 模式下乘以 font_size_pt 后的绝对行高）。

    仅在 **行数 ≥ 2** 时信任几何派生值，但对两行区块使用更严格的
    上下限（1.0–1.4），防止异常值导致行距偏大。

    Returns
    -------
    行距乘数（如 1.2），或 ``None``（使用渲染器全局默认值）。
    """
    if font_size_pt <= 0:
        return None
    num_lines = block.count_lines()
    if num_lines < 2:
        return None

    # 两行区块使用更严格的乘数范围
    ls_low = 1.0 if num_lines >= 3 else 1.05
    ls_high = 1.6 if num_lines >= 3 else 1.4

    # 优先方法：使用相邻文本行的平均垂直间距估算行距
    line_heights: List[float] = []
    lines = block.lines or []
    for ln in lines:
        if ln.text_region:
            ys = [float(pt[1]) for pt in ln.text_region if len(pt) >= 2]
            if len(ys) >= 2:
                line_heights.append(max(ys) - min(ys))

    if len(line_heights) >= 2:
        # 使用中位数行高（对异常值更鲁棒）
        sorted_h = sorted(line_heights)
        median_line_h = sorted_h[len(sorted_h) // 2]
        if median_line_h > 0:
            lh_pt = mapper.h(median_line_h)
            multiplier = lh_pt / font_size_pt
            if ls_low < multiplier < ls_high:
                return round(multiplier * 100) / 100

    # 回退：bbox 高度法（仅在文本行几何不可用时）
    if not line_heights:
        height_pt = mapper.h(block.bbox.height)
        lh_pt = height_pt / num_lines
        multiplier = lh_pt / font_size_pt
        if ls_low < multiplier < ls_high:
            return round(multiplier * 100) / 100
    return None


# ---------------------------------------------------------------------------
# 类别字号归一化
# ---------------------------------------------------------------------------

def _normalize_category_styles(blocks: List[TextBlock]) -> None:
    """按类别对字号和行距进行聚类归一化，消除 bbox 测量噪声。

    核心思路
    --------
    同类别区块的字号因 bbox / text_region 测量精度差异会出现离散噪声
    （如正文块估算出 7.5 / 8.0 / 8.5 / 9.0 / 9.5），但页面中可能确实存在
    有意义的多种字号（正文 10pt vs 脚注 8pt）。
    **全量对齐到中位数**会误伤合理差异，**阈值过滤**又对噪声幅度敏感。

    本方案采用 **1D 贪心聚类**：
    1. 将同类别字号排序后，按间距 ≤ ``merge_gap`` 合并相邻值到同一簇；
    2. 每簇内所有块统一到该簇的**加权中位数**（0.5pt 步长）；
    3. 簇间字号差异保留（如正文 vs 脚注不会被强行拉齐）。

    ``merge_gap`` 默认 2.0pt：
    - 覆盖 OCR 噪声（典型 ±1pt）；
    - 不合并有意义差异（正文 10pt 和脚注 8pt 差 2pt，刚好分开）。

    行距归一化
    ----------
    同类别内，若有推断行距值的块不超过一半，视为不可靠，全部清空为 None
    （交给全局默认值）。否则对有值的块用同样的聚类归一，无值的块赋予
    最大簇的中位数。

    Parameters
    ----------
    blocks : list[TextBlock]
        已完成单块推断（style.font_size_pt 已填充）的全部 TextBlock。
    """
    from collections import defaultdict

    by_type: dict = defaultdict(list)
    for block in blocks:
        if block.style and block.style.font_size_pt is not None:
            by_type[block.block_type].append(block)

    for btype, cat_blocks in by_type.items():
        if len(cat_blocks) < 2:
            continue
        _cluster_normalize_font_size(cat_blocks, merge_gap=2.0)
        _cluster_normalize_line_spacing(cat_blocks, merge_gap=0.15)


def _cluster_1d(values: List[float], merge_gap: float):
    """对一维有序数值做贪心聚类，返回 list[list[int]]（每簇的原始索引列表）。

    算法：排序后，相邻值差 ≤ merge_gap 归入同一簇。
    时间 O(n log n)，无外部依赖。
    """
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    clusters: List[List[int]] = [[indexed[0][0]]]
    prev_val = indexed[0][1]
    for idx, val in indexed[1:]:
        if val - prev_val <= merge_gap:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
        prev_val = val
    return clusters


def _weighted_median(values: List[float]) -> float:
    """简单中位数（不做加权，对小样本足够鲁棒）。"""
    s = sorted(values)
    return s[len(s) // 2]


def _cluster_normalize_font_size(
    cat_blocks: List[TextBlock],
    merge_gap: float = 2.0,
) -> None:
    """对同类别块的字号做聚类归一化。

    每簇内统一到 0.5pt 步长的中位数，簇间差异保留。
    """
    sizes = [b.style.font_size_pt for b in cat_blocks]
    clusters = _cluster_1d(sizes, merge_gap)

    for cluster_indices in clusters:
        if len(cluster_indices) < 2:
            continue
        cluster_sizes = [sizes[i] for i in cluster_indices]
        median_fs = _weighted_median(cluster_sizes)
        norm_fs = round(median_fs * 2) / 2.0
        for i in cluster_indices:
            cat_blocks[i].style.font_size_pt = norm_fs


def _cluster_normalize_line_spacing(
    cat_blocks: List[TextBlock],
    merge_gap: float = 0.15,
) -> None:
    """对同类别块的行距做聚类归一化。

    - 有推断值的块 < 50% → 全部清空为 None（使用全局默认值）。
    - 否则：有值的块做聚类归一；无值的块赋予最大簇的中位数。
    """
    has_ls = [(i, b.style.line_spacing)
              for i, b in enumerate(cat_blocks)
              if b.style.line_spacing is not None]
    no_ls = [i for i, b in enumerate(cat_blocks)
             if b.style.line_spacing is None]

    # 有值的块不满一半 → 视为 bbox 噪声，全部清空
    if len(has_ls) < len(cat_blocks) * 0.5:
        for b in cat_blocks:
            b.style.line_spacing = None
        return

    # 对有值的块做聚类归一
    ls_values = [v for _, v in has_ls]
    ls_indices = [i for i, _ in has_ls]
    clusters = _cluster_1d(ls_values, merge_gap)

    # 找到最大簇，用其中位数填充无值的块
    largest_cluster = max(clusters, key=len) if clusters else []
    largest_median = None

    for cluster_idx_list in clusters:
        cluster_ls = [ls_values[ci] for ci in cluster_idx_list]
        median_ls = _weighted_median(cluster_ls)
        norm_ls = round(median_ls * 20) / 20.0  # 0.05 步长
        for ci in cluster_idx_list:
            real_i = ls_indices[ci]
            cat_blocks[real_i].style.line_spacing = norm_ls
        if cluster_idx_list is largest_cluster:
            largest_median = norm_ls

    # 无推断值的块 → 赋予最大簇的中位数
    if largest_median is not None:
        for i in no_ls:
            cat_blocks[i].style.line_spacing = largest_median


# ---------------------------------------------------------------------------
# 字号修正：利用推断的行距反向修正初始字号估算
# ---------------------------------------------------------------------------

def _refine_font_size_from_line_spacing(
    blocks: List[TextBlock],
    mapper: "CoordMapper",
) -> None:
    """使用已推断的行距修正初始字号估算。

    初始估算假设 line_height = font_size × 1.05（text_region）
    或 1.20（bbox）。若实际推断的行距与假设偏差较大，
    说明初始字号需要修正。

    修正公式：font_size_corrected = font_size_initial × (ls_inferred / ls_assumed)

    仅在行距置信度足够高时修正，避免用噪声数据污染字号。
    """
    ls_assumed_text = 1.05  # text_region 使用的除数
    ls_assumed_bbox = 1.20  # bbox 回退使用的除数

    for block in blocks:
        if block.style is None or block.style.line_spacing is None:
            continue
        ls = block.style.line_spacing
        if not (0.95 < ls < 1.55):
            continue

        # 估算初始字号使用的是哪种假设
        has_text_region = any(
            ln.text_region for ln in (block.lines or []) if ln.text_region
        )
        ls_assumed = ls_assumed_text if has_text_region else ls_assumed_bbox

        # 若行距与假设接近（误差 < 15%），说明字号估算基本准确，不需修正
        ratio = ls / ls_assumed
        if abs(ratio - 1.0) < 0.15:
            continue

        # 修正：将字号乘以比例因子，但限制单次修正幅度
        current_fs = block.style.font_size_pt
        if current_fs is None or current_fs <= 0:
            continue

        correction = min(max(ratio, 0.85), 1.18)
        new_fs = current_fs * correction
        new_fs = max(6.0, min(36.0, new_fs))
        new_fs = TextBlock._snap_to_font_grid(new_fs)

        block.style.font_size_pt = new_fs
