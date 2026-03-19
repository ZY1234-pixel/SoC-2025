"""将版面区块按列数拆分为垂直区域以进行阅读顺序排序。

原始代码源自 sorted_layout_boxes (lines 1309-1357)。

“区域 (zone)”是一个应该先于下一个条带读取的水平区块条带。
宽区块（横跨大部分页宽）作为区域分界线；
两个宽区块之间的窄区块先分组，再按 y 轴重叠连通性进一步拆分。
"""

from __future__ import annotations

from typing import Callable, List, TYPE_CHECKING

if TYPE_CHECKING:
    from docflow.model.base import Block

from docflow.utils.constants import Y_OVERLAP_MIN_PX, WIDE_BLOCK_RATIO


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def has_y_overlap(b1: "Block", b2: "Block", min_overlap: int = Y_OVERLAP_MIN_PX) -> bool:
    """判断 *b1* 和 *b2* 在 y 轴上的重叠是否至少达到
    *min_overlap* 像素。"""
    overlap = min(b1.bbox.y2, b2.bbox.y2) - max(b1.bbox.y1, b2.bbox.y1)
    return overlap >= min_overlap


def _split_narrow_batch(
    batch: List["Block"],
    overlap_fn: Callable[["Block", "Block"], bool],
) -> List[List["Block"]]:
    """使用并查集算法将 *batch* 分为连通分量，
    当 *overlap_fn* 返回 True 时两个区块视为连通。

    返回分量列表，每个分量内部按 ``bbox.y1`` 排序，
    外层列表按每个分量的最小 ``bbox.y1`` 排序。
    """
    n = len(batch)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if overlap_fn(batch[i], batch[j]):
                union(i, j)

    # 收集连通分量
    components: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        components.setdefault(root, []).append(i)

    # 构建结果：每个分量内部按 y1 排序，外层按最小 y1 排序
    result: List[List["Block"]] = []
    for indices in components.values():
        group = sorted([batch[i] for i in indices], key=lambda b: b.bbox.y1)
        result.append(group)

    result.sort(key=lambda grp: grp[0].bbox.y1)
    return result


# ------------------------------------------------------------------
# 公开 API
# ------------------------------------------------------------------

def split_into_zones(
    blocks: List["Block"],
    image_width: int,
    width_threshold: float = WIDE_BLOCK_RATIO,
) -> List[List["Block"]]:
    """将 *blocks* 拆分为顺序排列的垂直区域。

    Parameters
    ----------
    blocks:
        页面上的所有版面区块。
    image_width:
        图像宽度（像素），用于区分宽区块与窄区块。
    width_threshold:
        宽区块判定阈值，为 *image_width* 的比例。

    Returns
    -------
    区域列表，每个区域包含一组区块。区域按从上到下排序。
    """
    if not blocks:
        return []

    sorted_blocks = sorted(blocks, key=lambda b: b.bbox.y1)
    wide_limit = image_width * width_threshold

    zones: List[List["Block"]] = []
    narrow_batch: List["Block"] = []

    for blk in sorted_blocks:
        blk_width = blk.bbox.x2 - blk.bbox.x1
        if blk_width > wide_limit:
            # 刷入累积的窄区块
            if narrow_batch:
                zones.extend(_split_narrow_batch(narrow_batch, has_y_overlap))
                narrow_batch = []
            # 宽区块成为单独区域
            zones.append([blk])
        else:
            narrow_batch.append(blk)

    # 刷入剩余的窄区块
    if narrow_batch:
        zones.extend(_split_narrow_batch(narrow_batch, has_y_overlap))

    return zones
