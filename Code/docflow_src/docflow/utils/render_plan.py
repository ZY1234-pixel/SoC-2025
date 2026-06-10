"""轻量 render plan 生成。"""

from __future__ import annotations

from collections import Counter

from docflow.model.base import BlockType


def _render_mode(profile: str) -> str:
    if profile in {"single_column", "table_heavy"}:
        return "reflow"
    if profile == "academic_two_col":
        return "native_columns"
    return "grid"


def _block_action(block) -> str:
    if block.block_type == BlockType.TABLE:
        return "table"
    if block.block_type in {BlockType.FIGURE, BlockType.FORMULA, BlockType.EQUATION}:
        return "image_like"
    if block.block_type in {
        BlockType.FIGURE_CAPTION,
        BlockType.TABLE_CAPTION,
        BlockType.FORMULA_CAPTION,
        BlockType.TABLE_FOOTNOTE,
    }:
        return "caption"
    return "paragraph"


def build_render_plan(document, output_format: str = "docx") -> dict:
    profile_counts = Counter()
    strategy_counts = Counter()
    pages_payload = []
    total_blocks = 0

    for page in document.pages:
        attrs = page.attributes or {}
        profile = attrs.get("layout_profile", "generic_complex")
        page_render_mode = _render_mode(profile)
        profile_counts[profile] += 1
        zones_payload = []
        block_type_counts = Counter()
        page_block_count = 0
        page_flow_ids = set()

        for zi, zone in enumerate(page.zones):
            raw_strategy = zone.rendering_strategy
            effective_strategy = "single_col" if page_render_mode == "reflow" else raw_strategy
            strategy_counts[effective_strategy] += 1
            if getattr(zone, "flow_id", ""):
                page_flow_ids.add(zone.flow_id)
            blocks_payload = []
            for block in zone.blocks:
                page_block_count += 1
                total_blocks += 1
                block_type_counts[block.block_type.value] += 1
                blocks_payload.append(
                    {
                        "id": getattr(block, "block_id", ""),
                        "type": block.block_type.value,
                        "bbox": [block.bbox.x1, block.bbox.y1, block.bbox.x2, block.bbox.y2],
                        "col_index": block.col_index,
                        "spanned_cols": list(getattr(block, "spanned_cols", []) or []),
                        "flow_id": str((getattr(block, "attributes", None) or {}).get("flow_id", "")),
                        "action": _block_action(block),
                    }
                )
            zones_payload.append(
                {
                    "zone_index": zi,
                    "flow_id": getattr(zone, "flow_id", ""),
                    "flow_kind": getattr(zone, "flow_kind", ""),
                    "region_id": getattr(zone, "region_id", ""),
                    "region_kind": getattr(zone, "region_kind", ""),
                    "col_count": zone.col_count,
                    "has_spanned": zone.has_spanned,
                    "rendering_strategy": effective_strategy,
                    "raw_rendering_strategy": raw_strategy,
                    "block_count": len(zone.blocks),
                    "blocks": blocks_payload,
                }
            )

        pages_payload.append(
            {
                "page_index": page.index,
                "layout_profile": profile,
                "render_mode": page_render_mode,
                "orientation": page.orientation,
                "size_pt": {"width": page.page_width_pt, "height": page.page_height_pt},
                "rule_stats": dict(attrs.get("rule_stats") or {}),
                "quality_metrics": dict(attrs.get("quality_metrics") or {}),
                "block_type_counts": dict(block_type_counts),
                "block_count": page_block_count,
                "flow_count": len(page_flow_ids),
                "zones": zones_payload,
            }
        )

    return {
        "version": "1.0",
        "output_format": output_format,
        "summary": {
            "page_count": len(document.pages),
            "block_count": total_blocks,
            "layout_profiles": dict(profile_counts),
            "rendering_strategies": dict(strategy_counts),
        },
        "pages": pages_payload,
    }
