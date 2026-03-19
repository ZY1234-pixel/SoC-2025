"""DocFlow JSON 输入数据的校验与规范化。"""

from __future__ import annotations

from typing import List, Tuple

from docflow.schema.models import BLOCK_CATEGORIES, SPAN_CATEGORIES, RELATION_TYPES


# ═══════════════════════════════════════════════════════════════════════════
# 校验
# ═══════════════════════════════════════════════════════════════════════════

def validate_input(data: dict) -> Tuple[bool, List[str]]:
    """校验 JSON 输入是否符合 v2.0 模式。

    返回 ``(is_valid, errors)``，其中 *errors* 是可读的
    错误消息列表（有效时为空）。
    """
    errors: List[str] = []

    if not isinstance(data, dict):
        return False, ["Input must be a dict."]

    # version
    if "version" not in data:
        errors.append("Missing 'version' field.")

    # pages
    pages = data.get("pages")
    if pages is None:
        errors.append("Missing 'pages' field.")
        return len(errors) == 0, errors

    if not isinstance(pages, list):
        errors.append("'pages' must be a list.")
        return len(errors) == 0, errors

    for page_idx, page in enumerate(pages):
        prefix = f"pages[{page_idx}]"

        if not isinstance(page, dict):
            errors.append(f"{prefix}: must be a dict.")
            continue

        # 必填页面字段
        for key in ("page_index", "width", "height", "blocks"):
            if key not in page:
                errors.append(f"{prefix}: missing '{key}'.")

        blocks = page.get("blocks")
        if blocks is not None:
            if not isinstance(blocks, list):
                errors.append(f"{prefix}.blocks: must be a list.")
            else:
                for blk_idx, block in enumerate(blocks):
                    blk_prefix = f"{prefix}.blocks[{blk_idx}]"
                    _validate_block(block, blk_prefix, errors)

        # 校验关系（仅 v2）
        relations = page.get("relations")
        if relations is not None:
            if not isinstance(relations, list):
                errors.append(f"{prefix}.relations: must be a list.")
            else:
                block_ids = set()
                if blocks and isinstance(blocks, list):
                    for b in blocks:
                        if isinstance(b, dict) and "id" in b:
                            block_ids.add(b["id"])

                for rel_idx, rel in enumerate(relations):
                    rel_prefix = f"{prefix}.relations[{rel_idx}]"
                    _validate_relation(rel, rel_prefix, block_ids, errors)

    return len(errors) == 0, errors


def _validate_block(block: dict, prefix: str,
                    errors: List[str]) -> None:
    """校验单个区块字典。"""
    if not isinstance(block, dict):
        errors.append(f"{prefix}: must be a dict.")
        return

    if "id" not in block:
        errors.append(f"{prefix}: missing 'id'.")
    if "category" not in block:
        errors.append(f"{prefix}: missing 'category'.")
    else:
        cat = block["category"]
        if cat not in BLOCK_CATEGORIES:
            errors.append(
                f"{prefix}: unknown category '{cat}'."
            )

    # bbox — required in both versions
    bbox = block.get("bbox")
    if bbox is None:
        errors.append(f"{prefix}: missing 'bbox'.")
    elif not isinstance(bbox, list) or len(bbox) != 4:
        errors.append(f"{prefix}: 'bbox' must be a list of 4 numbers.")
    else:
        for i, v in enumerate(bbox):
            if not isinstance(v, (int, float)):
                errors.append(f"{prefix}: bbox[{i}] is not a number.")

    # spans (v2)
    spans = block.get("spans")
    if spans is not None and isinstance(spans, list):
        for s_idx, span in enumerate(spans):
            s_prefix = f"{prefix}.spans[{s_idx}]"
            if not isinstance(span, dict):
                errors.append(f"{s_prefix}: must be a dict.")
                continue
            if "id" not in span:
                errors.append(f"{s_prefix}: missing 'id'.")
            if "category" not in span:
                errors.append(f"{s_prefix}: missing 'category'.")
            elif span["category"] not in SPAN_CATEGORIES:
                errors.append(
                    f"{s_prefix}: unknown span category '{span['category']}'."
                )
            s_bbox = span.get("bbox")
            if s_bbox is None:
                errors.append(f"{s_prefix}: missing 'bbox'.")
            elif not isinstance(s_bbox, list) or len(s_bbox) != 4:
                errors.append(f"{s_prefix}: 'bbox' must be a list of 4 numbers.")


def _validate_relation(rel: dict, prefix: str,
                       block_ids: set, errors: List[str]) -> None:
    """校验单个关系字典。"""
    if not isinstance(rel, dict):
        errors.append(f"{prefix}: must be a dict.")
        return

    for key in ("type", "source_id", "target_id"):
        if key not in rel:
            errors.append(f"{prefix}: missing '{key}'.")

    rel_type = rel.get("type")
    if rel_type is not None and rel_type not in RELATION_TYPES:
        errors.append(f"{prefix}: unknown relation type '{rel_type}'.")

    if block_ids:
        for key in ("source_id", "target_id"):
            ref = rel.get(key)
            if ref is not None and ref not in block_ids:
                errors.append(
                    f"{prefix}: {key} '{ref}' not found in page blocks."
                )


# ═══════════════════════════════════════════════════════════════════════════
# 规范化
# ═══════════════════════════════════════════════════════════════════════════

def normalize_input(data: dict) -> dict:
    """规范化输入，填充默认值（confidence、id、order）。

    返回 *新* 字典（不修改原始输入）。
    """
    out = dict(data)
    out.setdefault("version", "2.0")

    if "pages" not in out or not isinstance(out["pages"], list):
        out.setdefault("pages", [])
        return out

    new_pages = []
    for page in out["pages"]:
        new_page = dict(page)
        raw_blocks = new_page.get("blocks", [])
        new_blocks = []
        for idx, block in enumerate(raw_blocks):
            new_block = dict(block)

            # 确保 id
            new_block.setdefault("id", f"blk_{idx}")

            # 类别转小写
            if "category" in new_block:
                new_block["category"] = str(new_block["category"]).lower()

            # 确保 bbox 为浮点数列表
            if "bbox" in new_block and isinstance(new_block["bbox"], list):
                new_block["bbox"] = [float(v) for v in new_block["bbox"]]

            # 默认 confidence
            new_block.setdefault("confidence", 1.0)
            new_block["confidence"] = float(new_block["confidence"])

            # 默认 order
            if new_block.get("order") is None:
                new_block["order"] = idx

            new_blocks.append(new_block)

        new_page["blocks"] = new_blocks
        new_pages.append(new_page)

    out["pages"] = new_pages
    return out
