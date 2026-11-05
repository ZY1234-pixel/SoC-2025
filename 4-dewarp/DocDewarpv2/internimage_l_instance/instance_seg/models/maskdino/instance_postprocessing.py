"""Memory-bounded postprocessing for letterboxed instance masks."""

from typing import Any, Dict, Tuple

import torch
from torch.nn import functional as F


def select_topk_query_classes(
    mask_class_logits: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select query/class pairs before any full-resolution mask upsampling."""
    if mask_class_logits.ndim != 2:
        raise ValueError(
            f"Expected [queries, classes] logits, got {tuple(mask_class_logits.shape)}")
    num_queries, num_classes = mask_class_logits.shape
    if num_queries <= 0 or num_classes <= 0:
        raise ValueError("Mask classification logits must contain queries and classes")
    keep = min(max(int(topk), 0), num_queries * num_classes)
    if keep == 0:
        empty_float = mask_class_logits.new_empty((0,))
        empty_long = torch.empty((0,), dtype=torch.long, device=mask_class_logits.device)
        return empty_float, empty_long, empty_long

    probabilities = mask_class_logits.sigmoid().flatten()
    scores, flat_indices = probabilities.topk(keep, sorted=False)
    labels = flat_indices.remainder(num_classes)
    query_indices = torch.div(flat_indices, num_classes, rounding_mode="floor")
    return scores, labels, query_indices


def _validated_letterbox_meta(
    meta: Dict[str, Any],
    letterbox_size: Tuple[int, int],
) -> Tuple[int, int, int, int, int, int]:
    letterbox_height, letterbox_width = map(int, letterbox_size)
    original_height = int(meta["original_height"])
    original_width = int(meta["original_width"])
    resized_height = int(meta["resized_height"])
    resized_width = int(meta["resized_width"])
    top = int(meta["top"])
    left = int(meta["left"])
    values = (
        letterbox_height, letterbox_width, original_height, original_width,
        resized_height, resized_width,
    )
    if any(value <= 0 for value in values):
        raise ValueError(f"Invalid letterbox dimensions: {values}")
    if top < 0 or left < 0:
        raise ValueError(f"Invalid letterbox offsets: top={top}, left={left}")
    if top + resized_height > letterbox_height or left + resized_width > letterbox_width:
        raise ValueError(
            "Letterbox content exceeds its canvas: "
            f"content=({left}, {top}, {resized_width}, {resized_height}), "
            f"canvas=({letterbox_width}, {letterbox_height})")
    return (
        original_height, original_width, resized_height, resized_width, top, left)


def restore_letterbox_masks(
    selected_mask_logits: torch.Tensor,
    letterbox_size: Tuple[int, int],
    meta: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Restore selected masks one at a time to bound peak GPU/CPU memory.

    The old MaskDINO path expanded every query to the original camera
    resolution before top-k selection.  A 3472x4624 image with 100 float masks
    alone needs about 6.4 GB.  This routine receives only the selected queries
    and never materializes more than one full-resolution float mask at a time.
    """
    if selected_mask_logits.ndim != 3:
        raise ValueError(
            f"Expected [instances, height, width] logits, got "
            f"{tuple(selected_mask_logits.shape)}")
    (
        original_height, original_width, resized_height, resized_width, top, left,
    ) = _validated_letterbox_meta(meta, letterbox_size)

    restored_masks = []
    mask_scores = []
    for mask_logit in selected_mask_logits:
        letterbox_logit = F.interpolate(
            mask_logit[None, None], size=letterbox_size,
            mode="bilinear", align_corners=False)[0, 0]
        content_logit = letterbox_logit[
            top:top + resized_height, left:left + resized_width]
        restored_logit = F.interpolate(
            content_logit[None, None], size=(original_height, original_width),
            mode="bilinear", align_corners=False)[0, 0]
        foreground = restored_logit > 0
        if bool(foreground.any()):
            mask_score = restored_logit[foreground].sigmoid().mean()
        else:
            mask_score = restored_logit.new_zeros(())
        restored_masks.append(foreground)
        mask_scores.append(mask_score)

    if not restored_masks:
        return (
            torch.empty(
                (0, original_height, original_width), dtype=torch.bool,
                device=selected_mask_logits.device),
            selected_mask_logits.new_empty((0,)),
        )
    return torch.stack(restored_masks), torch.stack(mask_scores)


def restore_letterbox_boxes(
    selected_boxes_cxcywh: torch.Tensor,
    letterbox_size: Tuple[int, int],
    meta: Dict[str, Any],
) -> torch.Tensor:
    """Map normalized MaskDINO boxes from the letterbox canvas to the source."""
    if selected_boxes_cxcywh.ndim != 2 or selected_boxes_cxcywh.shape[-1] != 4:
        raise ValueError(
            f"Expected [instances, 4] boxes, got {tuple(selected_boxes_cxcywh.shape)}")
    (
        original_height, original_width, resized_height, resized_width, top, left,
    ) = _validated_letterbox_meta(meta, letterbox_size)
    if selected_boxes_cxcywh.shape[0] == 0:
        return selected_boxes_cxcywh.clone()

    letterbox_height, letterbox_width = map(int, letterbox_size)
    center_x, center_y, width, height = selected_boxes_cxcywh.unbind(-1)
    boxes = torch.stack((
        center_x - 0.5 * width,
        center_y - 0.5 * height,
        center_x + 0.5 * width,
        center_y + 0.5 * height,
    ), dim=-1)
    scale = boxes.new_tensor(
        [letterbox_width, letterbox_height, letterbox_width, letterbox_height])
    boxes = boxes * scale

    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    boxes[:, 0::2] = (boxes[:, 0::2] - left) / scale_x
    boxes[:, 1::2] = (boxes[:, 1::2] - top) / scale_y
    boxes[:, 0::2].clamp_(0, original_width)
    boxes[:, 1::2].clamp_(0, original_height)
    return boxes
