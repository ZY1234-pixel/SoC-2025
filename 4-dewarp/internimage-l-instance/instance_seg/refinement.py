"""Second-pass MaskDINO inference and ROI acceptance gates."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import time
import torch
from .fusion import fuse_candidate_with_coarse_mask
from .geometry import (
    boundary_change_metrics,
    candidate_metrics,
    expanded_crop_box,
    mask_bbox,
    select_roi_candidate,
    touches_internal_crop_edge,
)
from .preprocessing import inference_record_from_bgr
from .runtime import filter_and_sort_instances, synchronize_cuda
from .visualization import draw_roi_debug


def refine_instance_with_roi(
    *,
    model,
    image_bgr: np.ndarray,
    coarse_mask: np.ndarray,
    coarse_bbox: Sequence[float],
    image_id: int,
    coarse_image_size: int,
    refine_size: int,
    margins: Sequence[float],
    candidate_score_threshold: float,
    min_match_iou: float,
    min_area_ratio: float,
    max_area_ratio: float,
    min_useful_resolution_gain: float,
    low_gain_max_expansion: float,
    local_component_max_ratio: float,
    local_component_max_p95_shift: float,
    local_coarse_edge_advantage: float,
    contour_component_max_ratio: float,
    contour_min_p95_shift: float,
    contour_min_length_ratio: float,
    contour_min_edge_continuity_gap: float,
    contour_coarse_edge_advantage: float,
    contour_edge_magnitude: float,
    long_edge_component_max_ratio: float,
    long_edge_min_p95_shift: float,
    long_edge_min_elongation: float,
    long_edge_coarse_edge_advantage: float,
    long_edge_min_coarse_continuity: float,
    use_amp: bool,
    create_debug_image: bool,
) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    """Refine one coarse instance and return mask, diagnostics, debug image."""
    height, width = image_bgr.shape[:2]
    tight_bbox = mask_bbox(coarse_mask)
    source_bbox: Sequence[float] = tight_bbox if tight_bbox is not None else coarse_bbox
    global_scale = min(coarse_image_size / width, coarse_image_size / height)
    last_debug: Optional[np.ndarray] = None
    roi_model_seconds = 0.0
    last_diagnostic: Dict[str, Any] = {
        "attempted": True,
        "accepted": False,
        "reason": "no_attempt",
        "backend": "maskdino",
        "supports_detection_verification": True,
    }

    for attempt_index, margin in enumerate(margins, start=1):
        crop_box = expanded_crop_box(
            source_bbox, (height, width), margin_ratio=float(margin)
        )
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        crop_bgr = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        coarse_crop = coarse_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        roi_scale = min(
            refine_size / crop_bgr.shape[1],
            refine_size / crop_bgr.shape[0],
        )
        record = inference_record_from_bgr(crop_bgr, refine_size, image_id)
        synchronize_cuda()
        roi_model_start = time.perf_counter()
        with torch.inference_mode(), torch.cuda.amp.autocast(
            enabled=use_amp, cache_enabled=False
        ):
            roi_instances = model([record])[0]["instances"].to("cpu")
        synchronize_cuda()
        roi_model_seconds += time.perf_counter() - roi_model_start
        roi_instances = filter_and_sort_instances(
            roi_instances, candidate_score_threshold
        )
        candidate_masks = [
            tensor.numpy().astype(bool) for tensor in roi_instances.pred_masks
        ]
        candidate_scores = [float(value) for value in roi_instances.scores.tolist()]
        selected_index, raw_metrics = select_roi_candidate(
            candidate_masks, candidate_scores, coarse_crop
        )

        diagnostic: Dict[str, Any] = {
            "attempted": True,
            "accepted": False,
            "reason": "no_roi_candidate",
            "backend": "maskdino",
            "supports_detection_verification": True,
            "attempt": attempt_index,
            "margin_ratio": float(margin),
            "crop_bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
            "crop_width": crop_x2 - crop_x1,
            "crop_height": crop_y2 - crop_y1,
            "resolution_gain": float(roi_scale / global_scale),
            "candidate_count": len(candidate_masks),
            "roi_model_seconds": roi_model_seconds,
        }
        if selected_index is None:
            if create_debug_image:
                last_debug = draw_roi_debug(
                    crop_bgr, coarse_crop, None, "REJECT: no ROI candidate"
                )
            last_diagnostic = diagnostic
            continue

        raw_selected_mask = candidate_masks[selected_index]
        selected_mask, fusion_metrics = fuse_candidate_with_coarse_mask(
            raw_selected_mask,
            coarse_crop,
            crop_bgr,
            analysis_scale=global_scale,
            max_local_component_ratio=local_component_max_ratio,
            max_local_p95_shift=local_component_max_p95_shift,
            min_coarse_edge_advantage=local_coarse_edge_advantage,
            contour_component_max_ratio=contour_component_max_ratio,
            contour_min_p95_shift=contour_min_p95_shift,
            contour_min_length_ratio=contour_min_length_ratio,
            contour_min_edge_continuity_gap=contour_min_edge_continuity_gap,
            contour_coarse_edge_advantage=contour_coarse_edge_advantage,
            contour_edge_magnitude=contour_edge_magnitude,
            long_edge_component_max_ratio=long_edge_component_max_ratio,
            long_edge_min_p95_shift=long_edge_min_p95_shift,
            long_edge_min_elongation=long_edge_min_elongation,
            long_edge_coarse_edge_advantage=long_edge_coarse_edge_advantage,
            long_edge_min_coarse_continuity=long_edge_min_coarse_continuity,
        )
        metrics = candidate_metrics(selected_mask, coarse_crop)
        boundary_metrics = boundary_change_metrics(
            selected_mask, coarse_crop, analysis_scale=global_scale
        )
        touches_edge = touches_internal_crop_edge(
            selected_mask, crop_box, (height, width)
        )
        diagnostic.update(
            {
                **metrics,
                **boundary_metrics,
                **fusion_metrics,
                "raw_candidate_iou": raw_metrics["iou"],
                "raw_candidate_overlap_min": raw_metrics["overlap_min"],
                "raw_candidate_area_ratio": raw_metrics["area_ratio"],
                "candidate_score": raw_metrics["candidate_score"],
                "match_score": raw_metrics["match_score"],
                "selected_candidate": int(selected_index + 1),
                "touches_internal_crop_edge": bool(touches_edge),
            }
        )

        reasons: List[str] = []
        if metrics["iou"] < min_match_iou:
            reasons.append("low_match_iou")
        if metrics["area_ratio"] < min_area_ratio:
            reasons.append("area_too_small")
        if metrics["area_ratio"] > max_area_ratio:
            reasons.append("area_too_large")
        if (
            diagnostic["resolution_gain"] < min_useful_resolution_gain
            and metrics["area_ratio"] > low_gain_max_expansion
        ):
            reasons.append("low_gain_area_expansion")
        if touches_edge:
            reasons.append("touches_roi_edge")

        if reasons:
            diagnostic["reason"] = "+".join(reasons)
            label = (
                f"REJECT: {diagnostic['reason']} "
                f"IoU={metrics['iou']:.3f} area={metrics['area_ratio']:.3f}"
            )
            if create_debug_image:
                last_debug = draw_roi_debug(crop_bgr, coarse_crop, selected_mask, label)
            last_diagnostic = diagnostic
            # A second, larger ROI is useful only when the current crop is tight.
            if touches_edge and attempt_index < len(margins):
                continue
            break

        refined_mask = np.zeros_like(coarse_mask, dtype=bool)
        refined_mask[crop_y1:crop_y2, crop_x1:crop_x2] = selected_mask
        diagnostic["accepted"] = True
        fused_count = int(fusion_metrics["reverted_component_count"])
        diagnostic["reason"] = (
            "accepted_with_local_fusion" if fused_count else "accepted"
        )
        label = (
            f"ACCEPT IoU={metrics['iou']:.3f} "
            f"area={metrics['area_ratio']:.3f} "
            f"gain={diagnostic['resolution_gain']:.2f}x fused={fused_count}"
        )
        debug = (
            draw_roi_debug(crop_bgr, coarse_crop, selected_mask, label)
            if create_debug_image
            else None
        )
        return refined_mask, diagnostic, debug

    if create_debug_image and last_debug is None:
        x1, y1, x2, y2 = expanded_crop_box(
            source_bbox, (height, width), margin_ratio=float(margins[0])
        )
        crop_bgr = image_bgr[y1:y2, x1:x2]
        coarse_crop = coarse_mask[y1:y2, x1:x2]
        last_debug = draw_roi_debug(
            crop_bgr, coarse_crop, None, "REJECT: ROI refinement failed"
        )
    return coarse_mask.copy(), last_diagnostic, last_debug


def should_drop_unconfirmed_low_score(
    score: float,
    refinement: Dict[str, Any],
    score_threshold: float,
    maximum_mismatch_iou: float,
) -> bool:
    """Conservatively remove a weak coarse detection contradicted by its ROI.

    A missing ROI candidate is not sufficient evidence because a valid but
    difficult document can disappear on the second pass.  Removal therefore
    requires an actual ROI candidate whose best match strongly disagrees with
    the weak first-pass mask.
    """
    return bool(
        score < score_threshold
        and refinement.get("attempted", False)
        and refinement.get("supports_detection_verification", False)
        and not refinement.get("accepted", False)
        and int(refinement.get("candidate_count", 0)) > 0
        and "low_match_iou" in str(refinement.get("reason", ""))
        and float(refinement.get("iou", 1.0)) < maximum_mismatch_iou
    )
