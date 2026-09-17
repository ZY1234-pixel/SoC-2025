"""Batch inference orchestration, per-instance output and timing summary."""

from typing import Any, Dict, List
import cv2
from detectron2.data import detection_utils
import numpy as np
import time
import torch
from .config import load_config
from .geometry import mask_bbox
from .io import input_images, output_file, output_key, write_image
from .preprocessing import DocLetterboxDatasetMapper
from .refinement import refine_instance_with_roi, should_drop_unconfirmed_low_score
from .runtime import filter_and_sort_instances, load_model, synchronize_cuda
from .settings import DEFAULT_COARSE_PIXEL_REFERENCE_SIZE
from .visualization import COLORS, render_mask


def run(args) -> None:
    use_amp = bool(args.use_amp and torch.cuda.is_available())
    input_path = args.input.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    image_paths = list(input_images(input_path))
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No supported images found under: {input_path}")

    print("Inference settings:")
    print(f"  input: {input_path}")
    print(f"  images: {len(image_paths)}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  score threshold: {args.score_threshold}")
    print(f"  max detections: {args.max_detections}")
    print(f"  AMP/FP16: {use_amp}")
    print(f"  ROI refinement: {args.roi_refine}")
    print(f"  output mode: {args.output_mode}")
    if args.roi_refine:
        print("  ROI backend: second MaskDINO pass")
        print(f"  ROI margins: {args.roi_margin}, retry={args.roi_retry_margin}")
        print(f"  ROI size: {args.roi_refine_size}")
        print(
            "  ROI safeguards: "
            f"IoU>={args.roi_min_match_iou}, "
            f"area={args.roi_min_area_ratio}..{args.roi_max_area_ratio}, "
            f"low-gain<{args.roi_min_useful_resolution_gain}x/"
            f"expansion>{args.roi_low_gain_max_expansion}x"
        )
        print(
            "  ROI local fusion: "
            f"component<{args.roi_local_component_max_ratio:.3%}, "
            f"p95 shift>{args.roi_local_component_max_p95_shift:g}px@1024, "
            f"coarse edge>{args.roi_local_coarse_edge_advantage:g}x"
        )
    print(f"  output: {output_dir}")

    process_wall_start = time.perf_counter()
    model_load_start = time.perf_counter()
    opts = list(args.opts) + [
        "MODEL.WEIGHTS",
        str(checkpoint_path),
        "TEST.DETECTIONS_PER_IMAGE",
        str(args.max_detections),
    ]
    cfg = load_config(args.config_file, opts)
    coarse_pixel_scale = float(cfg.INPUT.IMAGE_SIZE) / float(
        DEFAULT_COARSE_PIXEL_REFERENCE_SIZE
    )
    print(
        f"Resolved input sizes: Stage-1={cfg.INPUT.IMAGE_SIZE}, "
        f"ROI Stage-2={args.roi_refine_size}"
    )
    if args.roi_refine:
        print(
            "Effective Stage-1 p95 gates: "
            f"local={args.roi_local_component_max_p95_shift * coarse_pixel_scale:g}px, "
            f"contour={args.roi_contour_min_p95_shift * coarse_pixel_scale:g}px, "
            f"long-edge={args.roi_long_edge_min_p95_shift * coarse_pixel_scale:g}px"
        )
    model = load_model(cfg, checkpoint_path)
    synchronize_cuda()
    model_load_seconds = time.perf_counter() - model_load_start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mapper = DocLetterboxDatasetMapper(
        is_train=False,
        image_size=cfg.INPUT.IMAGE_SIZE,
        image_format=cfg.INPUT.FORMAT,
        random_flip=False,
    )
    output_names = (
        ("roi_debug", "overlays")
        if args.output_mode == "debug"
        else ("instances",)
    )
    directories = {name: output_dir / name for name in output_names}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    image_seconds: List[float] = []
    preprocess_seconds: List[float] = []
    stage1_seconds: List[float] = []
    roi_seconds: List[float] = []
    roi_model_seconds: List[float] = []
    total_attempted_refinements = 0
    total_accepted_refinements = 0

    for image_index, image_path in enumerate(image_paths, start=1):
        image_start = time.perf_counter()
        preprocess_start = image_start
        image_for_model = detection_utils.read_image(
            str(image_path), format=cfg.INPUT.FORMAT
        )
        if cfg.INPUT.FORMAT == "RGB":
            image_bgr = cv2.cvtColor(image_for_model, cv2.COLOR_RGB2BGR)
        elif cfg.INPUT.FORMAT == "BGR":
            image_bgr = np.ascontiguousarray(image_for_model)
        else:
            raise ValueError(
                f"Standalone inference requires RGB or BGR input, got "
                f"{cfg.INPUT.FORMAT!r}"
            )
        height, width = image_for_model.shape[:2]
        record = mapper(
            {
                "file_name": str(image_path.resolve()),
                "height": height,
                "width": width,
                "image_id": 0,
                "_preloaded_image": image_for_model,
            }
        )
        preprocess_seconds.append(time.perf_counter() - preprocess_start)
        synchronize_cuda()
        stage1_start = time.perf_counter()
        with torch.inference_mode(), torch.cuda.amp.autocast(
            enabled=use_amp, cache_enabled=False
        ):
            instances = model([record])[0]["instances"].to("cpu")
        synchronize_cuda()
        stage1_seconds.append(time.perf_counter() - stage1_start)
        instances = filter_and_sort_instances(instances, args.score_threshold)
        refined_overlay = image_bgr.copy() if args.output_mode != "production" else None
        kept_count = 0
        dropped_count = 0
        key = output_key(image_path, input_path)
        accepted_refinements = 0
        attempted_refinements = 0
        for zero_index, (mask_tensor, box_tensor, score_tensor) in enumerate(
            zip(instances.pred_masks, instances.pred_boxes.tensor, instances.scores)
        ):
            instance_id = zero_index + 1
            coarse_mask = mask_tensor.numpy().astype(bool)
            coarse_bbox = [float(value) for value in box_tensor.tolist()]
            score = float(score_tensor)
            color = COLORS[zero_index % len(COLORS)]

            refinement: Dict[str, Any]
            if args.roi_refine and zero_index < args.roi_max_instances:
                attempted_refinements += 1
                synchronize_cuda()
                roi_start = time.perf_counter()
                refined_mask, refinement, debug_image = refine_instance_with_roi(
                    model=model,
                    image_bgr=image_bgr,
                    coarse_mask=coarse_mask,
                    coarse_bbox=coarse_bbox,
                    image_id=image_index * 100 + instance_id,
                    coarse_image_size=cfg.INPUT.IMAGE_SIZE,
                    refine_size=args.roi_refine_size,
                    margins=(args.roi_margin, args.roi_retry_margin),
                    candidate_score_threshold=args.roi_candidate_score_threshold,
                    min_match_iou=args.roi_min_match_iou,
                    min_area_ratio=args.roi_min_area_ratio,
                    max_area_ratio=args.roi_max_area_ratio,
                    min_useful_resolution_gain=args.roi_min_useful_resolution_gain,
                    low_gain_max_expansion=args.roi_low_gain_max_expansion,
                    local_component_max_ratio=args.roi_local_component_max_ratio,
                    local_component_max_p95_shift=(
                        args.roi_local_component_max_p95_shift * coarse_pixel_scale
                    ),
                    local_coarse_edge_advantage=args.roi_local_coarse_edge_advantage,
                    contour_component_max_ratio=(args.roi_contour_component_max_ratio),
                    contour_min_p95_shift=(
                        args.roi_contour_min_p95_shift * coarse_pixel_scale
                    ),
                    contour_min_length_ratio=args.roi_contour_min_length_ratio,
                    contour_min_edge_continuity_gap=(
                        args.roi_contour_min_edge_continuity_gap
                    ),
                    contour_coarse_edge_advantage=(
                        args.roi_contour_coarse_edge_advantage
                    ),
                    contour_edge_magnitude=args.roi_contour_edge_magnitude,
                    long_edge_component_max_ratio=(
                        args.roi_long_edge_component_max_ratio
                    ),
                    long_edge_min_p95_shift=(
                        args.roi_long_edge_min_p95_shift * coarse_pixel_scale
                    ),
                    long_edge_min_elongation=args.roi_long_edge_min_elongation,
                    long_edge_coarse_edge_advantage=(
                        args.roi_long_edge_coarse_edge_advantage
                    ),
                    long_edge_min_coarse_continuity=(
                        args.roi_long_edge_min_coarse_continuity
                    ),
                    use_amp=use_amp,
                    create_debug_image=args.output_mode == "debug",
                )
                synchronize_cuda()
                roi_seconds.append(time.perf_counter() - roi_start)
                roi_model_seconds.append(
                    float(refinement.get("roi_model_seconds", 0.0))
                )
                if refinement["accepted"]:
                    accepted_refinements += 1
                if debug_image is not None:
                    debug_path = output_file(
                        directories["roi_debug"],
                        key,
                        f"_instance_{instance_id:03d}.jpg",
                    )
                    write_image(debug_path, debug_image)
            else:
                refined_mask = coarse_mask.copy()
                refinement = {
                    "attempted": False,
                    "accepted": False,
                    "reason": (
                        "roi_refinement_disabled"
                        if not args.roi_refine
                        else "roi_instance_limit"
                    ),
                }

            instance_suffix = f"_instance_{instance_id:03d}.png"
            if should_drop_unconfirmed_low_score(
                score,
                refinement,
                score_threshold=args.low_score_verify_threshold,
                maximum_mismatch_iou=args.low_score_max_mismatch_iou,
            ):
                refinement["output_action"] = "dropped_low_score_unconfirmed"
                # A rerun may target an existing output directory.  Remove only
                # this rejected instance's previously generated final artifacts
                # so stale masks cannot masquerade as current predictions.
                if args.output_mode == "production":
                    output_file(directories["instances"], key, instance_suffix).unlink(
                        missing_ok=True
                    )
                dropped_count += 1
                continue

            if refined_overlay is not None:
                final_bbox_tuple = mask_bbox(refined_mask)
                final_bbox = (
                    [float(value) for value in final_bbox_tuple]
                    if final_bbox_tuple is not None
                    else coarse_bbox
                )
                render_mask(
                    refined_overlay,
                    refined_mask,
                    final_bbox,
                    score,
                    instance_id,
                    color,
                )
            if args.output_mode == "production":
                mask_path = output_file(directories["instances"], key, instance_suffix)
                write_image(mask_path, refined_mask.astype(np.uint8) * 255)
            kept_count += 1

        if refined_overlay is not None:
            overlay_path = output_file(directories["overlays"], key, ".jpg")
            write_image(overlay_path, refined_overlay)
        image_seconds.append(time.perf_counter() - image_start)
        total_attempted_refinements += attempted_refinements
        total_accepted_refinements += accepted_refinements
        print(
            f"[{image_index}/{len(image_paths)}] {image_path.name}: "
            f"{kept_count} document instance(s), "
            f"dropped {dropped_count}, "
            f"ROI accepted {accepted_refinements}/{attempted_refinements}"
        )
    print("Done.")
    if "overlays" in directories:
        print(f"  Final overlays:   {directories['overlays']}")
    if "roi_debug" in directories:
        print(f"  ROI diagnostics:  {directories['roi_debug']}")
    if "instances" in directories:
        print(f"  Instance masks:   {directories['instances']}")

    process_wall_seconds = time.perf_counter() - process_wall_start
    image_array = np.asarray(image_seconds, dtype=np.float64)
    stage1_total = float(np.sum(stage1_seconds))
    roi_total = float(np.sum(roi_seconds))
    roi_model_total = float(np.sum(roi_model_seconds))
    preprocess_total = float(np.sum(preprocess_seconds))
    image_total = float(np.sum(image_array))
    other_total = max(0.0, image_total - preprocess_total - stage1_total - roi_total)
    print("Timing summary:")
    print(f"  Model load:       {model_load_seconds:.3f} s")
    print(f"  Image processing: {image_total:.3f} s")
    print(f"  Total wall time:  {process_wall_seconds:.3f} s")
    print(f"  Per image mean:   {float(np.mean(image_array)):.3f} s")
    print(f"  Per image P50:    {float(np.percentile(image_array, 50)):.3f} s")
    print(f"  Per image P95:    {float(np.percentile(image_array, 95)):.3f} s")
    print(f"  Throughput:       {len(image_paths) / image_total:.3f} image/s")
    print(f"  Preprocess total: {preprocess_total:.3f} s")
    print(f"  Stage-1 total:    {stage1_total:.3f} s")
    print(f"  ROI total:        {roi_total:.3f} s")
    print(f"    ROI model:      {roi_model_total:.3f} s")
    print(f"  Other/write:      {other_total:.3f} s")
    print(
        f"  ROI accepted:     {total_accepted_refinements}/"
        f"{total_attempted_refinements}"
    )
    if torch.cuda.is_available():
        allocated_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"  Peak GPU allocated: {allocated_gib:.3f} GiB")
        print(f"  Peak GPU reserved:  {reserved_gib:.3f} GiB")
