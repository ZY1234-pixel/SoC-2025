"""Argument parsing and validation without importing the model runtime."""

from pathlib import Path
import argparse
from . import settings


def parse_args(argv=None, *, output_mode=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run InternImage-L + MaskDINO instance inference. With no arguments, "
            "process the 4-dewarp/TEST_dewarp directory."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=settings.DEFAULT_INPUT,
        help=f"Image file or directory (default: {settings.DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=settings.DEFAULT_CHECKPOINT,
        help=f"Model checkpoint (default: {settings.DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument("--config-file", type=Path, default=settings.DEFAULT_CONFIG)
    parser.add_argument(
        "--score-threshold", type=float, default=settings.DEFAULT_SCORE_THRESHOLD
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=settings.DEFAULT_MAX_DETECTIONS,
        help=(
            "Maximum full-resolution instances retained from 100 queries. "
            "The production dataset contains at most three documents."
        ),
    )
    parser.add_argument(
        "--amp",
        dest="use_amp",
        action="store_true",
        help="Use CUDA AMP/FP16 for both MaskDINO passes (default).",
    )
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Use FP32 for both MaskDINO passes.",
    )
    parser.add_argument("--output-dir", type=Path, default=settings.DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-mode",
        choices=("debug", "production"),
        default=settings.DEFAULT_OUTPUT_MODE if output_mode is None else output_mode,
        help=(
            "debug saves only roi_debug and final overlays; production saves "
            "only final full-resolution uint8 instance masks (0/255)."
        ),
    )
    parser.add_argument(
        "--roi-refine",
        dest="roi_refine",
        action="store_true",
        help="Refine each original-image ROI with a second MaskDINO pass (default).",
    )
    parser.add_argument(
        "--no-roi-refine",
        dest="roi_refine",
        action="store_false",
        help="Disable ROI refinement.",
    )
    parser.add_argument(
        "--roi-refine-size",
        type=int,
        default=settings.DEFAULT_ROI_REFINE_SIZE,
        help="Square input size for the second MaskDINO ROI pass.",
    )
    parser.add_argument(
        "--roi-margin",
        type=float,
        default=settings.DEFAULT_ROI_MARGIN,
        help="Fractional margin around the coarse mask bbox.",
    )
    parser.add_argument(
        "--roi-retry-margin",
        type=float,
        default=settings.DEFAULT_ROI_RETRY_MARGIN,
        help="Larger margin used once when a candidate touches an internal ROI edge.",
    )
    parser.add_argument(
        "--roi-candidate-score-threshold",
        type=float,
        default=settings.DEFAULT_ROI_CANDIDATE_SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--roi-min-match-iou", type=float, default=settings.DEFAULT_ROI_MIN_MATCH_IOU
    )
    parser.add_argument(
        "--roi-min-area-ratio", type=float, default=settings.DEFAULT_ROI_MIN_AREA_RATIO
    )
    parser.add_argument(
        "--roi-max-area-ratio", type=float, default=settings.DEFAULT_ROI_MAX_AREA_RATIO
    )
    parser.add_argument(
        "--roi-min-useful-resolution-gain",
        type=float,
        default=settings.DEFAULT_ROI_MIN_USEFUL_RESOLUTION_GAIN,
        help=(
            "Treat an ROI pass below this original-pixel resolution gain as "
            "low-gain when it also expands the mask too much."
        ),
    )
    parser.add_argument(
        "--roi-low-gain-max-expansion",
        type=float,
        default=settings.DEFAULT_ROI_LOW_GAIN_MAX_EXPANSION,
        help=(
            "Reject a low-gain ROI candidate when its area exceeds this "
            "multiple of the coarse mask."
        ),
    )
    parser.add_argument(
        "--roi-local-component-max-ratio",
        type=float,
        default=settings.DEFAULT_ROI_LOCAL_COMPONENT_MAX_RATIO,
        help=(
            "Only changed components smaller than this coarse-mask area ratio "
            "are eligible for local rollback."
        ),
    )
    parser.add_argument(
        "--roi-local-component-max-p95-shift",
        type=float,
        default=settings.DEFAULT_ROI_LOCAL_COMPONENT_MAX_P95_SHIFT,
        help=(
            "Rollback a small changed component when its p95 depth exceeds "
            "this many 1024-reference pixels; it is rescaled to the actual "
            "first-pass input size."
        ),
    )
    parser.add_argument(
        "--roi-local-coarse-edge-advantage",
        type=float,
        default=settings.DEFAULT_ROI_LOCAL_COARSE_EDGE_ADVANTAGE,
        help=(
            "A local rollback additionally requires the coarse boundary's "
            "median image-edge support to exceed the ROI boundary by this ratio."
        ),
    )
    parser.add_argument(
        "--roi-contour-component-max-ratio",
        type=float,
        default=settings.DEFAULT_ROI_CONTOUR_COMPONENT_MAX_RATIO,
        help=(
            "Maximum coarse-mask area ratio for the conservative deep-notch "
            "contour rollback."
        ),
    )
    parser.add_argument(
        "--roi-contour-min-p95-shift",
        type=float,
        default=settings.DEFAULT_ROI_CONTOUR_MIN_P95_SHIFT,
        help=(
            "Minimum deep-notch p95 depth in 1024-reference pixels; it is "
            "rescaled to the actual first-pass input size."
        ),
    )
    parser.add_argument(
        "--roi-contour-min-length-ratio",
        type=float,
        default=settings.DEFAULT_ROI_CONTOUR_MIN_LENGTH_RATIO,
        help=(
            "Require the ROI boundary to be this much longer than the coarse "
            "boundary inside the changed component."
        ),
    )
    parser.add_argument(
        "--roi-contour-min-edge-continuity-gap",
        type=float,
        default=settings.DEFAULT_ROI_CONTOUR_MIN_EDGE_CONTINUITY_GAP,
        help=(
            "Minimum coarse-minus-ROI fraction of locally edge-supported "
            "boundary pixels."
        ),
    )
    parser.add_argument(
        "--roi-contour-coarse-edge-advantage",
        type=float,
        default=settings.DEFAULT_ROI_CONTOUR_COARSE_EDGE_ADVANTAGE,
        help="Minimum median edge-support ratio for contour rollback.",
    )
    parser.add_argument(
        "--roi-contour-edge-magnitude",
        type=float,
        default=settings.DEFAULT_ROI_CONTOUR_EDGE_MAGNITUDE,
        help="LAB-gradient magnitude counted as a continuous image edge.",
    )
    parser.add_argument(
        "--roi-long-edge-component-max-ratio",
        type=float,
        default=settings.DEFAULT_ROI_LONG_EDGE_COMPONENT_MAX_RATIO,
        help="Maximum area ratio for a long, narrow removed boundary strip.",
    )
    parser.add_argument(
        "--roi-long-edge-min-p95-shift",
        type=float,
        default=settings.DEFAULT_ROI_LONG_EDGE_MIN_P95_SHIFT,
        help=(
            "Minimum p95 erosion depth in 1024-reference pixels; it is "
            "rescaled to the actual first-pass input size."
        ),
    )
    parser.add_argument(
        "--roi-long-edge-min-elongation",
        type=float,
        default=settings.DEFAULT_ROI_LONG_EDGE_MIN_ELONGATION,
        help="Minimum long-to-short side ratio of a removed boundary component.",
    )
    parser.add_argument(
        "--roi-long-edge-coarse-edge-advantage",
        type=float,
        default=settings.DEFAULT_ROI_LONG_EDGE_COARSE_EDGE_ADVANTAGE,
        help="Minimum median image-edge advantage required for the coarse line.",
    )
    parser.add_argument(
        "--roi-long-edge-min-coarse-continuity",
        type=float,
        default=settings.DEFAULT_ROI_LONG_EDGE_MIN_COARSE_CONTINUITY,
        help="Minimum edge-supported fraction along the coarse boundary segment.",
    )
    parser.add_argument(
        "--roi-max-instances",
        type=int,
        default=settings.DEFAULT_ROI_MAX_INSTANCES,
        help="Maximum number of score-sorted instances refined per image.",
    )
    parser.add_argument(
        "--low-score-verify-threshold",
        type=float,
        default=settings.DEFAULT_LOW_SCORE_VERIFY_THRESHOLD,
        help=(
            "Only first-pass detections below this score are eligible for "
            "second-pass false-positive removal."
        ),
    )
    parser.add_argument(
        "--low-score-max-mismatch-iou",
        type=float,
        default=settings.DEFAULT_LOW_SCORE_MAX_MISMATCH_IOU,
        help=(
            "Drop a low-score detection only when a second-pass candidate "
            "exists but its best IoU with the coarse mask is below this value."
        ),
    )
    parser.set_defaults(
        roi_refine=settings.DEFAULT_ROI_REFINE, use_amp=settings.DEFAULT_USE_AMP
    )
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Process only the first N images; 0 processes all.",
    )
    args = parser.parse_args(argv)

    if args.roi_refine_size <= 0 or args.roi_refine_size % 32 != 0:
        raise ValueError("--roi-refine-size must be a positive multiple of 32")
    if args.max_detections <= 0:
        raise ValueError("--max-detections must be positive")
    if not 0.0 <= args.roi_margin <= args.roi_retry_margin:
        raise ValueError("Require 0 <= --roi-margin <= --roi-retry-margin")
    if not 0.0 <= args.roi_candidate_score_threshold <= 1.0:
        raise ValueError("--roi-candidate-score-threshold must be in [0, 1]")
    if not 0.0 <= args.roi_min_match_iou <= 1.0:
        raise ValueError("--roi-min-match-iou must be in [0, 1]")
    if (
        args.roi_min_area_ratio <= 0
        or args.roi_max_area_ratio < args.roi_min_area_ratio
    ):
        raise ValueError("Require 0 < min ROI area ratio <= max ROI area ratio")
    if args.roi_min_useful_resolution_gain <= 0.0:
        raise ValueError("--roi-min-useful-resolution-gain must be positive")
    if args.roi_low_gain_max_expansion < 1.0:
        raise ValueError("--roi-low-gain-max-expansion must be at least 1")
    if not 0.0 <= args.roi_local_component_max_ratio <= 1.0:
        raise ValueError("--roi-local-component-max-ratio must be in [0, 1]")
    if args.roi_local_component_max_p95_shift < 0.0:
        raise ValueError("--roi-local-component-max-p95-shift must be non-negative")
    if args.roi_local_coarse_edge_advantage < 1.0:
        raise ValueError("--roi-local-coarse-edge-advantage must be at least 1")
    if not 0.0 <= args.roi_contour_component_max_ratio <= 1.0:
        raise ValueError("--roi-contour-component-max-ratio must be in [0, 1]")
    if args.roi_contour_min_p95_shift < 0.0:
        raise ValueError("--roi-contour-min-p95-shift must be non-negative")
    if args.roi_contour_min_length_ratio < 1.0:
        raise ValueError("--roi-contour-min-length-ratio must be at least 1")
    if not 0.0 <= args.roi_contour_min_edge_continuity_gap <= 1.0:
        raise ValueError("--roi-contour-min-edge-continuity-gap must be in [0, 1]")
    if args.roi_contour_coarse_edge_advantage < 1.0:
        raise ValueError("--roi-contour-coarse-edge-advantage must be at least 1")
    if args.roi_contour_edge_magnitude < 0.0:
        raise ValueError("--roi-contour-edge-magnitude must be non-negative")
    if not 0.0 <= args.roi_long_edge_component_max_ratio <= 1.0:
        raise ValueError("--roi-long-edge-component-max-ratio must be in [0, 1]")
    if args.roi_long_edge_min_p95_shift < 0.0:
        raise ValueError("--roi-long-edge-min-p95-shift must be non-negative")
    if args.roi_long_edge_min_elongation < 1.0:
        raise ValueError("--roi-long-edge-min-elongation must be at least 1")
    if args.roi_long_edge_coarse_edge_advantage < 1.0:
        raise ValueError("--roi-long-edge-coarse-edge-advantage must be at least 1")
    if not 0.0 <= args.roi_long_edge_min_coarse_continuity <= 1.0:
        raise ValueError("--roi-long-edge-min-coarse-continuity must be in [0, 1]")
    if args.roi_max_instances < 0:
        raise ValueError("--roi-max-instances must be non-negative")
    if not 0.0 <= args.low_score_verify_threshold <= 1.0:
        raise ValueError("--low-score-verify-threshold must be in [0, 1]")
    if not 0.0 <= args.low_score_max_mismatch_iou <= 1.0:
        raise ValueError("--low-score-max-mismatch-iou must be in [0, 1]")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be in [0, 1]")
    if args.max_images < 0:
        raise ValueError("--max-images must be non-negative")
    if args.output_mode not in ("debug", "production"):
        raise ValueError("OUTPUT_MODE must be debug or production")
    return args
