#!/usr/bin/env python3
"""Estimate page aspect ratio from a model-exported NumPy 3D grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dewarp_geometry import canvas_from_aspect, estimate_aspect_from_grid


def json_safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("grid", type=Path, help=".npy file containing HxWx3, 3xHxW or 1x3xHxW")
    parser.add_argument("--layout", choices=("auto", "hwc", "chw"), default="auto")
    parser.add_argument("--valid-mask", type=Path, help="optional HxW NumPy boolean mask")
    parser.add_argument("--long-edge", type=int, default=1400)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    grid = np.load(args.grid, allow_pickle=False)
    mask = np.load(args.valid_mask, allow_pickle=False) if args.valid_mask else None
    estimate = estimate_aspect_from_grid(grid, layout=args.layout, valid_mask=mask)
    result = estimate.to_dict()
    if estimate.valid:
        width, height = canvas_from_aspect(estimate.aspect_ratio, long_edge=args.long_edge)
        result.update({"output_width": width, "output_height": height})
    encoded = json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False)
    print(encoded)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
