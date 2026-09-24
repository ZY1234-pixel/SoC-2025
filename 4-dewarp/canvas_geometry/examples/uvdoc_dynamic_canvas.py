#!/usr/bin/env python3
"""Optional UVDoc example for the model-independent 4-dewarp geometry module."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dewarp_geometry import canvas_from_aspect, estimate_aspect_from_grid
from adapters.uvdoc import UVDocAdapter


def image_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in source.iterdir() if p.suffix.lower() in extensions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--uvdoc-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--long-edge", type=int, default=1400)
    parser.add_argument("--min-confidence", type=float, default=0.15)
    parser.add_argument("--allow-low-confidence", action="store_true")
    args = parser.parse_args()

    paths = image_paths(args.input)
    if not paths:
        parser.error("No supported images found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = UVDocAdapter(args.uvdoc_root, args.checkpoint, args.device)

    all_metadata = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        grid2d, grid3d = adapter.predict(rgb)
        estimate = estimate_aspect_from_grid(grid3d, layout="chw")
        accepted = estimate.valid and (
            estimate.confidence >= args.min_confidence or args.allow_low_confidence
        )
        record = {"input": str(path), **estimate.to_dict(), "accepted": accepted}
        if accepted:
            size = canvas_from_aspect(estimate.aspect_ratio, long_edge=args.long_edge)
            output = adapter.unwarp(rgb, grid2d, size)
            target = args.output_dir / f"{path.stem}_dynamic.png"
            ok = cv2.imwrite(
                str(target), cv2.cvtColor((output * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
            )
            if not ok:
                raise RuntimeError(f"Cannot write image: {target}")
            record.update({"output": str(target), "output_width": size[0], "output_height": size[1]})
        else:
            record["warnings"] = record["warnings"] + ["output_skipped_due_to_low_confidence"]
        all_metadata.append(record)
        print(json.dumps(record, ensure_ascii=False))

    with (args.output_dir / "aspect_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(all_metadata, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
