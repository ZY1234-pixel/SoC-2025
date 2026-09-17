from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

from dewarp_core import DewarpError, dewarp_document


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = ROOT / "img"
DEFAULT_MASK_DIR = ROOT / "mask"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_GRID_DIR = ROOT / "grid2d"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _relative_key(path: Path, directory: Path, *, is_mask: bool) -> Path:
    relative = path.relative_to(directory)
    stem = relative.stem
    if is_mask and stem.endswith("_mask"):
        stem = stem[:-5]
    return relative.parent / stem


def collect_files(directory: Path, *, is_mask: bool = False) -> dict[Path, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"directory does not exist: {directory}")
    files: dict[Path, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = _relative_key(path, directory, is_mask=is_mask)
        if key in files:
            raise ValueError(
                f"duplicate sample key '{key.as_posix()}' in {directory}: "
                f"{files[key]} and {path}"
            )
        files[key] = path
    return files


def build_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path, Path]]:
    images = collect_files(image_dir)
    masks = collect_files(mask_dir, is_mask=True)
    if not images:
        raise FileNotFoundError(f"no supported images found in {image_dir}")

    missing = sorted(images.keys() - masks.keys())
    if missing:
        preview = ", ".join(path.as_posix() for path in missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} image(s) have no same-name mask: {preview}{suffix}"
        )
    unused_masks = sorted(masks.keys() - images.keys())
    if unused_masks:
        print(f"Warning: ignoring {len(unused_masks)} mask(s) without a same-name image.")
    return [(key, images[key], masks[key]) for key in sorted(images)]


def _matlab_v73_header() -> bytes:
    description = (
        f"MATLAB 7.3 MAT-file, Platform: CPython {platform.python_version()}, "
        f"Created on: {time.ctime()} HDF5 schema 1.00 ."
    ).encode("ascii")
    if len(description) > 116:
        raise ValueError("MATLAB v7.3 header description is too long")
    return description.ljust(116, b" ") + b"\0" * 8 + b"\0\2IM"


def save_grid2d(path: Path, grid2d: np.ndarray, upsample: float) -> None:
    """Write a UVDoc MATLAB v7.3 grid in (height, width, xy) layout."""
    if grid2d.ndim != 3 or grid2d.shape[2] != 2:
        raise DewarpError(f"invalid grid2d shape: {grid2d.shape}")
    if not np.isfinite(grid2d).all():
        raise DewarpError("grid2d contains non-finite coordinates")
    path.parent.mkdir(parents=True, exist_ok=True)
    matlab_data = np.transpose(np.asarray(grid2d, dtype=np.float64), (2, 1, 0))
    with h5py.File(path, "w", userblock_size=512) as mat_file:
        dataset = mat_file.create_dataset(
            "grid2d",
            data=matlab_data,
            compression="gzip",
            compression_opts=4,
        )
        dataset.attrs["MATLAB_class"] = np.bytes_("double")
        dataset.attrs["Python.Shape"] = np.asarray(grid2d.shape, dtype=np.uint64)
        dataset.attrs["Python.Type"] = np.bytes_("numpy.ndarray")
        dataset.attrs["Python.numpy.Container"] = np.bytes_("ndarray")
        dataset.attrs["Python.numpy.UnderlyingType"] = np.bytes_("float64")
        mat_file.attrs["upsample"] = float(upsample)
    with path.open("r+b") as mat_file:
        mat_file.write(_matlab_v73_header())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten document images using same-name binary masks."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--masks", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-output", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--grid-columns", type=int, default=80)
    parser.add_argument("--grid-rows", type=int, default=60)
    parser.add_argument("--grid-upsample", type=float, default=14.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = build_pairs(args.images.resolve(), args.masks.resolve())
    output_dir = args.output.resolve()
    grid_dir = args.grid_output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for index, (relative_key, image_path, mask_path) in enumerate(pairs, 1):
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            result = dewarp_document(
                image,
                mask,
                grid_columns=args.grid_columns,
                grid_rows=args.grid_rows,
                grid_upsample=args.grid_upsample,
            )
            output_path = (output_dir / relative_key).with_suffix(".png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), result.image):
                raise DewarpError(f"cannot write output: {output_path}")
            grid_path = (grid_dir / relative_key).with_suffix(".mat")
            save_grid2d(grid_path, result.grid2d, args.grid_upsample)
            print(
                f"[{index}/{len(pairs)}] {relative_key.as_posix()}: "
                f"rotation={result.rotation_degrees:.2f} deg, "
                f"output={result.output_size[0]}x{result.output_size[1]}, "
                f"grid={result.grid2d.shape}"
            )
        except (DewarpError, OSError, ValueError) as error:
            sample_name = relative_key.as_posix()
            failures.append((sample_name, str(error)))
            print(
                f"[{index}/{len(pairs)}] {sample_name}: FAILED - {error}",
                file=sys.stderr,
            )

    print(f"Completed: {len(pairs) - len(failures)}/{len(pairs)} succeeded.")
    if failures:
        print("Failures:", file=sys.stderr)
        for stem, message in failures:
            print(f"  {stem}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
