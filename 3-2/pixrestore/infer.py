"""Restore one image or a directory of images."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tile", type=int)
    parser.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()
    from restore import SUFFIXES, WatermarkRemover, folder_tile_indices, tile_plan
    from PIL import Image

    source = args.input.expanduser().resolve()
    if source.is_file() and source.suffix.lower() in SUFFIXES:
        paths = [source]
    elif source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    else:
        parser.error("Input must be an image or image directory.")
    if not paths:
        parser.error("No supported images found.")
    if len({p.stem for p in paths}) != len(paths):
        parser.error("Input filenames must have distinct stems.")
    model = WatermarkRemover(device=args.device, precision=args.precision, steps=args.steps,
                             seed=args.seed, tile=args.tile, overlap=args.overlap)
    indices = folder_tile_indices(paths, model.tile, model.overlap)
    output = args.output.expanduser().resolve()
    (output / "restored").mkdir(parents=True, exist_ok=True)
    for number, path in enumerate(paths, 1):
        with Image.open(path) as source_image:
            _, boxes = tile_plan(*source_image.size, model.tile, model.overlap)
        tile_ids = [indices[f"{path.stem}__{top:06d}_{left:06d}.png"] for top, left in boxes]
        results = model.restore(path, tile_indices=tile_ids)
        results["restored"].save(output / "restored" / f"{path.stem}.png")
        print(f"[{number}/{len(paths)}] {path.name}", flush=True)


if __name__ == "__main__":
    main()
