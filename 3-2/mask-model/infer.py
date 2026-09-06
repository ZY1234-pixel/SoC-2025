"""Full-resolution paired watermark-mask inference with overlapping tiles."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

# Allow running this file directly from the project root or from handoff/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import paired_model_from_checkpoint


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


@torch.inference_mode()
def predict_full_resolution(
    model: torch.nn.Module,
    source_image: Image.Image,
    candidate_image: Image.Image,
    device: torch.device,
    tile: int = 512,
    overlap: int = 64,
    batch_size: int = 1,
) -> np.ndarray:
    source = image_tensor(source_image)
    original_height, original_width = source.shape[-2:]
    candidate = image_tensor(
        candidate_image.resize((original_width, original_height), Image.Resampling.BILINEAR)
    )
    pad_right = max(0, tile - original_width)
    pad_bottom = max(0, tile - original_height)
    source = F.pad(source[None], (0, pad_right, 0, pad_bottom), mode="replicate")[0]
    candidate = F.pad(candidate[None], (0, pad_right, 0, pad_bottom), mode="replicate")[0]
    height, width = source.shape[-2:]
    positions = [
        (top, left)
        for top in tile_starts(height, tile, overlap)
        for left in tile_starts(width, tile, overlap)
    ]
    window_1d = torch.hann_window(tile, periodic=False).clamp_min(1e-3)
    window = window_1d[:, None] * window_1d[None, :]
    accumulator = torch.zeros((height, width), dtype=torch.float32)
    divisor = torch.zeros_like(accumulator)

    for start in range(0, len(positions), batch_size):
        current = positions[start : start + batch_size]
        source_batch = torch.stack(
            [source[:, top : top + tile, left : left + tile] for top, left in current]
        ).to(device, non_blocking=device.type == "cuda")
        candidate_batch = torch.stack(
            [candidate[:, top : top + tile, left : left + tile] for top, left in current]
        ).to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device.type, enabled=device.type == "cuda", dtype=torch.float16):
            probabilities = model(source_batch, candidate_batch).sigmoid()[:, 0].float().cpu()
        for probability, (top, left) in zip(probabilities, current):
            accumulator[top : top + tile, left : left + tile] += probability * window
            divisor[top : top + tile, left : left + tile] += window

    probability = accumulator / divisor.clamp_min(1e-6)
    return probability[:original_height, :original_width].numpy()


def main() -> None:
    model_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Paired sliding-window watermark mask inference")
    parser.add_argument("--source", type=Path, required=True, help="Watermarked Source image")
    parser.add_argument("--candidate", type=Path, required=True, help="Clean Candidate PNG")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=model_dir / "weights/watermark_mask.pt",
    )
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Binary-mask threshold; 0.4 is calibrated for the recall-oriented submission profile",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not args.source.is_file() or not args.candidate.is_file():
        parser.error("--source and --candidate must point to readable image files")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.tile <= 0 or not 0 <= args.overlap < args.tile:
        parser.error("--tile must be positive and overlap must satisfy 0 <= overlap < tile")
    if args.batch_size <= 0 or not 0 <= args.threshold <= 1:
        parser.error("batch size must be positive and threshold must be in [0, 1]")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, architecture = paired_model_from_checkpoint(checkpoint)
    model = model.to(device)
    model.eval()

    with Image.open(args.source) as source, Image.open(args.candidate) as candidate:
        probability = predict_full_resolution(
            model,
            source,
            candidate,
            device,
            args.tile,
            args.overlap,
            args.batch_size,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    probability_path = args.output / f"{args.source.stem}_probability.png"
    mask_path = args.output / f"{args.source.stem}_mask.png"
    Image.fromarray(np.round(probability * 65535).astype(np.uint16)).save(probability_path)
    Image.fromarray((probability >= args.threshold).astype(np.uint8) * 255).save(mask_path)
    print(
        f"source={args.source} candidate={args.candidate} -> {mask_path} "
        f"({probability.shape[1]}x{probability.shape[0]}) architecture={architecture}"
    )


if __name__ == "__main__":
    main()
