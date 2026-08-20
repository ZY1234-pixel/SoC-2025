"""Full-resolution watermark mask inference with overlapping tiles."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from models import WatermarkMaskNet


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.inference_mode()
def predict_full_resolution(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    tile: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
) -> np.ndarray:
    """Return an H x W float32 watermark probability map in [0, 1]."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    original_height, original_width = tensor.shape[-2:]
    tensor = F.pad(
        tensor[None],
        (0, max(0, tile - original_width), 0, max(0, tile - original_height)),
        mode="replicate",
    )[0]
    height, width = tensor.shape[-2:]
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
        patches = torch.stack([tensor[:, top : top + tile, left : left + tile] for top, left in current])
        patches = patches.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device.type, enabled=device.type == "cuda", dtype=torch.float16):
            probabilities = model(patches).sigmoid()[:, 0].float().cpu()
        for probability, (top, left) in zip(probabilities, current):
            accumulator[top : top + tile, left : left + tile] += probability * window
            divisor[top : top + tile, left : left + tile] += window

    result = accumulator / divisor.clamp_min(1e-6)
    return result[:original_height, :original_width].numpy()


def input_images(path: Path) -> tuple[list[Path], Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image: {path}")
        return [path], path.parent
    if not path.is_dir():
        raise FileNotFoundError(path)
    images = sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images found in {path}")
    return images, path


def main() -> None:
    model_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Sliding-window watermark mask inference")
    parser.add_argument("--input", type=Path, required=True, help="Input image or directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--checkpoint", type=Path, default=model_dir / "weights/watermark_mask.pt")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4, help="Number of tiles per forward pass")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()
    if args.tile <= 0 or not 0 <= args.overlap < args.tile:
        parser.error("--tile must be positive and --overlap must satisfy 0 <= overlap < tile")
    if args.batch_size <= 0 or not 0 <= args.threshold <= 1:
        parser.error("--batch-size must be positive and --threshold must be in [0, 1]")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    model = WatermarkMaskNet(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()

    images, input_root = input_images(args.input)
    for index, image_path in enumerate(images, 1):
        with Image.open(image_path) as image:
            probability = predict_full_resolution(model, image, device, args.tile, args.overlap, args.batch_size)
        relative = image_path.relative_to(input_root)
        output_dir = args.output / relative.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        probability_path = output_dir / f"{image_path.stem}_probability.png"
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        Image.fromarray(np.round(probability * 65535).astype(np.uint16)).save(probability_path)
        Image.fromarray((probability >= args.threshold).astype(np.uint8) * 255).save(mask_path)
        print(f"[{index}/{len(images)}] {image_path} -> {mask_path} ({probability.shape[1]}x{probability.shape[0]})")


if __name__ == "__main__":
    main()
