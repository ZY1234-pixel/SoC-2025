"""Evaluate WatermarkMaskNet with the official SLBR CLWD protocol."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models import WatermarkMaskNet


class CLWDTestDataset(Dataset):
    def __init__(self, root: Path, size: int):
        self.root = root
        self.size = size
        self.images = sorted((root / "Watermarked_image").glob("*.jpg"))
        if not self.images:
            raise FileNotFoundError(f"No test images found in {root}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = Image.open(self.root / "Mask" / f"{image_path.stem}.png").convert("L")
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        mask_array = np.asarray(mask, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(image_array).permute(2, 0, 1), torch.from_numpy(mask_array)[None]


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> dict[str, float]:
    macro_iou = macro_f1 = true_positive = false_positive = false_negative = 0.0
    count = 0
    for images, masks in loader:
        probabilities = model(images.to(device, non_blocking=True)).sigmoid()
        predicted = probabilities > threshold
        truth = masks.to(device, non_blocking=True) > 0.1  # SLBR CLWD loader threshold.
        tp = (predicted & truth).sum((1, 2, 3)).float()
        fp = (predicted & ~truth).sum((1, 2, 3)).float()
        fn = (~predicted & truth).sum((1, 2, 3)).float()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        macro_iou += (tp / (tp + fp + fn + 1e-5)).sum().item()
        macro_f1 += (2 * precision * recall / (precision + recall + 1e-6)).sum().item()
        true_positive += tp.sum().item()
        false_positive += fp.sum().item()
        false_negative += fn.sum().item()
        count += images.shape[0]

    return {
        "images": count,
        "macro_iou": macro_iou / count,
        "macro_f1": macro_f1 / count,
        "micro_iou": true_positive / (true_positive + false_positive + false_negative),
        "micro_f1": 2 * true_positive / (2 * true_positive + false_positive + false_negative),
    }


def main() -> None:
    model_dir = Path(__file__).resolve().parent
    project = model_dir.parent
    parser = argparse.ArgumentParser(description="Evaluate with the official SLBR CLWD mask protocol")
    parser.add_argument("--checkpoint", type=Path, default=model_dir / "weights/watermark_mask.pt")
    parser.add_argument("--data", type=Path, default=project / "0-数据/data（CLWD 格式）/clwd_crop/test")
    parser.add_argument("--output", type=Path, default=model_dir / "runs/mask_v8_mask_only/slbr_protocol.json")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WatermarkMaskNet(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    loader = DataLoader(
        CLWDTestDataset(args.data, args.size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    result = {
        "protocol": "SLBR official CLWD" if args.size == 256 else "SLBR-style diagnostic",
        "checkpoint": str(args.checkpoint),
        "size": args.size,
        "prediction_threshold": args.threshold,
        "gt_threshold": 0.1,
        **evaluate(model, loader, device, args.threshold),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
