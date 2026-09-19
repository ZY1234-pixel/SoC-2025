"""水印 Mask 推理：输入「有水印原图」+「去水印候选图」，输出水印 Mask。

Python 调用
-----------
    from infer import load_model, predict
    model, device, info = load_model("weights/watermark_mask.pt")
    probability = predict(model, source_image, candidate_image, device)["probability"]
    mask = probability >= 0.35
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models import model_from_checkpoint

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE / "weights/watermark_mask.pt"

# 检测器的训练裁剪就是围绕这个尺度构造的
TILE = 512
OVERLAP = 64
# 原图比候选图大多少倍时改用候选图的尺度推理
MATCH_SCALE_THRESHOLD = 1.5


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """返回覆盖 ``length`` 的滑窗起点，最后一块贴齐边缘。"""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def image_tensor(image: Image.Image) -> torch.Tensor:
    """PIL 图转 (3,H,W) 的 [0,1] 张量。"""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


@torch.inference_mode()
def _sliding(model, source, candidate, device, tile=TILE, overlap=OVERLAP):
    """对一对已对齐尺寸的图像做 Hann 窗滑窗推理。

    候选图要先缩到原图尺寸再切块。模型内部虽然也会缩放候选图，但那样同一像素坐标
    切出来的两块覆盖的是不同区域，模型看到的一对图是错位的。
    """
    if candidate.size != source.size:
        candidate = candidate.resize(source.size, Image.Resampling.BILINEAR)
    width, height = source.size
    src = image_tensor(source)
    cand = image_tensor(candidate)
    positions = [(top, left)
                 for top in tile_starts(height, tile, overlap)
                 for left in tile_starts(width, tile, overlap)]
    window_1d = torch.hann_window(tile, periodic=False).clamp_min(1e-3)
    window = window_1d[:, None] * window_1d[None, :]

    accumulator = torch.zeros((height, width))
    divisor = torch.zeros((height, width))
    for top, left in positions:
        # 起点本身不会越界；图像比一块还小时贴边裁切，保证切片和窗尺寸一致
        bottom = min(top + tile, height)
        right = min(left + tile, width)
        batch_src = src[:, top:bottom, left:right][None].to(device)
        batch_cand = cand[:, top:bottom, left:right][None].to(device)
        # 门控融合在模型内部完成，这里只取最终输出
        logits = model(batch_src, batch_cand)
        probability = logits.sigmoid()[0, 0].float().cpu()
        h, w = probability.shape
        accumulator[top:bottom, left:right] += probability * window[:h, :w]
        divisor[top:bottom, left:right] += window[:h, :w]
    return (accumulator / divisor.clamp_min(1e-6)).numpy()


@torch.inference_mode()
def predict(model, source_image, candidate_image, device,
            tile=TILE, overlap=OVERLAP, match_scale_threshold=MATCH_SCALE_THRESHOLD):
    """返回一张图对的原分辨率水印概率图。

    原图明显大于候选图时按候选图的尺度推理，再把概率图放回原图分辨率。
    """
    source_image = source_image.convert("RGB")
    candidate_image = candidate_image.convert("RGB")
    mismatch = source_image.width / max(1, candidate_image.width)
    matched = mismatch >= match_scale_threshold

    if matched:
        work_source = source_image.resize(candidate_image.size, Image.Resampling.LANCZOS)
        probability = _sliding(model, work_source, candidate_image, device, tile, overlap)
        # 走一遍 16 位 PNG 保证与落盘结果一致，再放回原图分辨率
        restored = Image.fromarray(np.round(probability * 65535).astype(np.uint16)).resize(
            source_image.size, Image.Resampling.BILINEAR
        )
        probability = np.asarray(restored, dtype=np.float32) / 65535.0
    else:
        probability = _sliding(model, source_image, candidate_image, device, tile, overlap)

    return {
        "probability": probability,
        "matched_scale": bool(matched),
        "scale_mismatch": float(mismatch),
    }


def load_model(checkpoint_path=DEFAULT_CHECKPOINT, device="auto"):
    """加载检测器，返回 ``(model, device, info)``。"""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, architecture = model_from_checkpoint(checkpoint)
    model = model.to(device).eval()
    info = {"architecture": architecture,
            "epoch": checkpoint.get("epoch"),
            "best_iou": checkpoint.get("best_iou")}
    return model, device, info


def save_probability(probability: np.ndarray, path: Path) -> None:
    """存 16 位 PNG，像素值 / 65535 = 概率。"""
    Image.fromarray(np.round(np.clip(probability, 0, 1) * 65535).astype(np.uint16)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="有水印原图")
    parser.add_argument("--candidate", type=Path, required=True, help="去水印候选图")
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="概率阈值。0.35 偏召回，适合「mask 外回退到原图」的融合方式")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model, device, info = load_model(args.checkpoint, args.device)
    print(f"architecture={info['architecture']} epoch={info['epoch']} "
          f"best_iou={info['best_iou']} device={device}", flush=True)

    with Image.open(args.source) as handle:
        source_image = handle.convert("RGB")
    with Image.open(args.candidate) as handle:
        candidate_image = handle.convert("RGB")

    result = predict(model, source_image, candidate_image, device)
    probability = result["probability"]
    mask = probability >= args.threshold

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.source.stem
    save_probability(probability, args.output / f"{stem}_probability.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(args.output / f"{stem}_mask.png")
    print(f"source={source_image.size} candidate={candidate_image.size} "
          f"mismatch={result['scale_mismatch']:.2f}x matched={result['matched_scale']} "
          f"mask={mask.mean() * 100:.2f}%", flush=True)
    print(f"wrote {args.output / (stem + '_probability.png')}", flush=True)
    print(f"wrote {args.output / (stem + '_mask.png')}", flush=True)


if __name__ == "__main__":
    main()
