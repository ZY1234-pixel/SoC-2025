#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint


PACKAGE_DIR = Path(__file__).resolve().parent
SEGMENTATION_DIR = PACKAGE_DIR / "segmentation"
sys.path.insert(0, str(SEGMENTATION_DIR))

import mmseg_custom  # noqa: E402,F401,F403
from mmseg.models import build_segmentor  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_INPUT = PACKAGE_DIR / "img"
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "docseg_internimage_l_1024.py"
DEFAULT_CHECKPOINT_DIR = PACKAGE_DIR / "checkpoints"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"


def checkpoint_epoch(path):
    match = re.search(r"epoch_(\d+)\.pth$", path.name)
    return int(match.group(1)) if match else -1


def resolve_checkpoint(checkpoint):
    if checkpoint:
        return Path(checkpoint)

    candidates = sorted(DEFAULT_CHECKPOINT_DIR.glob("best_hd95_epoch_*.pth"))
    if candidates:
        return max(candidates, key=lambda path: (checkpoint_epoch(path), path.stat().st_mtime))

    candidates = sorted(DEFAULT_CHECKPOINT_DIR.glob("*.pth"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)

    raise FileNotFoundError(f"No checkpoint found in {DEFAULT_CHECKPOINT_DIR}")


def collect_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def preprocess(image, target):
    h, w = image.shape[:2]
    scale = min(target / w, target / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_LANCZOS4 if scale <= 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (nw, nh), interpolation=interpolation)

    canvas = np.full((target, target, 3), 128, dtype=np.uint8)
    left = (target - nw) // 2
    top = (target - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    meta = dict(h=h, w=w, top=top, left=left, nh=nh, nw=nw)
    return canvas, meta


def restore_mask(mask_1024, meta):
    crop = mask_1024[
        meta["top"]:meta["top"] + meta["nh"],
        meta["left"]:meta["left"] + meta["nw"],
    ]
    return cv2.resize(crop, (meta["w"], meta["h"]), interpolation=cv2.INTER_NEAREST)


def build_model(config, checkpoint, device):
    cfg = mmcv.Config.fromfile(str(config))
    cfg.model.pretrained = None
    if "backbone" in cfg.model:
        cfg.model.backbone.init_cfg = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model)
    load_checkpoint(model, str(checkpoint), map_location="cpu")
    model.to(device).eval()
    return model


def infer_one(model, image_bgr, threshold, device, target=1024):
    image_1024, meta = preprocess(image_bgr, target)
    rgb = cv2.cvtColor(image_1024, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.inference_mode():
        feats = model.extract_feat(tensor)
        logit = model.decode_head.forward(feats)
        logit = torch.nn.functional.interpolate(
            logit, size=(target, target), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(logit)[0, 0].detach().cpu().numpy()

    mask_1024 = (prob >= threshold).astype(np.uint8) * 255
    return restore_mask(mask_1024, meta), mask_1024


def save_overlay(image_bgr, mask, path):
    color = np.zeros_like(image_bgr)
    color[:, :, 1] = 255
    alpha = (mask > 0).astype(np.float32)[:, :, None] * 0.35
    overlay = (image_bgr * (1.0 - alpha) + color * alpha).astype(np.uint8)
    cv2.imwrite(str(path), overlay)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run InternImage-L document-mask inference on img/ by default.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Image file or directory.")
    parser.add_argument("--checkpoint", default=None, help="Defaults to latest best_hd95 checkpoint.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-overlays", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir)
    checkpoint = resolve_checkpoint(args.checkpoint)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    image_paths = collect_images(input_path)
    if args.max_samples is not None:
        image_paths = image_paths[:args.max_samples]
    if not image_paths:
        raise RuntimeError(f"No images found under {input_path}")

    print(f"Input: {input_path}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Images: {len(image_paths)}")
    print(f"Device: {device}")
    print(f"Threshold: {args.threshold}")

    model = build_model(args.config, checkpoint, device)
    masks_dir = output_dir / "masks"
    masks_1024_dir = output_dir / "masks_1024"
    overlays_dir = output_dir / "overlays"

    for idx, path in enumerate(image_paths, 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[skip] unreadable image: {path}")
            continue

        mask, mask_1024 = infer_one(
            model, image, args.threshold, device, target=args.target_size)

        try:
            rel = path.relative_to(input_path if input_path.is_dir() else input_path.parent)
        except ValueError:
            rel = Path(path.name)
        stem = rel.with_suffix("")

        mask_path = masks_dir / stem.with_name(stem.name + "_mask.png")
        mask_1024_path = masks_1024_dir / stem.with_name(stem.name + "_mask_1024.png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_1024_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(mask_1024_path), mask_1024)

        if not args.no_overlays:
            overlay_path = overlays_dir / stem.with_name(stem.name + "_overlay.jpg")
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            save_overlay(image, mask, overlay_path)

        print(f"[{idx}/{len(image_paths)}] {rel} -> {mask_path}")

    print("Done.")


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    main()
