from pathlib import Path
import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "yolo_seg_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "yolo_seg_matplotlib_cache"))

ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = ROOT.parents[1] / "Instance_seg_train"
if TRAIN_ROOT.exists():
    sys.path.insert(0, str(TRAIN_ROOT))

import cv2
import torch
from ultralytics import YOLO


DEFAULT_SOURCE = ROOT / "img"
DEFAULT_WEIGHTS = ROOT / "weights" / "best.pt"
DEFAULT_PROJECT = ROOT / "img_out"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(source):
    source = Path(source)
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_SUFFIXES else []
    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")
    return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def run_name_from_weights(weights):
    weights = Path(weights)
    if weights.parent.name == "weights":
        if weights.parent.parent == ROOT:
            return weights.stem
        return weights.parent.parent.name
    return weights.stem


def relative_output_path(image_path, source_root, output_root, suffix=None):
    image_path = Path(image_path)
    source_root = Path(source_root)
    try:
        rel = image_path.relative_to(source_root if source_root.is_dir() else source_root.parent)
    except ValueError:
        rel = Path(image_path.name)
    if suffix is not None:
        rel = rel.with_suffix(suffix)
    return Path(output_root) / rel


def save_seg_txt(result, txt_path):
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    boxes = result.boxes
    masks = result.masks
    if boxes is None or masks is None or masks.xyn is None:
        txt_path.write_text("", encoding="utf-8")
        return 0

    classes = boxes.cls.detach().cpu().tolist()
    confs = boxes.conf.detach().cpu().tolist()
    lines = []
    for cls, conf, polygon in zip(classes, confs, masks.xyn):
        coords = " ".join(f"{float(v):.6f}" for xy in polygon for v in xy)
        lines.append(f"{int(cls)} {float(conf):.6f} {coords}".rstrip())

    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Run instance segmentation on images under img/.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to trained .pt weights.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Image file or directory to infer.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Output root directory.")
    parser.add_argument("--name", type=str, default=None, help="Output run name. Defaults to weight run name.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--batch", type=int, default=1, help="Prediction batch size.")
    parser.add_argument("--device", default=None, help="Device, e.g. 0 or cpu. Defaults to CUDA if available.")
    parser.add_argument("--no-txt", action="store_true", help="Do not save prediction txt files.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"weights not found: {args.weights}")

    source = args.source.resolve()
    images = collect_images(source)
    if not images:
        raise FileNotFoundError(f"no images found in: {source}")

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    name = args.name or run_name_from_weights(args.weights)
    save_dir = args.project / name
    image_dir = save_dir / "images"
    label_dir = save_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_txt:
        label_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Predict] weights: {args.weights}", flush=True)
    print(f"[Predict] source: {source}", flush=True)
    print(f"[Predict] images: {len(images)}", flush=True)
    print(f"[Predict] output: {save_dir}", flush=True)

    model = YOLO(str(args.weights))
    results = model.predict(
        source=[str(p) for p in images],
        task="segment",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        batch=args.batch,
        device=device,
        retina_masks=True,
        stream=True,
        verbose=False,
    )

    total_instances = 0
    for index, result in enumerate(results, start=1):
        image_path = Path(result.path)
        out_image = relative_output_path(image_path, source, image_dir)
        out_image.parent.mkdir(parents=True, exist_ok=True)
        plotted = result.plot()
        cv2.imwrite(str(out_image), plotted)

        count = 0
        if not args.no_txt:
            out_txt = relative_output_path(image_path, source, label_dir, suffix=".txt")
            count = save_seg_txt(result, out_txt)
            total_instances += count

        print(f"[{index}/{len(images)}] {image_path.name}: {count} instances", flush=True)

    print(f"\n[Predict] annotated images: {image_dir}", flush=True)
    if not args.no_txt:
        print(f"[Predict] prediction labels: {label_dir}", flush=True)
        print(f"[Predict] total instances: {total_instances}", flush=True)


if __name__ == "__main__":
    main()
