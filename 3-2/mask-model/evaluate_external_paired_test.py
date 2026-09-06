"""Match comparison panels to external images and run paired mask inference."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

try:
    from infer_paired_sliding import predict_full_resolution
except ModuleNotFoundError:
    # 交接目录中的推理脚本
    from handoff.infer import predict_full_resolution
from models import paired_model_from_checkpoint


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CATEGORIES = ("日常图片水印", "电子文档水印", "试卷水印")


def images(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def thumbnail(image: Image.Image, size: int = 96) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def split_comparison(path: Path) -> tuple[Image.Image, Image.Image]:
    """Split a comparison panel in its actual layout.

    The exported comparison images use a horizontal layout for landscape
    panels and a vertical layout for portrait panels.  Always splitting at
    half width turns portrait candidates into narrow strips.
    """
    comparison = Image.open(path).convert("RGB")
    if comparison.width >= comparison.height:
        middle = comparison.width // 2
        boxes = ((0, 0, middle, comparison.height), (middle, 0, comparison.width, comparison.height))
    else:
        middle = comparison.height // 2
        boxes = ((0, 0, comparison.width, middle), (0, middle, comparison.width, comparison.height))
    return tuple(comparison.crop(box) for box in boxes)


def match_category(source_root: Path, comparison_root: Path, candidate_root: Path) -> list[dict]:
    source_paths = images(source_root)
    comparison_paths = images(comparison_root)
    panels = [split_comparison(path) for path in comparison_paths]
    rotations = (0, 90, 180, 270)
    cost = np.zeros((len(comparison_paths), len(source_paths)), dtype=np.float32)
    best_rotation = np.zeros_like(cost, dtype=np.int16)
    source_panel_index = np.zeros_like(cost, dtype=np.int8)

    for row, comparison_panels in enumerate(panels):
        for column, source_path in enumerate(source_paths):
            source = Image.open(source_path).convert("RGB")
            panel_choices = []
            for panel_index, panel in enumerate(comparison_panels):
                panel_thumbnail = thumbnail(panel)
                choices = []
                for angle in rotations:
                    rotated = source.rotate(angle, expand=True) if angle else source
                    choices.append(float(np.mean(np.abs(thumbnail(rotated) - panel_thumbnail))))
                rotation_index = int(np.argmin(choices))
                panel_choices.append((choices[rotation_index], rotations[rotation_index], panel_index))
            cost[row, column], best_rotation[row, column], source_panel_index[row, column] = min(panel_choices)

    matched_rows, matched_columns = linear_sum_assignment(cost)
    records = []
    candidate_root.mkdir(parents=True, exist_ok=True)
    for row, column in zip(matched_rows, matched_columns):
        comparison_path = comparison_paths[row]
        source_path = source_paths[column]
        angle = int(best_rotation[row, column])
        comparison_panels = panels[row]
        candidate_panel = comparison_panels[1 - int(source_panel_index[row, column])]
        candidate = candidate_panel.rotate(-angle, expand=True) if angle else candidate_panel
        candidate_path = candidate_root / f"{source_path.stem}.png"
        candidate.save(candidate_path)
        records.append(
            {
                "id": source_path.stem,
                "category": source_root.name,
                "source": str(source_path.resolve()),
                "candidate": str(candidate_path.resolve()),
                "comparison": str(comparison_path.resolve()),
                "rotation": angle,
                "match_cost": float(cost[row, column]),
            }
        )
    records.sort(key=lambda record: record["id"])
    matched_sources = {record["source"] for record in records}
    for source_path in source_paths:
        if str(source_path.resolve()) not in matched_sources:
            print(f"UNMATCHED {source_path}")
    return records


def save_probability(probability: np.ndarray, path: Path) -> None:
    Image.fromarray(np.round(probability * 65535).astype(np.uint16)).save(path)


def save_overlay(source: Image.Image, probability: np.ndarray, threshold: float, path: Path) -> None:
    rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    mask = probability >= threshold
    overlay = rgb.copy()
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.array([1.0, 0.0, 0.0])
    Image.fromarray(np.round(overlay.clip(0, 1) * 255).astype(np.uint8)).save(path, quality=92)


def main() -> None:
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate paired model on provided external data")
    parser.add_argument("--data-root", type=Path, default=project / "0-数据")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "runs/paired_mask_clwd_photo_v2_bnfix/best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "runs/paired_mask_clwd_photo_v2_bnfix/external_test",
    )
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, architecture = paired_model_from_checkpoint(checkpoint)
    model = model.to(device).eval()
    print(f"architecture={architecture}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)

    all_records = []
    for category in CATEGORIES:
        records = match_category(
            args.data_root / category,
            args.data_root / "对比结果" / category,
            args.output / "candidates" / category,
        )
        print(f"matched {category}: {len(records)}")
        all_records.extend(records)

    results = []
    for index, record in enumerate(all_records, 1):
        source = Image.open(record["source"]).convert("RGB")
        candidate = Image.open(record["candidate"]).convert("RGB")
        probability = predict_full_resolution(
            model,
            source,
            candidate,
            device,
            tile=args.tile,
            overlap=args.overlap,
            batch_size=args.batch_size,
        )
        category_output = args.output / record["category"]
        category_output.mkdir(parents=True, exist_ok=True)
        probability_path = category_output / f"{record['id']}_probability.png"
        mask_path = category_output / f"{record['id']}_mask.png"
        overlay_path = category_output / f"{record['id']}_overlay.jpg"
        save_probability(probability, probability_path)
        Image.fromarray((probability >= args.threshold).astype(np.uint8) * 255).save(mask_path)
        save_overlay(source, probability, args.threshold, overlay_path)
        result = {
            **record,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "max_probability": float(probability.max()),
            "p99_probability": float(np.percentile(probability, 99)),
            "mask_fraction": float((probability >= args.threshold).mean()),
            "probability": str(probability_path.resolve()),
            "mask": str(mask_path.resolve()),
            "overlay": str(overlay_path.resolve()),
        }
        results.append(result)
        print(
            f"[{index}/{len(all_records)}] {record['category']}/{record['id']} "
            f"match={record['match_cost']:.4f} area={result['mask_fraction'] * 100:.3f}%",
            flush=True,
        )

    manifest_path = args.output / "results.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = {}
    for category in CATEGORIES:
        category_results = [result for result in results if result["category"] == category]
        summary[category] = {
            "images": len(category_results),
            "median_mask_fraction": float(
                np.median([result["mask_fraction"] for result in category_results])
            ),
            "median_match_cost": float(
                np.median([result["match_cost"] for result in category_results])
            ),
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
