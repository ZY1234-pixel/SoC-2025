"""Infer book spine with automatic 90-degree normalization for sideways books.

Usage:
  python infer_spine_auto_orientation.py INPUT_DIR OUTPUT_DIR
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / 'runs_6kpts/pose_1024_orientft/weights/best.pt'


def infer_spine(model, image, mode='auto', device=None, orientation_model=None):
    """Return spine endpoints in original pixel coordinates and the chosen rotation."""
    if mode not in {'auto', 'upright', 'sideways'}:
        raise ValueError(f'Unsupported orientation mode: {mode}')
    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
    if orientation_model is not None and mode == 'auto':
        route = infer_spine(orientation_model, image, mode='auto', device=device)
        if route is not None:
            mode = 'sideways' if route['rotation'] == 'cw90' else 'upright'
    original = model.predict(image, imgsz=1024, conf=0.01, device=device, verbose=False)[0]
    original_found = original.boxes is not None and len(original.boxes) > 0
    original_index = int(original.boxes.conf.argmax().item()) if original_found else None
    original_box = original.boxes.xyxy[original_index].cpu().numpy() if original_found else None
    original_ratio = (float((original_box[2] - original_box[0]) /
                            (original_box[3] - original_box[1])) if original_found else None)
    result = original
    index = original_index
    rotation = 'original'
    if mode != 'upright':
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        candidate = model.predict(rotated, imgsz=1024, conf=0.01, device=device, verbose=False)[0]
        if candidate.boxes is not None and len(candidate.boxes):
            candidate_index = int(candidate.boxes.conf.argmax().item())
            box = candidate.boxes.xyxy[candidate_index].cpu().numpy()
            candidate_ratio = float((box[2] - box[0]) / (box[3] - box[1]))
            candidate_conf = float(candidate.boxes.conf[candidate_index])
            original_conf = float(original.boxes.conf[original_index]) if original_found else 0.0
            choose_rotated = (mode == 'sideways' or not original_found or
                              candidate_ratio > original_ratio + 0.01 or
                              (original_conf < 0.5 and candidate_conf > original_conf + 0.1))
            if choose_rotated:
                result = candidate
                index = candidate_index
                rotation = 'cw90'

    if index is None:
        return None

    endpoints = result.keypoints.xy[index].cpu().numpy()[4:6].copy()
    if rotation == 'cw90':
        original_height = image.shape[0]
        endpoints = np.column_stack([endpoints[:, 1], original_height - 1 - endpoints[:, 0]])
    return {
        'endpoints': endpoints,
        'rotation': rotation,
        'box_ratio': original_ratio,
        'confidence': float(result.boxes.conf[index]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--orientation-weights', type=Path,
                        help='Optional stable checkpoint for direction selection.')
    parser.add_argument('--mode', choices=['auto', 'upright', 'sideways'], default='auto',
                        help='Use auto comparison or force the known book orientation.')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    orientation_model = YOLO(str(args.orientation_weights)) if args.orientation_weights else None
    images = sorted(p for p in args.input_dir.iterdir()
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    rows = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        prediction = infer_spine(model, image, args.mode, orientation_model=orientation_model)
        if prediction is None:
            preview = image.copy()
            cv2.putText(preview, 'NO DETECTION', (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 3)
            cv2.imwrite(str(args.output_dir / image_path.name), preview)
            rows.append([image_path.name, 'miss', '', '', '', '', '', ''])
            continue
        points = np.rint(prediction['endpoints']).astype(int)
        preview = image.copy()
        cv2.line(preview, tuple(points[0]), tuple(points[1]), (255, 0, 0),
                 max(3, round(min(image.shape[:2]) / 300)))
        for point in points:
            cv2.circle(preview, tuple(point), max(5, round(min(image.shape[:2]) / 200)),
                       (0, 0, 255), -1)
        cv2.imwrite(str(args.output_dir / image_path.name), preview)
        rows.append([image_path.name, prediction['rotation'],
                     (round(prediction['box_ratio'], 4)
                      if prediction['box_ratio'] is not None else ''),
                     round(prediction['confidence'], 4),
                     *[round(float(value), 2) for value in prediction['endpoints'].flatten()]])
    with (args.output_dir / 'predictions.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image', 'rotation', 'original_box_w_over_h', 'confidence',
                         'spine_x1', 'spine_y1', 'spine_x2', 'spine_y2'])
        writer.writerows(rows)
    print(f'Processed {len(rows)} images; rotated {sum(row[1] == "cw90" for row in rows)}; '
          f'output: {args.output_dir}')


if __name__ == '__main__':
    main()
