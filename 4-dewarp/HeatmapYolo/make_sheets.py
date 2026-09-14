"""把全量预测结果拼成联系表（contact sheet），按误差排序分组

用法: python make_sheets.py --pred_dir <预测图目录> --csv <误差表> [--out 输出目录]
"""
import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np


COLS, ROWS = 6, 5
THUMB_H = 220


def thumb_with_label(img_path, title):
    im = cv2.imread(img_path)
    h, w = im.shape[:2]
    scale = THUMB_H / h
    im = cv2.resize(im, (int(w * scale), THUMB_H))
    bar = np.full((28, im.shape[1], 3), 0, dtype=np.uint8)
    cv2.putText(bar, title, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return np.vstack([im, bar])


def make_sheet(items, out_path, header):
    if not items:
        return
    n = len(items)
    pages = (n + COLS * ROWS - 1) // (COLS * ROWS)
    for pi in range(pages):
        chunk = items[pi * COLS * ROWS : (pi + 1) * COLS * ROWS]
        cells = [thumb_with_label(p[0], p[1]) for p in chunk]
        while len(cells) < COLS * ROWS:
            cells.append(np.full((THUMB_H + 28, 320, 3), 40, dtype=np.uint8))
        grid_w = max(c.shape[1] for c in cells[:COLS])
        rows = []
        for ri in range(ROWS):
            row = [cv2.resize(c, (grid_w, THUMB_H + 28)) for c in cells[ri * COLS : (ri + 1) * COLS]]
            rows.append(np.hstack(row))
        sheet = np.vstack(rows)
        title_bar = np.full((36, sheet.shape[1], 3), 20, dtype=np.uint8)
        cv2.putText(title_bar, f"{header}  page {pi+1}/{pages}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(out_path.replace(".png", f"_p{pi+1}.png"), np.vstack([title_bar, sheet]))
        print("saved:", out_path.replace(".png", f"_p{pi+1}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="含 *_pred.jpg 的目录")
    ap.add_argument("--csv", required=True, help="evaluate.py 输出的 per_image_errors.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else Path(args.pred_dir) / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            errs = [float(r[f"kpt{i}_err_px"]) for i in range(4)]
            mx = max(errs)
            img_path = os.path.join(args.pred_dir, r["image"] + "_pred.jpg")
            if os.path.exists(img_path):
                items.append((img_path, f"{r['image']} max={mx:.1f}px", mx))
    items.sort(key=lambda x: x[2], reverse=True)

    make_sheet([(p, t) for p, t, _ in items[:30]], str(out_dir / "top30_worst.png"), "worst 30")
    bad = [x for x in items if x[2] > 20]
    make_sheet([(p, t) for p, t, _ in bad], str(out_dir / "bad_gt20px.png"), f"maxerr>20px ({len(bad)} imgs)")
    good = [x for x in items if x[2] <= 3][:60]
    make_sheet([(p, t) for p, t, _ in good], str(out_dir / "good_le3px.png"), "good sample (maxerr<=3px)")


if __name__ == "__main__":
    main()
