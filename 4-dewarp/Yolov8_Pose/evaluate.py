# -*- coding: utf-8 -*-
"""四角点精度评测: 对含 images/ 与 labels/ 的数据集统计角点误差

口径与推理一致(可见区域四边形): GT 与预测都先裁到画面矩形。

用法:
  python -X utf8 evaluate.py --ds <数据集目录> [--weights weights/best.pt] [--no-snap] [--out DIR]

指标:
  点集误差 = 对 4 个角点的所有排列取最小的平均距离(与角点角色无关)
  另按"是否落在画面边框上"把角点分成 贴边角 / 画面内角 两类分别统计
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
from ultralytics import YOLO

from corner_order import reorder_corners
from corner_postprocess import refine_quad, visible_quad

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "best.pt")


def cv_imread(p):
    import cv2
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def load_gt(lp, W, H):
    with open(lp, "r", encoding="utf-8") as f:
        toks = f.readline().split()
    vals = np.array([float(t) for t in toks[1:]])
    return vals[4:].reshape(4, 3)[:, :2] * np.array([W, H])


def set_err4(gt, pred):
    return min(sum(np.linalg.norm(pred[q[i]] - gt[i]) for i in range(4))
               for q in itertools.permutations(range(4))) / 4.0


def best_match(gt, pred):
    return min(itertools.permutations(range(4)),
               key=lambda q: sum(np.linalg.norm(pred[q[i]] - gt[i]) for i in range(4)))


def stat(a, name):
    a = np.array(a, float)
    if not len(a):
        return name + ": 无数据"
    return (f"{name:<12} n={len(a):<5} 中位={np.median(a):7.1f}px "
            f"均值={a.mean():7.1f}px p90={np.percentile(a, 90):7.1f}px "
            f"<=10px={np.mean(a <= 10) * 100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, help="含 images/ 与 labels/ 的数据集目录")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--no-snap", action="store_true", help="关闭边拟合精修(对照用)")
    ap.add_argument("--out", default=None, help="可选: 输出 results.json 的目录")
    args = ap.parse_args()
    snap = not args.no_snap

    model = YOLO(args.weights)
    files = sorted(glob.glob(os.path.join(args.ds, "images", "*.jpg"))
                   + glob.glob(os.path.join(args.ds, "images", "*.png")))
    set_errs, corner_all, corner_in, corner_border, rows = [], [], [], [], []
    n_det = 0
    for ip in files:
        stem = os.path.splitext(os.path.basename(ip))[0]
        img = cv_imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        gt_raw = load_gt(os.path.join(args.ds, "labels", stem + ".txt"), W, H)
        gt = reorder_corners(visible_quad(gt_raw, W, H)[0])
        r = model(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            rows.append({"stem": stem, "detected": False})
            continue
        j = int(r.boxes.conf.argmax())
        pred = reorder_corners(r.keypoints.xy[j].cpu().numpy())
        if snap:
            pred = reorder_corners(refine_quad(img, pred)[0])
        pred = reorder_corners(visible_quad(pred, W, H)[0])
        e = set_err4(gt, pred)
        perm = best_match(gt, pred)
        set_errs.append(e)
        n_det += 1
        for i in range(4):
            d = float(np.linalg.norm(pred[perm[i]] - gt[i]))
            corner_all.append(d)
            (corner_border if min(gt[i][0], W - gt[i][0], gt[i][1], H - gt[i][1]) <= 2
             else corner_in).append(d)
        rows.append({"stem": stem, "detected": True, "conf": float(r.boxes.conf[j]),
                     "set_err_px": round(e, 1),
                     "gt": np.round(gt, 1).tolist(), "pred": np.round(pred, 1).tolist()})

    print(f"数据集: {args.ds}")
    print(f"权重  : {args.weights}   边拟合精修: {'开' if snap else '关'}")
    print(f"检出  : {n_det}/{len(files)}")
    print(stat(set_errs, "点集误差"))
    print(stat(corner_all, "全部角点"))
    print(stat(corner_in, "画面内角点"))
    print(stat(corner_border, "贴边角点"))
    if set_errs:
        e = np.array(set_errs)
        print(f"最差 5 张: " + ", ".join(
            f"{r['stem'][:22]}={r['set_err_px']:.0f}px"
            for r in sorted([x for x in rows if x.get("detected")],
                            key=lambda x: -x["set_err_px"])[:5]))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        json.dump(rows, open(os.path.join(args.out, "results.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("json:", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
