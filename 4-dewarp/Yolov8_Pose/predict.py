# -*- coding: utf-8 -*-
"""文档四角点推理: 输出"可见区域四边形"

输出口径: 角点 = 目标四边形与画面矩形的交集。
  - 只有边被裁掉 -> 交集就是四边形, 直接输出(角点落在画面边框上)
  - 角被裁掉(交集是五边形以上) -> 取面积最大的内接四边形, 标记 corner_cut=True

用法:
  python -X utf8 predict.py --source <图片或目录> [--out DIR]
  python -X utf8 predict.py --eval_ds <含 images/labels 的目录>   # 评测模式

权重默认放在本目录 weights/best.pt, 可用 --weights 指定。
"""
import argparse
import glob
import itertools
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

from corner_order import reorder_corners
from corner_postprocess import refine_quad, visible_quad, clip_poly

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "best.pt")
NAMES = {0: "double_page_book", 1: "newspaper_poster", 2: "receipt", 3: "screen",
         4: "single_page", 5: "unclassified", 6: "id_card"}


def cv_imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def cv_imwrite(path, img):
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(path)


def predict(model, img, imgsz, conf, snap=True):
    res = model(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    dets = []
    if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
        return dets
    for j in range(len(res.boxes)):
        k = reorder_corners(res.keypoints.xy[j].cpu().numpy())
        k_raw = k.copy()
        if snap:
            # 边拟合精修: 用图像里的真实边缘重新拟合四条边, 修正"缺边时画面内角点被拉跑"
            k = reorder_corners(refine_quad(img, k)[0])
        vis, corner_cut = visible_quad(k, img.shape[1], img.shape[0])
        vis = reorder_corners(vis)
        kc = res.keypoints.conf[j].cpu().numpy() if res.keypoints.conf is not None else np.ones(4)
        dets.append({
            "cls": int(res.boxes.cls[j].item()),
            "conf": float(res.boxes.conf[j].item()),
            "box": res.boxes.xyxy[j].cpu().numpy().tolist(),
            "keypoints_raw": k_raw.tolist(),
            "keypoints_snapped": k.tolist(),
            "keypoints": vis.tolist(),
            "keypoints_conf": kc.tolist(),
            "corner_cut": corner_cut,
        })
    return dets


def annotate(img, dets):
    out = img.copy()
    H, W = img.shape[:2]
    scale = min(1.0, 2200.0 / W)
    if scale < 1.0:
        out = cv2.resize(out, (int(W * scale), int(H * scale)))
    fs = max(0.7, scale * 1.2)
    th = max(2, int(2 * fs))
    for d in dets:
        color = (0, 0, 255) if d["corner_cut"] else (0, 200, 0)
        raw = np.array([[int(p[0] * scale), int(p[1] * scale)] for p in d["keypoints_raw"]], np.int32)
        cv2.polylines(out, [raw.reshape(-1, 1, 2)], True, (160, 160, 160), max(1, th - 1))
        pts = np.array([[int(p[0] * scale), int(p[1] * scale)] for p in d["keypoints"]], np.int32)
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, color, th)
        label = f"cls{d['cls']}:{NAMES.get(d['cls'], '?')} {d['conf']:.2f}"
        cv2.putText(out, label, (pts[:, 0].min(), max(20, pts[:, 1].min() - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, th)
        for i, (x, y) in enumerate(pts):
            cv2.circle(out, (x, y), int(6 * fs), (0, 0, 255), -1)
            cv2.putText(out, f"p{i}", (x + int(6 * fs), y - int(6 * fs)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), th)
    return out


def load_gt(lp, W, H):
    with open(lp, "r", encoding="utf-8") as f:
        toks = f.readline().split()
    vals = np.array([float(t) for t in toks[1:]])
    return vals[4:].reshape(4, 3)[:, :2] * np.array([W, H])


def set_err4(gt, pred):
    return min(
        sum(np.linalg.norm(pred[q[i]] - gt[i]) for i in range(4))
        for q in itertools.permutations(range(4))
    ) / 4.0


def eval_ds(model, ds, imgsz, conf, snap=True):
    """评测: GT 也用同一口径(可见区域四边形)"""
    rows = []
    for ip in sorted(glob.glob(os.path.join(ds, "images", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        img = cv_imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        gt_raw = load_gt(os.path.join(ds, "labels", stem + ".txt"), W, H)
        gt, gt_cut = visible_quad(gt_raw, W, H)
        dets = predict(model, img, imgsz, conf, snap=snap)
        if not dets:
            rows.append({"stem": stem, "detected": False})
            continue
        best = max(dets, key=lambda d: d["conf"])
        pred = np.array(best["keypoints"], float)
        rows.append({
            "stem": stem, "detected": True,
            "err": set_err4(gt, pred),
            "err_ch": float(np.mean(np.linalg.norm(pred - gt, axis=1))),
            "gt_cut": bool(gt_cut), "pred_cut": bool(best["corner_cut"]),
            "conf": best["conf"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--source", nargs="*", default=[])
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "infer"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--eval_ds", default=None)
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--no-snap", action="store_true",
                    help="关闭边拟合精修(对照用; 默认开启)")
    args = ap.parse_args()
    snap = not args.no_snap

    model = YOLO(args.weights)

    if args.eval_ds:
        rows = eval_ds(model, args.eval_ds, args.imgsz, args.conf, snap=snap)
        ok = [r for r in rows if r.get("detected")]
        e = np.array([r["err"] for r in ok])
        print(f"ds={args.eval_ds}  检出 {len(ok)}/{len(rows)}")
        print(f"  可见区域口径 点集误差 med={np.median(e):.1f} mean={e.mean():.1f} "
              f"p90={np.percentile(e,90):.1f}")
        print(f"  GT 本身角被裁的: {sum(1 for r in rows if r.get('gt_cut'))} 张, "
              f"预测被裁角的: {sum(1 for r in ok if r['pred_cut'])} 张")
        if args.json_out:
            json.dump(rows, open(args.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print("  json:", args.json_out)
        return

    files = []
    for s in args.source:
        files += sorted(glob.glob(os.path.join(s, "*.jpg")) + glob.glob(os.path.join(s, "*.png"))) \
            if os.path.isdir(s) else [s]
    os.makedirs(args.out, exist_ok=True)
    rec = {"weights": args.weights, "imgsz": args.imgsz, "convention": "visible_quad",
           "edge_snap": snap, "images": []}
    for ip in files:
        img = cv_imread(ip)
        if img is None:
            print("read fail:", ip)
            continue
        dets = predict(model, img, args.imgsz, args.conf, snap=snap)
        stem = os.path.splitext(os.path.basename(ip))[0]
        cv_imwrite(os.path.join(args.out, f"{stem}_visible.jpg"), annotate(img, dets))
        H, W = img.shape[:2]
        for d in dets:
            d["keypoints_norm"] = (np.array(d["keypoints"]) / np.array([W, H])).tolist()
        rec["images"].append({"file": os.path.basename(ip), "detections": dets})
        print(f"{os.path.basename(ip)}: {len(dets)} det")
    json.dump(rec, open(os.path.join(args.out, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("done:", args.out)


if __name__ == "__main__":
    main()
