"""角点检测验证脚本：输出逐角点像素误差指标 + 报告（JSON/CSV/直方图）

用法:
  python evaluate.py --weights <权重路径> [--imgsz 512] [--out 输出目录]
  python evaluate.py --compare <旧权重> <新权重> [--imgsz 512]

输出:
  val_metrics.json       全部数值指标
  val_metrics.csv        汇总表
  per_image_errors.csv   每张图逐角点误差（便于查 badcase）
  error_histogram.png    误差分布直方图
"""
import argparse
import csv
import glob
import json
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from heatmap_cls_model import HeatmapClsModel
from heatmap_utils import decode_heatmap


VAL_IMG = r"D:\奔图\deeplabv3p_zzh\YoloV8_Pose\data\val\images"
VAL_LBL = r"D:\奔图\deeplabv3p_zzh\YoloV8_Pose\data\val\labels"
NAMES = {0: "double_page_book", 1: "newspaper_poster", 2: "receipt", 3: "screen", 4: "single_page", 5: "unclassified"}
THRESHOLDS = (1, 2, 3, 5, 10)


def letterbox(im, new_shape=(512, 512), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def load_gt(path):
    """读取 YOLO pose label，返回 (cls, kpt_norm(4,2))"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) < 17:
                continue
            cls = int(toks[0])
            vals = [float(t) for t in toks[1:]]
            kpt = np.array([[vals[4 + i * 3], vals[5 + i * 3]] for i in range(4)])
            return cls, kpt
    return None, None


def evaluate(weights_path, imgsz, device, val_img_dir, val_lbl_dir, max_images=None):
    model = HeatmapClsModel(cfg="configs/yolov8s.yaml", nc=6, num_keypoints=4)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    sd = ckpt["ema"].state_dict() if ckpt.get("ema") else ckpt["model"].state_dict()
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    errs, clss, sizes, gts = [], [], [], []
    ch_stats = np.zeros((4, 3))
    files = sorted(glob.glob(os.path.join(val_lbl_dir, "*.txt")))
    if max_images:
        files = files[:max_images]
    stems = []
    for lp in files:
        stem = Path(lp).stem
        ip = os.path.join(val_img_dir, stem + ".jpg")
        if not os.path.exists(ip):
            continue
        cls, gt_norm = load_gt(lp)
        if cls is None:
            continue
        img_raw = cv2.imread(ip)
        if img_raw is None:
            continue
        H, W = img_raw.shape[:2]
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_pad, r, (dw, dh) = letterbox(img_rgb, new_shape=(imgsz, imgsz))
        t = torch.from_numpy(img_pad).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            _, hm = model(t.unsqueeze(0).to(device))
        hm = hm[0].cpu().numpy()
        for k in range(4):
            ch_stats[k, 0] += float((hm[k] ** 2).sum())
            ch_stats[k, 1] += float(hm[k].max())
            ch_stats[k, 2] += 1
        dec = decode_heatmap(torch.from_numpy(hm[None]).to(device))[0].cpu().numpy()
        pred = np.stack([(dec[:, 0] * imgsz - dw) / r, (dec[:, 1] * imgsz - dh) / r], axis=1)
        gt_abs = gt_norm * np.array([W, H])
        errs.append(np.linalg.norm(pred - gt_abs, axis=1))
        clss.append(cls)
        sizes.append((W, H))
        gts.append(gt_norm)
        stems.append(stem)
    return np.array(errs), np.array(clss), np.array(sizes), np.array(gts), ch_stats, stems


def _stats(e):
    return {
        "mean_px": round(float(e.mean()), 2),
        "median_px": round(float(np.median(e)), 2),
        "p90_px": round(float(np.percentile(e, 90)), 2),
        "p95_px": round(float(np.percentile(e, 95)), 2),
        "max_px": round(float(e.max()), 2),
        "acc": {f"le{thr}px": round(float((e <= thr).mean()), 4) for thr in THRESHOLDS},
    }


def summarize(errs, clss, gts):
    per_kpt = {}
    all_err = errs.reshape(-1)
    for k in range(4):
        per_kpt[f"kpt{k}"] = _stats(errs[:, k])
        # 贴边 vs 内部：GT 归一化坐标贴近图像边缘的角点
        g = gts[:, k, :]
        edge = (g[:, 0] < 0.05) | (g[:, 0] > 0.95) | (g[:, 1] < 0.05) | (g[:, 1] > 0.95)
        interior = ~edge
        if edge.sum() > 0:
            per_kpt[f"kpt{k}"]["near_edge"] = {"n": int(edge.sum()), **_stats(errs[:, k][edge])}
        if interior.sum() > 0:
            per_kpt[f"kpt{k}"]["interior"] = {"n": int(interior.sum()), **_stats(errs[:, k][interior])}
    per_kpt["overall"] = _stats(all_err)

    top = errs[:, :2].reshape(-1)
    bottom = errs[:, 2:].reshape(-1)
    per_kpt["top_avg(kpt0,1)"] = {"mean_px": round(float(top.mean()), 2),
                                  "median_px": round(float(np.median(top)), 2)}
    per_kpt["bottom_avg(kpt2,3)"] = {"mean_px": round(float(bottom.mean()), 2),
                                     "median_px": round(float(np.median(bottom)), 2)}

    per_cls = {}
    for c in np.unique(clss):
        mask = clss == c
        e = errs[mask].reshape(-1)
        per_cls[NAMES.get(int(c), str(c))] = {
            "n_images": int(mask.sum()),
            "mean_px": round(float(e.mean()), 2),
            "median_px": round(float(np.median(e)), 2),
        }
    return per_kpt, per_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, help="权重路径（单模型模式）")
    ap.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"), help="对比两个权重")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--val_imgs", default=VAL_IMG)
    ap.add_argument("--val_labels", default=VAL_LBL)
    ap.add_argument("--max_images", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.compare:
        targets = [(Path(p).parent.parent.name, p) for p in args.compare]
    elif args.weights:
        targets = [(Path(args.weights).parent.parent.name, args.weights)]
    else:
        ap.error("需要 --weights 或 --compare")

    results = {}
    compare_mode = len(targets) == 2
    for tag, w in targets:
        print(f"\n===== 评估 {tag} =====")
        errs, clss, sizes, gts, ch_stats, stems = evaluate(
            w, args.imgsz, device, args.val_imgs, args.val_labels, args.max_images
        )
        per_kpt, per_cls = summarize(errs, clss, gts)
        results[tag] = {"per_keypoint": per_kpt, "per_class": per_cls}
        print(f"images: {len(errs)}")
        for k in range(4):
            m = per_kpt[f"kpt{k}"]
            print(
                f"  kpt{k}: mean={m['mean_px']} median={m['median_px']} p90={m['p90_px']}  "
                f"le3px={m['acc']['le3px']:.1%} le5px={m['acc']['le5px']:.1%}"
            )
        print("通道健康(平均每图 sum2/max):",
              [f"ch{k}: {ch_stats[k,0]/ch_stats[k,2]:.2f}/{ch_stats[k,1]/ch_stats[k,2]:.2f}" for k in range(4)])

        out_dir = Path(args.out) if args.out else Path(w).parent.parent / "val_metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        suf = f"_{tag}" if compare_mode else ""
        with open(out_dir / f"per_image_errors{suf}.csv", "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(["image", "class", "kpt0_err_px", "kpt1_err_px", "kpt2_err_px", "kpt3_err_px", "img_w", "img_h"])
            for i in range(len(errs)):
                wr.writerow(
                    [stems[i], NAMES.get(int(clss[i]), int(clss[i]))]
                    + [round(float(v), 2) for v in errs[i]]
                    + [sizes[i][0], sizes[i][1]]
                )
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
        for k, ax in enumerate(axes):
            ax.hist(np.clip(errs[:, k], 0, 50), bins=50, color="steelblue")
            ax.set_title(f"kpt{k}  median={np.median(errs[:, k]):.1f}px")
            ax.set_xlabel("error (px)")
        fig.suptitle(f"{tag} corner error distribution (N={len(errs)})")
        fig.tight_layout()
        fig.savefig(out_dir / f"error_histogram{suf}.png", dpi=120)
        plt.close(fig)

    out_dir = Path(args.out) if args.out else Path(targets[-1][1]).parent.parent / "val_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, res in results.items():
        for k in range(4):
            m = res["per_keypoint"][f"kpt{k}"]
            rows.append([tag, f"kpt{k}", m["mean_px"], m["median_px"], m["p90_px"], m["p95_px"],
                         m["acc"]["le3px"], m["acc"]["le5px"]])
        m = res["per_keypoint"]["overall"]
        rows.append([tag, "overall", m["mean_px"], m["median_px"], m["p90_px"], "-",
                     m["acc"]["le3px"], m["acc"]["le5px"]])
    with open(out_dir / "val_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "keypoint", "mean_px", "median_px", "p90_px", "p95_px", "acc_le3px", "acc_le5px"])
        wr.writerows(rows)
    with open(out_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存到 {out_dir}")
    if compare_mode:
        t1, t2 = targets
        print("\n===== 对比 =====")
        for k in range(4):
            a = results[t1[0]]["per_keypoint"][f"kpt{k}"]
            b = results[t2[0]]["per_keypoint"][f"kpt{k}"]
            print(f"kpt{k}: {t1[0]} mean={a['mean_px']} median={a['median_px']}  ->  "
                  f"{t2[0]} mean={b['mean_px']} median={b['median_px']}")


if __name__ == "__main__":
    main()
