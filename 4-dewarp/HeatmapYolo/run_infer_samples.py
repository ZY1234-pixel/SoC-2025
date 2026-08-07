"""对指定图片跑推理，叠加 GT 角点，输出可视化结果

用法: python run_infer_samples.py [--weights 权重] [--out 输出目录] <图片路径或文件夹...>
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from heatmap_cls_model import HeatmapClsModel
from heatmap_utils import decode_heatmap


VAL_LBL = r"D:\奔图\deeplabv3p_zzh\YoloV8_Pose\data\val\labels"
WX_LBL = r"D:\奔图\deeplabv3p_zzh\YoloV8_Pose\data\0722WKX\Data\labels"
NAMES = {0: "double_page_book", 1: "newspaper_poster", 2: "receipt", 3: "screen", 4: "single_page", 5: "unclassified"}


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


def find_label(image_path):
    stem = Path(image_path).stem
    for d in (VAL_LBL, WX_LBL):
        p = os.path.join(d, stem + ".txt")
        if os.path.exists(p):
            return p
    return None


def load_gt(label_path):
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        toks = f.readline().strip().split()
    if len(toks) < 17:
        return None, None
    cls = int(toks[0])
    vals = [float(t) for t in toks[1:]]
    kpt = np.array([[vals[4 + i * 3], vals[5 + i * 3]] for i in range(4)])
    return cls, kpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/heatmap_v12_512_aug/last.pt")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--out", default="sample_predictions")
    ap.add_argument("images", nargs="+", help="图片路径或文件夹")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HeatmapClsModel(cfg="configs/yolov8s.yaml", nc=6, num_keypoints=4)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    sd = ckpt["ema"].state_dict() if ckpt.get("ema") else ckpt["model"].state_dict()
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    imgs = []
    for p in args.images:
        if os.path.isdir(p):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                imgs += [str(x) for x in Path(p).glob(ext)]
        else:
            imgs.append(p)
    imgs = sorted(imgs)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"共 {len(imgs)} 张图片")

    for ip in imgs:
        img = cv2.imread(ip)
        if img is None:
            print("无法读取:", ip)
            continue
        H, W = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pad, r, (dw, dh) = letterbox(img_rgb, new_shape=(args.imgsz, args.imgsz))
        t = torch.from_numpy(img_pad).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            cls_logits, hm = model(t.unsqueeze(0).to(device))
        cls_idx = int(cls_logits.argmax(-1).item())
        dec = decode_heatmap(hm)[0].cpu().numpy()
        pred = np.stack([(dec[:, 0] * args.imgsz - dw) / r, (dec[:, 1] * args.imgsz - dh) / r], axis=1)

        lp = find_label(ip)
        gt_cls, gt_norm = load_gt(lp) if lp else (None, None)
        gt = gt_norm * np.array([W, H]) if gt_norm is not None else None

        # 按原图分辨率输出，标注随宽度轻微缩放，圆点保持小尺寸
        out = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = max(0.6, min(1.3, W / 2200.0))
        th = max(2, int(round(2.0 * fs)))
        rad = max(5, min(10, W // 500))
        off = max(10, W // 240)
        cv2.putText(out, f"pred: {NAMES.get(cls_idx, cls_idx)}", (off, int(44 * fs)), font, fs, (255, 0, 255), th)
        cv2.putText(out, "p=pred(红) g=GT(绿) 黄=误差px", (off, int(70 * fs)), font, fs * 0.7, (255, 255, 255), max(1, th - 1))

        pts = pred.astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 0, 255), th)
        for i, (x, y) in enumerate(pred):
            xi, yi = int(round(x)), int(round(y))
            cv2.circle(out, (xi, yi), rad, (0, 0, 255), -1)
            cv2.putText(out, f"p{i}", (xi + off, yi - off), font, fs, (0, 0, 255), th)
        if gt is not None:
            gpts = gt.astype(np.int32)
            cv2.polylines(out, [gpts], True, (0, 255, 0), th)
            for i, (x, y) in enumerate(gt):
                xi, yi = int(round(x)), int(round(y))
                cv2.circle(out, (xi, yi), int(rad * 0.8), (0, 255, 0), -1)
                cv2.putText(out, f"g{i}", (xi + off, yi + int(24 * fs)), font, fs, (0, 255, 0), th)
                err = np.linalg.norm(pred[i] - gt[i])
                cv2.putText(out, f"{err:.1f}px", (xi + off, yi + int(48 * fs)), font, fs * 0.85, (255, 255, 0), th)
        else:
            cv2.putText(out, "no GT", (off, int(96 * fs)), font, fs, (0, 255, 255), th)

        op = os.path.join(args.out, Path(ip).stem + "_pred.jpg")
        cv2.imwrite(op, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("saved:", op)


if __name__ == "__main__":
    main()
