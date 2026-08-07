import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from heatmap_cls_model import HeatmapClsModel
from heatmap_utils import decode_heatmap


NAMES = {
    0: "double_page_book",
    1: "newspaper_poster",
    2: "receipt",
    3: "screen",
    4: "single_page",
    5: "unclassified",
}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


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


def infer_one(model, img_path, device, imgsz):
    img_raw = cv2.imread(img_path)
    if img_raw is None:
        return None, None, None
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_padded, r, (dw, dh) = letterbox(img_rgb, new_shape=(imgsz, imgsz))
    img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        cls_logits, kpt_heatmap = model(img_tensor)
    cls_idx = cls_logits.argmax(-1).item()
    coords_norm = decode_heatmap(kpt_heatmap)[0].cpu().numpy()
    coords_abs = np.zeros_like(coords_norm)
    coords_abs[:, 0] = (coords_norm[:, 0] * imgsz - dw) / r
    coords_abs[:, 1] = (coords_norm[:, 1] * imgsz - dh) / r

    img_out = img_raw.copy()
    for i, (x, y) in enumerate(coords_abs):
        xi, yi = int(x), int(y)
        cv2.circle(img_out, (xi, yi), 6, (0, 0, 255), -1)
        cv2.putText(img_out, str(i), (xi + 8, yi - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    pts = coords_abs.astype(np.int32)
    cv2.polylines(img_out, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.putText(img_out, f"Class: {NAMES.get(cls_idx, cls_idx)}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
    return NAMES.get(cls_idx, cls_idx), coords_abs, img_out


def main():
    ap = argparse.ArgumentParser(description="热力图角点检测 - 批量推理")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", default="predict_results")
    ap.add_argument("--weights", default="weights/heatmap_v12_512_aug/last.pt")
    ap.add_argument("--imgsz", type=int, default=512)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = HeatmapClsModel(cfg="configs/yolov8s.yaml", nc=6, num_keypoints=4)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = ckpt["ema"].state_dict() if ckpt.get("ema") else ckpt["model"].state_dict()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    img_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(IMAGE_EXTS)]
    print(f"共 {len(img_files)} 张图片")
    for filename in sorted(img_files):
        img_path = os.path.join(args.input_dir, filename)
        cls_name, coords, img_out = infer_one(model, img_path, device, args.imgsz)
        if img_out is None:
            continue
        out_path = os.path.join(args.output_dir, Path(filename).stem + "_result.jpg")
        cv2.imwrite(out_path, img_out)
        print(f"{filename}: {cls_name} {coords.round(1).tolist()}")

    print("批量推理完成，结果保存在:", args.output_dir)


if __name__ == "__main__":
    main()
