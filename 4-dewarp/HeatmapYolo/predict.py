import argparse
import os

import cv2
import numpy as np
import torch

from corner_postprocess import postprocess_corners
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


def letterbox(im, new_shape=(512, 512), color=(114, 114, 114)):
    """等比例缩放 + 居中灰边填充，与训练时保持一致"""
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


def main():
    ap = argparse.ArgumentParser(description="热力图角点检测 - 单张推理")
    ap.add_argument("--image", required=True, help="输入图片路径")
    ap.add_argument("--weights", default="weights/heatmap_v14_512_aug/last.pt")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--out", default="predict_result.jpg")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = HeatmapClsModel(cfg="configs/yolov8s.yaml", nc=6, num_keypoints=4)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = ckpt["ema"].state_dict() if ckpt.get("ema") else ckpt["model"].state_dict()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    img_raw = cv2.imread(args.image)
    if img_raw is None:
        raise FileNotFoundError(f"无法读取图片: {args.image}")
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_padded, r, (dw, dh) = letterbox(img_rgb, new_shape=(args.imgsz, args.imgsz))

    img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        cls_logits, kpt_heatmap = model(img_tensor)

    cls_idx = cls_logits.argmax(-1).item()
    cls_name = NAMES.get(cls_idx, "Unknown")
    coords_norm = decode_heatmap(kpt_heatmap)[0].cpu().numpy()

    # 反算回原图坐标
    coords_abs = np.zeros_like(coords_norm)
    coords_abs[:, 0] = (coords_norm[:, 0] * args.imgsz - dw) / r
    coords_abs[:, 1] = (coords_norm[:, 1] * args.imgsz - dh) / r
    # 置信度门控 + 几何一致性后处理（防飞出画面 / 角点扎堆 / 连线交叉）
    conf = kpt_heatmap[0].cpu().numpy().max(axis=(1, 2))
    coords_abs = postprocess_corners(coords_abs, conf, img_raw.shape[1], img_raw.shape[0])

    print(f"类别: {cls_name}")
    print(f"角点坐标 (原图像素):\n{coords_abs.round(2)}")

    # 可视化
    for i, (x, y) in enumerate(coords_abs):
        xi, yi = int(x), int(y)
        cv2.circle(img_raw, (xi, yi), 6, (0, 0, 255), -1)
        cv2.putText(img_raw, str(i), (xi + 8, yi - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    pts = coords_abs.astype(np.int32)
    cv2.polylines(img_raw, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.putText(img_raw, f"Class: {cls_name}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
    cv2.imwrite(args.out, img_raw)
    print("结果已保存:", args.out)


if __name__ == "__main__":
    main()
