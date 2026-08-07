import argparse
import os

import torch
import torch.nn as nn

from heatmap_cls_model import HeatmapClsModel


class ExportWrapper(nn.Module):
    """把自定义模型包成纯 tensor 输入输出，供 torch.jit.trace"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats = self.model._forward_features(x)
        return self.model.model[-1](feats)


def main():
    ap = argparse.ArgumentParser(description="导出 TorchScript 模型")
    ap.add_argument("--weights", default="weights/heatmap_v12_512_aug/last.pt")
    ap.add_argument("--out", default="weights/heatmap_v12_512_aug/heatmap_v12_512_aug.torchscript.pt")
    ap.add_argument("--imgsz", type=int, default=512)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HeatmapClsModel(cfg="configs/yolov8s.yaml", nc=6, num_keypoints=4)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = ckpt["ema"].state_dict() if ckpt.get("ema") else ckpt["model"].state_dict()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    wrap = ExportWrapper(model).to(device).eval()
    dummy = torch.rand(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(wrap, dummy, check_trace=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.jit.save(traced, args.out)
    print("saved:", args.out, os.path.getsize(args.out), "bytes")

    # 验证 TorchScript 与 PyTorch 输出一致
    with torch.no_grad():
        cls_e, hm_e = wrap(dummy)
        cls_j, hm_j = traced(dummy)
    print("cls max diff:", float((cls_e - cls_j).abs().max()))
    print("heatmap max diff:", float((hm_e - hm_j).abs().max()))


if __name__ == "__main__":
    main()
