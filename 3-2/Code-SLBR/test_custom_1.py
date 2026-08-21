# 原图（任意大小，高清）
#      ↓
# 缩小到 512×512（给模型用）
#      ↓
# 模型去水印
#      ↓
# 放大回 原图大小
#      ↓
# 非水印区 = 原图（超级清晰）
# 水印区   = 去水印结果
#      ↓
# 保存：高清去水印图（和原图一样大）
import argparse
import torch
import os
import cv2
import numpy as np

torch.backends.cudnn.benchmark = True

import datasets as datasets
import src.models as models
from options import Options
import torch.nn.functional as F


def tensor2np(x, isMask=False):
    if isMask:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = ((x.cpu().detach())) * 255
    else:
        x = x.cpu().detach()
        mean = 0
        std = 1
        x = (x * std + mean) * 255

    return x.numpy().transpose(0, 2, 3, 1).astype(np.uint8)


def save_output_single(img_np, save_dir, img_fn):
    img_fn = os.path.split(img_fn)[-1]
    out_fn = os.path.join(save_dir, f"{os.path.splitext(img_fn)[0]}{os.path.splitext(img_fn)[1]}")
    cv2.imwrite(out_fn, img_np)


def preprocess_image(img_np):
    patch = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    patch = torch.from_numpy(patch.transpose(2, 0, 1)[np.newaxis, ...])
    return patch


def process_full_resolution(img_path, model, device, down_size=512):
    # 1. 读取全尺寸原图
    img = cv2.imread(img_path)
    assert img is not None, f"无法读取图片: {img_path}"
    orig_h, orig_w = img.shape[:2]

    # 2. 预处理
    img_tensor = preprocess_image(img).to(device).float()
    orig_tensor = preprocess_image(img).to(device).float()

    # 3. 下采样到模型输入尺寸
    img_tensor_small = F.interpolate(img_tensor, size=(down_size, down_size), mode='bilinear', align_corners=False)

    # 4. 模型推理
    with torch.no_grad():
        outputs = model.model(img_tensor_small)

        # 解包嵌套输出
        def get_tensor(x):
            while isinstance(x, (list, tuple)):
                x = x[0]
            return x

        imoutput_small = get_tensor(outputs[0])
        immask_all_small = get_tensor(outputs[1])

        # ===================== 核心修复 =====================
        # 把【去水印图 + mask】都上采样回原图大小
        # ====================================================
        imoutput_full = F.interpolate(imoutput_small, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        mask_full = F.interpolate(immask_all_small, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        # 现在尺寸完全一致，可以计算
        imfinal = imoutput_full * mask_full + model.norm(orig_tensor) * (1 - mask_full)

    # 转回高清图像
    result_np = cv2.cvtColor(tensor2np(imfinal)[0], cv2.COLOR_RGB2BGR)
    return result_np


def test_dataloder(img_path):
    loaders = []
    for fn in os.listdir(img_path):
        if fn.startswith('.'):
            continue
        if not (fn.lower().endswith(('.jpg', '.jpeg', '.png'))):
            continue
        full_path = os.path.join(img_path, fn)
        loaders.append(full_path)
    return loaders


def main(args):
    args.result_dir = args.test_dir + "_global_results"
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    Machine = models.__dict__[args.models](datasets=(None, None), args=args)
    model = Machine
    model.model.eval()
    device = model.device

    print("==> 测试【全局水印清除 + 全分辨率】模型")
    print(f"输入目录: {args.test_dir}")
    print(f"输出目录: {args.result_dir}")
    print(f"全局推理尺寸: {args.crop_size}")

    img_paths = test_dataloder(args.test_dir)

    for idx, img_path in enumerate(img_paths):
        print(f"\n处理第 {idx + 1}/{len(img_paths)} 张: {os.path.basename(img_path)}")
        final_img = process_full_resolution(
            img_path,
            model,
            device,
            down_size=args.crop_size
        )
        save_output_single(final_img, args.result_dir, img_path)

    print("\n✅ 所有图片处理完成（全分辨率 + 全局信息）！")


if __name__ == '__main__':
    parser = Options().init(argparse.ArgumentParser(description='WaterMark Removal（全局推理 + 全分辨率输出）'))
    main(parser.parse_args())

