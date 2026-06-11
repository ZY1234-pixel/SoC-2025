import argparse
import torch
import os
import cv2
import numpy as np
import math
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

import datasets as datasets
import src.models as models
from options import Options


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


def save_final_result(img_np, save_dir, img_fn):
    """只保存最终拼接好的去水印结果"""
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_name = os.path.split(img_fn)[-1]
    out_path = os.path.join(save_dir, img_name)
    cv2.imwrite(out_path, img_bgr)


def split_image(img_np, crop_size, overlap=32):
    """
    将大图分块切割
    :param img_np: H,W,3 RGB 原图数组
    :param crop_size: 单块尺寸
    :param overlap: 块之间重叠像素（避免拼接缝）
    :return: 块列表、位置信息、原图尺寸
    """
    H, W, _ = img_np.shape
    stride = crop_size - overlap

    # 计算行列块数
    num_h = math.ceil((H - crop_size) / stride) + 1
    num_w = math.ceil((W - crop_size) / stride) + 1

    patches = []
    pos_list = []

    for i in range(num_h):
        for j in range(num_w):
            y = i * stride
            x = j * stride
            # 边界修正
            if y + crop_size > H:
                y = H - crop_size
            if x + crop_size > W:
                x = W - crop_size

            patch = img_np[y:y + crop_size, x:x + crop_size, :]
            patches.append(patch)
            pos_list.append((x, y))

    return patches, pos_list, (H, W)


def merge_patches(patches_np, pos_list, ori_h, ori_w, crop_size, overlap=32):
    """分块结果拼接回原图"""
    stride = crop_size - overlap
    canvas = np.zeros((ori_h, ori_w, 3), dtype=np.float32)
    weight_map = np.zeros((ori_h, ori_w, 3), dtype=np.float32)

    for patch, (x, y) in zip(patches_np, pos_list):
        h, w = patch.shape[:2]
        canvas[y:y+h, x:x+w, :] += patch
        weight_map[y:y+h, x:x+w, :] += 1.0

    # 重叠区域加权平均
    canvas = np.divide(canvas, weight_map, where=weight_map != 0)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return canvas


def preprocess_patch(patch, crop_size):
    """单块图像转模型输入 tensor"""
    img_J = patch.astype(np.float32) / 255.
    img_J = torch.from_numpy(img_J.transpose(2, 0, 1)[np.newaxis, ...])
    img_J = F.interpolate(img_J, size=(crop_size, crop_size), mode='bilinear')
    return img_J


def process_single_image(img_path, model, crop_size, device, overlap=32):
    """单张图片：分块推理 + 拼接"""
    # 读取原图
    img_raw = cv2.imread(img_path)
    assert img_raw is not None, f"读取失败: {img_path}"
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # 尺寸小于等于crop_size：直接整张处理
    if H <= crop_size and W <= crop_size:
        img_tensor = preprocess_patch(img_rgb, crop_size).to(device).float()
        with torch.no_grad():
            outputs = model.model(img_tensor)
            imoutput, immask_all, _ = outputs
            imoutput = imoutput[0]
            immask = immask_all[0]
            imfinal = imoutput * immask + model.norm(img_tensor) * (1 - immask)
        res_np = tensor2np(imfinal)[0]
        return res_np

    # 大图：分块处理
    patches, pos_list, (ori_h, ori_w) = split_image(img_rgb, crop_size, overlap)
    patch_results = []

    with torch.no_grad():
        for patch in patches:
            inp = preprocess_patch(patch, crop_size).to(device).float()
            outputs = model.model(inp)
            imoutput, immask_all, _ = outputs
            imoutput = imoutput[0]
            immask = immask_all[0]
            imfinal = imoutput * immask + model.norm(inp) * (1 - immask)
            # 转回原图块尺寸
            patch_res = tensor2np(imfinal)[0]
            patch_results.append(patch_res)

    # 拼接所有块
    merged_img = merge_patches(patch_results, pos_list, ori_h, ori_w, crop_size, overlap)
    return merged_img


def test_dataloder(img_path):
    """遍历目录获取图片路径"""
    file_list = []
    for fn in os.listdir(img_path):
        if fn.startswith('.'):
            continue
        if not (fn.endswith('.jpg') or fn.endswith('.jpeg') or fn.endswith('.png')):
            continue
        full_path = os.path.join(img_path, fn)
        file_list.append(full_path)
    return file_list


def main(args):
    Machine = models.__dict__[args.models](datasets=(None, None), args=args)
    model = Machine
    model.model.eval()
    device = model.device
    print("==> testing VM model ")

    # 输出目录
    prediction_dir = args.test_dir + "_output"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    img_paths = test_dataloder(args.test_dir)
    crop_size = args.crop_size
    overlap = 32  # 块重叠像素，可自行调整

    for idx, img_path in enumerate(img_paths):
        print(f"处理第 {idx+1}/{len(img_paths)} 张: {os.path.basename(img_path)}")
        final_img = process_single_image(img_path, model, crop_size, device, overlap)
        save_final_result(final_img, prediction_dir, img_path)

    print(f"\n✅ 处理完成！结果保存在：\n{prediction_dir}")


if __name__ == '__main__':
    parser = Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
    main(parser.parse_args())
