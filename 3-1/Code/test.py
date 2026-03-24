"""Inference entry for PyTorch CPU/GPU and NCNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import build_model
from preprocess import img_to_tensor, list_image_files, save_tensor_image
from utils import (
    DEFAULT_OVERLAP,
    DEFAULT_TILE,
    MAX_SAFE_TILE,
    NcnnModel,
    get_blend_window_np,
    get_blend_window_torch,
    load_checkpoint_state_dict,
    pad_tile_array,
    pad_tile_tensor,
)


def load_torch_model(model_path: str, preset: str, device: str):
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval().to(device)
        print(f'[i] 以 TorchScript 方式加载模型: {model_path}')
        return model
    except Exception:
        model = build_model(preset)
        model.load_state_dict(load_checkpoint_state_dict(model_path), strict=True)
        model.eval().to(device)
        print(f'[i] 以 checkpoint 方式加载模型: {model_path}')
        return model


@torch.no_grad()
def run_torch_model(model, image: torch.Tensor, device: str) -> torch.Tensor:
    output = model(image.to(device))
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.detach().cpu().clamp(0, 1)


@torch.no_grad()
def infer_whole_torch(model, image: torch.Tensor, device: str) -> torch.Tensor:
    return run_torch_model(model, image, device)


@torch.no_grad()
def infer_tile_torch(model, image: torch.Tensor, tile: int, tile_overlap: int, device: str, color_correct: bool = True) -> torch.Tensor:
    _, channels, height, width = image.shape
    padder = getattr(model, 'padder_size', 8)
    tile_size = max(padder, (tile // padder) * padder)
    step = tile_size - tile_overlap
    if step <= 0:
        raise ValueError('tile_overlap must be smaller than tile')

    output = torch.zeros(1, channels, height, width, dtype=torch.float32)
    weight = torch.zeros(1, 1, height, width, dtype=torch.float32)
    window = get_blend_window_torch(tile_size, tile_overlap)

    ys = list(dict.fromkeys(list(range(0, max(1, height - tile_size + 1), step)) + [max(0, height - tile_size)]))
    xs = list(dict.fromkeys(list(range(0, max(1, width - tile_size + 1), step)) + [max(0, width - tile_size)]))

    for y in ys:
        for x in xs:
            patch = image[:, :, y:y + tile_size, x:x + tile_size]
            patch, patch_h, patch_w = pad_tile_tensor(patch, tile_size)
            tile_output = run_torch_model(model, patch, device)[:, :, :patch_h, :patch_w].clone()
            tile_window = window[:, :, :patch_h, :patch_w]

            if color_correct:
                existing_weight = weight[:, :, y:y + patch_h, x:x + patch_w]
                overlap_mask = existing_weight[0, 0] > 0.05
                if overlap_mask.sum() >= 64:
                    placed = output[:, :, y:y + patch_h, x:x + patch_w] / existing_weight.expand(1, channels, patch_h, patch_w).clamp(min=1e-6)
                    delta = (placed[0][:, overlap_mask] - tile_output[0][:, overlap_mask]).mean(dim=1)
                    tile_output = (tile_output + delta.view(1, channels, 1, 1)).clamp(0, 1)

            output[:, :, y:y + patch_h, x:x + patch_w] += tile_output * tile_window
            weight[:, :, y:y + patch_h, x:x + patch_w] += tile_window

    return (output / weight.clamp(min=1e-6)).clamp(0, 1)


def infer_tile_ncnn(model: NcnnModel, image: torch.Tensor, tile: int, tile_overlap: int, color_correct: bool = True) -> torch.Tensor:
    _, channels, height, width = image.shape
    tile_size = max(model.padder_size, (tile // model.padder_size) * model.padder_size)
    step = tile_size - tile_overlap
    if step <= 0:
        raise ValueError('tile_overlap must be smaller than tile')

    image_np = np.ascontiguousarray(image.squeeze(0).numpy(), dtype=np.float32)
    output = np.zeros((channels, height, width), dtype=np.float32)
    weight = np.zeros((1, height, width), dtype=np.float32)
    window = get_blend_window_np(tile_size, tile_overlap)

    ys = list(dict.fromkeys(list(range(0, max(1, height - tile_size + 1), step)) + [max(0, height - tile_size)]))
    xs = list(dict.fromkeys(list(range(0, max(1, width - tile_size + 1), step)) + [max(0, width - tile_size)]))

    for y in ys:
        for x in xs:
            patch = image_np[:, y:y + tile_size, x:x + tile_size]
            patch, patch_h, patch_w = pad_tile_array(patch, tile_size)
            tile_output = np.clip(model.infer_chw(patch), 0.0, 1.0)[:, :patch_h, :patch_w].copy()
            tile_window = window[:patch_h, :patch_w][None, :, :]

            if color_correct:
                existing_weight = weight[:, y:y + patch_h, x:x + patch_w]
                overlap_mask = existing_weight[0] > 0.05
                if int(overlap_mask.sum()) >= 64:
                    placed = output[:, y:y + patch_h, x:x + patch_w] / np.clip(existing_weight, 1e-6, None)
                    delta = (placed[:, overlap_mask] - tile_output[:, overlap_mask]).mean(axis=1)
                    tile_output = np.clip(tile_output + delta[:, None, None], 0.0, 1.0)

            output[:, y:y + patch_h, x:x + patch_w] += tile_output * tile_window
            weight[:, y:y + patch_h, x:x + patch_w] += tile_window

    fused = np.clip(output / np.clip(weight, 1e-6, None), 0.0, 1.0)
    return torch.from_numpy(np.ascontiguousarray(fused)).unsqueeze(0)


def build_parser():
    parser = argparse.ArgumentParser(description='ESDNet-Lite 推理入口（仅保留 PyTorch CPU/GPU 和 NCNN）')
    parser.add_argument('--backend', choices=['torch', 'ncnn'], required=True)
    parser.add_argument('--preset', choices=['full', 'lite-s', 'lite-xs'], default='lite-s')
    parser.add_argument('--model_path', default=None, help='PyTorch checkpoint 或 TorchScript 路径')
    parser.add_argument('--ncnn_param', default=None, help='NCNN param 路径')
    parser.add_argument('--ncnn_bin', default=None, help='NCNN bin 路径')
    parser.add_argument('--device', default='cpu', help='PyTorch 设备，例如 cpu / cuda / cuda:0')
    parser.add_argument('--mode', choices=['tile', 'whole'], default='tile')
    parser.add_argument('--input_dir', required=True, help='输入图片目录')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--tile', type=int, default=DEFAULT_TILE)
    parser.add_argument('--tile_overlap', type=int, default=DEFAULT_OVERLAP)
    parser.add_argument('--save_format', choices=['png', 'jpg'], default='png')
    parser.add_argument('--ncnn_vulkan', action='store_true')
    parser.add_argument('--no_color_correct', action='store_true')
    return parser


def main():
    args = build_parser().parse_args()
    color_correct = not args.no_color_correct
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == 'torch':
        if not args.model_path:
            raise ValueError('--backend torch 时必须指定 --model_path')
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError(f'CUDA 不可用，无法使用设备 {args.device}')
        model = load_torch_model(args.model_path, args.preset, args.device)
    else:
        if not args.ncnn_param or not args.ncnn_bin:
            raise ValueError('--backend ncnn 时必须同时指定 --ncnn_param 和 --ncnn_bin')
        if args.mode != 'tile':
            raise ValueError('NCNN 仅支持 tile 模式')
        if not args.ncnn_vulkan and args.tile > MAX_SAFE_TILE:
            raise ValueError(f'NCNN CPU 模式限制 tile <= {MAX_SAFE_TILE}')
        model = NcnnModel(args.ncnn_param, args.ncnn_bin, use_vulkan=args.ncnn_vulkan)

    image_files = list_image_files(args.input_dir)
    if not image_files:
        raise ValueError(f'No image files found in {args.input_dir}')

    for image_path in tqdm(image_files, desc='Inferencing'):
        image = img_to_tensor(image_path)
        if args.backend == 'torch':
            if args.mode == 'whole':
                output = infer_whole_torch(model, image, args.device)
            else:
                output = infer_tile_torch(model, image, args.tile, args.tile_overlap, args.device, color_correct=color_correct)
        else:
            output = infer_tile_ncnn(model, image, args.tile, args.tile_overlap, color_correct=color_correct)

        save_name = f'{image_path.stem}.{args.save_format}'
        save_tensor_image(output, output_dir / save_name, image_format=args.save_format)

    print(f'[✓] 推理完成，结果已保存到: {output_dir}')


if __name__ == '__main__':
    main()
