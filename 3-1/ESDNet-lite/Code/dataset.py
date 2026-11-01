"""Dataset helpers for UHDM-style training and validation."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def find_uhdm_pairs(data_dir: str | Path) -> list[tuple[str, str]]:
    pairs = []
    for root, _, files in os.walk(data_dir):
        for file_name in sorted(files):
            if not file_name.endswith('gt.jpg'):
                continue
            gt_path = os.path.join(root, file_name)
            moire_path = os.path.join(root, f'{file_name[:4]}_moire.jpg')
            if os.path.isfile(moire_path):
                pairs.append((moire_path, gt_path))
    pairs.sort()
    return pairs


class UHDMTrainDataset(data.Dataset):
    def __init__(self, data_dir: str | Path, crop_size: int = 512):
        self.pairs = find_uhdm_pairs(data_dir)
        self.crop_size = crop_size
        if not self.pairs:
            raise ValueError(f'No UHDM pairs found in {data_dir}')

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        moire_path, gt_path = self.pairs[index]
        moire = Image.open(moire_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')
        width, height = moire.size
        crop = self.crop_size
        x = random.randint(0, max(0, width - crop))
        y = random.randint(0, max(0, height - crop))
        moire = moire.crop((x, y, x + crop, y + crop))
        gt = gt.crop((x, y, x + crop, y + crop))

        if random.random() > 0.5:
            moire = moire.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        moire = torch.from_numpy(np.array(moire, dtype=np.float32) / 255.0).permute(2, 0, 1)
        gt = torch.from_numpy(np.array(gt, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return {'in_img': moire, 'label': gt}


class UHDMValDataset(data.Dataset):
    def __init__(self, data_dir: str | Path, max_num: int = 50):
        self.pairs = find_uhdm_pairs(data_dir)[:max_num]
        if not self.pairs:
            raise ValueError(f'No UHDM pairs found in {data_dir}')

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        moire_path, gt_path = self.pairs[index]
        moire = torch.from_numpy(np.array(Image.open(moire_path).convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1)
        gt = torch.from_numpy(np.array(Image.open(gt_path).convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1)
        return {'in_img': moire, 'label': gt, 'number': Path(gt_path).name[:4]}
