from PIL import Image
import numpy as np
import cv2
import os.path as osp
import os
import sys
import torch
from torchvision import datasets, transforms
from .base_dataset import get_transform
import random


class CLWDDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, args):

        args.is_train = is_train == 'train'
        if args.is_train == True:
            self.root = args.dataset_dir + '/train/'
            self.keep_background_prob = -1
        elif args.is_train == False:
            self.root = args.dataset_dir + '/test/'
            self.keep_background_prob = -1
            args.preprocess = 'resize'
            args.no_flip = True

        self.args = args
        self.crop_size = args.crop_size
        self.use_tile_crop = args.preprocess == 'crop_only'

        self.transform_norm = transforms.Compose([
            transforms.ToTensor()])

        self.augment_transform = get_transform(args,
                                               additional_targets={'J': 'image', 'I': 'image', 'watermark': 'image',
                                                                   'mask': 'mask', 'alpha': 'mask'})
        self.transform_tensor = transforms.ToTensor()

        self.imageJ_path = osp.join(self.root, 'Watermarked_image', '%s.jpg')
        self.imageI_path = osp.join(self.root, 'Watermark_free_image', '%s.jpg')
        self.mask_path = osp.join(self.root, 'Mask', '%s.png')
        self.alpha_path = osp.join(self.root, 'Alpha', '%s.png')
        self.W_path = osp.join(self.root, 'Watermark', '%s.png')

        self.ids = list()
        for file in os.listdir(self.root + '/Watermarked_image'):
            self.ids.append(file.strip('.jpg'))

        # 切片索引：大图切多块
        self.tiles = []
        if self.use_tile_crop and args.is_train:
            self._prepare_tiles()
        else:
            self.tiles = None

        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

    def _prepare_tiles(self):
        """训练时：大图按 crop_size 切成多块，扩展数据集长度"""
        crop_size = self.crop_size
        for idx, img_id in enumerate(self.ids):
            h, w = self._get_image_size(img_id)
            n_h = h // crop_size
            n_w = w // crop_size
            n_h = max(n_h, 1)
            n_w = max(n_w, 1)

            for i in range(n_h):
                for j in range(n_w):
                    self.tiles.append((idx, i, j))

    def _get_image_size(self, img_id):
        path = self.imageJ_path % img_id
        img = cv2.imread(path)
        return img.shape[:2]

    def __len__(self):
        if self.tiles is not None:
            return len(self.tiles)
        return len(self.ids)

    def get_sample(self, index):
        # 切片模式
        if self.tiles is not None:
            idx, i, j = self.tiles[index]
            img_id = self.ids[idx]
            crop_size = self.crop_size

            img_J = cv2.imread(self.imageJ_path % img_id)
            img_J = cv2.cvtColor(img_J, cv2.COLOR_BGR2RGB)
            img_I = cv2.imread(self.imageI_path % img_id)
            img_I = cv2.cvtColor(img_I, cv2.COLOR_BGR2RGB)
            w = cv2.imread(self.W_path % img_id)
            w = cv2.cvtColor(w, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.mask_path % img_id)
            alpha = cv2.imread(self.alpha_path % img_id)

            h, w_img = img_J.shape[:2]
            y = i * crop_size
            x = j * crop_size
            y = min(y, h - crop_size)
            x = min(x, w_img - crop_size)

            img_J = img_J[y:y+crop_size, x:x+crop_size]
            img_I = img_I[y:y+crop_size, x:x+crop_size]
            w = w[y:y+crop_size, x:x+crop_size]
            mask = mask[y:y+crop_size, x:x+crop_size]
            alpha = alpha[y:y+crop_size, x:x+crop_size]

        else:
            img_id = self.ids[index]
            img_J = cv2.imread(self.imageJ_path % img_id)
            img_J = cv2.cvtColor(img_J, cv2.COLOR_BGR2RGB)
            img_I = cv2.imread(self.imageI_path % img_id)
            img_I = cv2.cvtColor(img_I, cv2.COLOR_BGR2RGB)
            w = cv2.imread(self.W_path % img_id)
            w = cv2.cvtColor(w, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.mask_path % img_id)
            alpha = cv2.imread(self.alpha_path % img_id)

        # 保持三维形状 (H, W, 1)，避免 Albumentations 尺寸解析错误
        mask = mask[:, :, 0:1].astype(np.float32) / 255.
        alpha = alpha[:, :, 0:1].astype(np.float32) / 255.

        return {'J': img_J, 'I': img_I, 'watermark': w, 'mask': mask, 'alpha': alpha,
                'img_path': self.imageJ_path % img_id}

    def __getitem__(self, index):
        sample = self.get_sample(index)
        self.check_sample_types(sample)
        sample = self.augment_sample(sample)

        J = self.transform_norm(sample['J'])
        I = self.transform_norm(sample['I'])
        w = self.transform_norm(sample['watermark'])

        # ✅ 正确维度转换：numpy 使用 transpose
        mask = sample['mask'].astype(np.float32).transpose(2, 0, 1)
        mask = np.where(mask > 0.1, 1, 0).astype(np.uint8)
        alpha = sample['alpha'].astype(np.float32).transpose(2, 0, 1)

        data = {
            'image': J,
            'target': I,
            'wm': w,
            'mask': mask,
            'alpha': alpha,
            'img_path': sample['img_path']
        }
        return data

    def check_sample_types(self, sample):
        assert sample['J'].dtype == 'uint8'
        assert sample['I'].dtype == 'uint8'
        assert sample['watermark'].dtype == 'uint8'

    def augment_sample(self, sample):
        if self.augment_transform is None:
            return sample
        additional_targets = {target_name: sample[target_name]
                              for target_name in self.augment_transform.additional_targets.keys()}

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.augment_transform(image=sample['I'], **additional_targets)
            valid_augmentation = self.check_augmented_sample(sample, aug_output)

        for target_name, transformed_target in aug_output.items():
            sample[target_name] = transformed_target

        return sample

    def check_augmented_sample(self, sample, aug_output):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True
        return aug_output['mask'].sum() > 100
