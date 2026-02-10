import os.path as osp
import random

import cv2
import numpy as np
import paddle
from paddle.io import Dataset

from paddle.vision.transforms.functional import normalize
from utils.misc import scandir
from utils.file_client import FileClient
from utils.img_utils import padding, img2tensor, imfrombytes
from data.transforms import paired_random_crop, random_augmentation

class Dataset_PairedImage(Dataset):
    """图像修复数据集配对
    读取低质量（LQ，例如低分辨率LR、模糊、噪声等）图像与目标图像（GT）的配对数据。
    支持三种模式：
    1. ‘lmdb’：使用lmdb文件。
        当opt[‘io_backend’] == lmdb时生效。
    2. ‘meta_info_file’: 使用元信息文件生成路径。
        当 opt[‘io_backend’] != lmdb 且 opt[‘meta_info_file’] 不为空时生效。
    3. ‘folder’: 扫描文件夹生成路径。
        其余情况适用。
    参数：
        opt (dict): 训练数据集配置。包含以下：
            dataroot_gt (str): GT数据根路径。
            dataroot_lq (str): LQ数据根路径。
            meta_info_file (str): 元数据文件路径。
            io_backend (dict): I/O后端类型及其他关键字参数。
            filename_tmpl (str): 文件名模板。注意：模板不包含文件扩展名。默认值{}。
            gt_size (int)：GT裁剪后的尺寸。
            geometric_augs (bool)：是否使用几何增强。
            scale (bool)：缩放因子（自动添加）。
            phase (str)：‘train’ 或 ‘val’。
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if 'meta_info_file' in self.opt and self.opt[
            'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
        elif self.opt['phase'] == 'val':
            img_gt = cv2.resize(img_gt, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            img_lq = cv2.resize(img_lq, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, 1024, scale,
                                                gt_path)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            img_lq = normalize(img_lq, self.mean, self.std)
            img_gt = normalize(img_gt, self.mean, self.std)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


class Dataset_GaussianDenoising(Dataset):
    """基本功能与上文一致
        use_flip (bool)：启用水平翻转。
        use_rot (bool)：启用旋转（通过垂直翻转并交换h/w轴实现）。
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']

        self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)


            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = sigma_value / 255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            # noise = np.random.randn(*img_lq.shape) * noise_level
            noise = paddle.randn(img_lq.shape,dtype='float32').numpy() * noise_level

            img_lq = img_lq + noise.astype('float32')

        else:
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


def paired_paths_from_meta_info_file(folders, keys, meta_info_file,
                                     filename_tmpl):
    """从元信息文件生成配对路径。
    元信息文件的每行包含图像名称和图像尺寸（通常为gt格式），以空格分隔。
    元信息文件示例：
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```
    参数：
        folders (list[str]): 文件夹路径列表。列表顺序应为 [输入文件夹, gt文件夹]。
        keys (list[str]): 标识文件夹的键值列表。顺序需与folders保持一致，例如[‘lq’, ‘gt’]。
        元数据文件 (str)：元数据文件路径。
        文件名模板 (str)：每个文件名的模板。注意模板不包含文件扩展名。通常文件名模板用于输入文件夹中的文件。
    返回值：
        列表[str]：返回的路径列表。
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        gt_path = osp.join(gt_folder, gt_name)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths

def paired_paths_from_folder(folders, keys, filename_tmpl):
    """生成配对路径
    参数与上文一致
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(input_paths) == len(gt_paths), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx]
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_path = input_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(input_path))
        input_name = f'{filename_tmpl.format(basename)}{ext_input}'
        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, (f'{input_name} is not in '
                                           f'{input_key}_paths.')
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths
