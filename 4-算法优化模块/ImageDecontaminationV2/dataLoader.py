from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class DirtyDocumentsDataset(Dataset):
    def __init__(self, clean_dirs, dirty_dirs, transform=None, img_size=(512, 512), apply_augmentation=True):
        """
        初始化文档去污数据集
        
        参数:
            clean_dirs (str): 干净图像的目录路径
            dirty_dirs (str): 脏图像的目录路径
            transform (callable, optional): 额外的转换操作
            img_size (tuple): 图像大小，默认为(512, 512)
            apply_augmentation (bool): 是否应用数据增强
        """
        self.H, self.W = img_size
        
        # 获取所有图像文件路径（支持png和jpg）
        self.clean_files = sorted(glob.glob(os.path.join(clean_dirs, '*.png')) + 
                                 glob.glob(os.path.join(clean_dirs, '*.jpg')))
        self.dirty_files = sorted(glob.glob(os.path.join(dirty_dirs, '*.png')) + 
                                 glob.glob(os.path.join(dirty_dirs, '*.jpg')))
        
        # 确保干净和脏图像数量相同
        assert len(self.clean_files) == len(self.dirty_files), "干净图像和脏图像数量不匹配"
        
        self.transform = transform
        self.apply_augmentation = apply_augmentation
        
        # 定义数据增强操作（使用 torchvision，兼容 NumPy 2.x）
        if self.apply_augmentation:
            self.augmentations = self._setup_augmentations()
            print(f"数据集: {len(self.clean_files)}原始图像，应用{len(self.augmentations)}种增强方法")
            print(f"增强后总样本数: {len(self)}")
        else:
            self.augmentations = []
            print(f"数据集: {len(self.clean_files)}原始图像，不使用增强")
    
    def _setup_augmentations(self):
        """设置数据增强操作列表"""
        return [
            {"name": "rot90", "k": 1},  # 旋转90度
            {"name": "rot90", "k": 2},  # 旋转180度
            {"name": "rot90", "k": 3},  # 旋转270度
            {"name": "perspective", "scale": (0.02, 0.1)},  # 透视变换
            {"name": "crop", "px": (5, 32)},  # 随机裁剪后还原尺寸
            {"name": "hflip"},  # 水平翻转
            {"name": "vflip"},  # 垂直翻转
            {  # 旋转(90/180/270) + 透视
                "name": "sequence",
                "ops": [
                    {"name": "random_rot90"},
                    {"name": "perspective", "scale": (0.02, 0.1)},
                ],
            },
            {  # 裁剪 + 50%水平翻转 + 高斯模糊
                "name": "sequence",
                "ops": [
                    {"name": "crop", "px": (5, 32)},
                    {"name": "hflip", "p": 0.5},
                    {"name": "gaussian_blur", "sigma": (0.0, 1.5)},
                ],
            },
            {  # 垂直翻转 + 运动模糊
                "name": "sequence",
                "ops": [
                    {"name": "vflip"},
                    {"name": "motion_blur", "k": 6},
                ],
            },
        ]

    def _apply_single_aug(self, y_tensor, aug_conf, rng):
        """对单通道或多通道张量应用一次增强（保持输出尺寸不变）"""
        h, w = y_tensor.shape[-2:]
        name = aug_conf["name"]
        # 获取通道数，支持单通道 [H, W] 或多通道 [C, H, W]
        num_channels = y_tensor.shape[0] if len(y_tensor.shape) == 3 else 1

        if name == "rot90":
            k = aug_conf.get("k", 1)
            return torch.rot90(y_tensor, k=k, dims=(-2, -1))

        if name == "random_rot90":
            k = int(torch.randint(1, 4, (1,), generator=rng).item())
            return torch.rot90(y_tensor, k=k, dims=(-2, -1))

        if name == "perspective":
            scale_min, scale_max = aug_conf.get("scale", (0.02, 0.1))
            scale = torch.rand(1, generator=rng).item() * (scale_max - scale_min) + scale_min
            startpoints = torch.tensor(
                [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
                device=y_tensor.device,
                dtype=y_tensor.dtype,
            )
            displacement = scale * min(h, w)
            # 使用传入的rng生成随机偏移，确保所有通道使用相同的随机数
            # uniform_不支持generator参数，使用rand配合缩放来实现
            offsets = (torch.rand(startpoints.shape, generator=rng, device=startpoints.device, dtype=startpoints.dtype) * 2.0 - 1.0) * displacement
            endpoints = startpoints + offsets
            endpoints[:, 0].clamp_(0.0, w - 1.0)
            endpoints[:, 1].clamp_(0.0, h - 1.0)
            return TF.perspective(
                y_tensor,
                startpoints=startpoints,
                endpoints=endpoints,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )

        if name == "crop":
            min_px, max_px = aug_conf.get("px", (5, 32))
            top = int(torch.randint(min_px, max_px + 1, (1,), generator=rng).item())
            bottom = int(torch.randint(min_px, max_px + 1, (1,), generator=rng).item())
            left = int(torch.randint(min_px, max_px + 1, (1,), generator=rng).item())
            right = int(torch.randint(min_px, max_px + 1, (1,), generator=rng).item())
            crop_h = max(h - top - bottom, 1)
            crop_w = max(w - left - right, 1)
            return TF.resized_crop(
                y_tensor,
                top=top,
                left=left,
                height=crop_h,
                width=crop_w,
                size=[h, w],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

        if name == "hflip":
            p = aug_conf.get("p", 1.0)
            if p >= 1.0 or torch.rand(1, generator=rng).item() < p:
                return TF.hflip(y_tensor)
            return y_tensor

        if name == "vflip":
            p = aug_conf.get("p", 1.0)
            if p >= 1.0 or torch.rand(1, generator=rng).item() < p:
                return TF.vflip(y_tensor)
            return y_tensor

        if name == "gaussian_blur":
            sigma_min, sigma_max = aug_conf.get("sigma", (0.0, 1.5))
            sigma = torch.rand(1, generator=rng).item() * (sigma_max - sigma_min) + sigma_min
            # 使用3x3核，保持尺寸
            return TF.gaussian_blur(y_tensor, kernel_size=3, sigma=sigma)

        if name == "motion_blur":
            k = int(aug_conf.get("k", 6))
            k = max(1, k)
            if k % 2 == 0:
                k += 1  # 使用奇数核以对称填充
            
            # 创建运动模糊核：水平方向的一维核
            blur_kernel = torch.zeros((k, k), device=y_tensor.device, dtype=y_tensor.dtype)
            blur_kernel[k // 2, :] = 1.0 / k  # 水平运动模糊
            padding = (k // 2, k // 2)
            
            # 统一处理：确保输入是 [C, H, W] 格式
            input_was_2d = len(y_tensor.shape) == 2
            if input_was_2d:
                y_tensor = y_tensor.unsqueeze(0)  # [H, W] -> [1, H, W]
            
            num_channels = y_tensor.shape[0]
            
            # 对每个通道分别应用相同的模糊核，确保所有通道使用相同的变换
            blurred_channels = []
            for c in range(num_channels):
                kernel = blur_kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]
                channel_data = y_tensor[c:c+1].unsqueeze(0)  # [1, 1, H, W]
                blurred_channel = F.conv2d(channel_data, kernel, padding=padding)
                blurred_channels.append(blurred_channel.squeeze(0).squeeze(0))  # [H, W]
            
            blurred = torch.stack(blurred_channels, dim=0)  # [C, H, W]
            # 如果输入是 [H, W]，返回 [H, W]；否则返回 [C, H, W]
            return blurred.squeeze(0) if input_was_2d else blurred

        if name == "sequence":
            y_out = y_tensor
            for op in aug_conf.get("ops", []):
                y_out = self._apply_single_aug(y_out, op, rng)
            return y_out

        return y_tensor

    def _apply_augmentation(self, clean_img_np, dirty_img_np, aug_conf):
        """
        对两张匹配图像的所有通道同步应用增强，保持尺寸不变。
        确保所有通道使用相同的随机数。
        """
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
        rng = torch.Generator()
        rng.manual_seed(seed)

        # 转换为张量 [H, W, 3] -> [3, H, W]
        clean_img_t = torch.from_numpy(clean_img_np.transpose((2, 0, 1))).float()  # [3, H, W]
        dirty_img_t = torch.from_numpy(dirty_img_np.transpose((2, 0, 1))).float()

        # 对整个图像（所有通道一起）应用增强，确保所有通道使用相同的随机数
        # 重置随机数生成器以确保两张图像使用相同的随机参数
        rng.manual_seed(seed)
        clean_aug = self._apply_single_aug(clean_img_t, aug_conf, rng)
        rng.manual_seed(seed)
        dirty_aug = self._apply_single_aug(dirty_img_t, aug_conf, rng)
        
        # 转换回 [H, W, 3] 格式
        clean_aug = clean_aug.permute(1, 2, 0)  # [H, W, 3]
        dirty_aug = dirty_aug.permute(1, 2, 0)
        
        return clean_aug.numpy(), dirty_aug.numpy()

    def __len__(self):
        """返回数据集大小"""
        if self.apply_augmentation:
            return len(self.clean_files) * (1 + len(self.augmentations))
        else:
            return len(self.clean_files)

    def _load_image(self, image_path):
        """
        加载并预处理单个图像，转换为YUV格式。
        对于小于目标尺寸 (H, W) 的图像，不进行放大，只做必要的边缘填充以保证张量尺寸一致。
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")

        orig_h, orig_w = img.shape[:2]
        
        # 检测图像类型（RGB或灰度）
        if len(img.shape) == 3 and img.shape[2] == 3:
            # RGB 图像：如果任一边大于目标尺寸，则缩小到 (W, H)，否则保持原尺寸
            if orig_h > self.H or orig_w > self.W:
                img_proc = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            else:
                img_proc = img

            # 使用 OpenCV 的标准 YUV 转换
            img_yuv_cv = cv2.cvtColor(img_proc, cv2.COLOR_BGR2YUV)
            # 归一化到 [0, 1] 范围
            img_yuv = img_yuv_cv.astype("float32") / 255.0
        else:
            # 灰度图像：如果任一边大于目标尺寸，则缩小到 (W, H)，否则保持原尺寸
            if len(img.shape) == 2:
                img_gray = img
            else:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if orig_h > self.H or orig_w > self.W:
                img_gray = cv2.resize(img_gray, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

            # 使用 OpenCV 的标准灰度到YUV转换（自动处理UV通道）
            img_yuv_cv = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2YUV)
            # 归一化到 [0, 1] 范围
            img_yuv = img_yuv_cv.astype("float32") / 255.0

        # 对于小于目标尺寸的图像，不放大，只在右侧和下侧做 0 填充到 (H, W)
        h, w = img_yuv.shape[:2]
        pad_bottom = max(0, self.H - h)
        pad_right = max(0, self.W - w)
        if pad_bottom > 0 or pad_right > 0:
            img_yuv = np.pad(
                img_yuv,
                ((0, pad_bottom), (0, pad_right), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )
        
        return img_yuv

    def __getitem__(self, idx):
        """获取单个数据样本"""
        # 转换张量索引为列表
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 计算原始图像索引和增强类型
        orig_size = len(self.clean_files)
        
        if idx < orig_size:
            # 原始图像
            img_idx = idx
            aug_idx = None
        else:
            # 增强图像
            aug_idx = (idx - orig_size) // orig_size
            img_idx = (idx - orig_size) % orig_size
            
            # 确保增强索引有效
            if aug_idx >= len(self.augmentations):
                aug_idx = aug_idx % len(self.augmentations)
        
        # 加载图像
        clean_path = self.clean_files[img_idx]
        dirty_path = self.dirty_files[img_idx]
        
        clean_img = self._load_image(clean_path)
        dirty_img = self._load_image(dirty_path)
        
        # 如果需要，应用增强
        if aug_idx is not None:
            aug_conf = self.augmentations[aug_idx]
            # 对所有通道同步应用增强，确保所有通道使用相同的随机数
            clean_img, dirty_img = self._apply_augmentation(clean_img, dirty_img, aug_conf)
        
        # 将NumPy数组转换为PyTorch张量，并调整通道顺序 [H, W, 3] -> [3, H, W]
        clean_img = torch.from_numpy(clean_img.transpose((2, 0, 1)).copy()).float()
        dirty_img = torch.from_numpy(dirty_img.transpose((2, 0, 1)).copy()).float()
        
        # 应用任何额外的变换
        if self.transform:
            clean_img = self.transform(clean_img)
            dirty_img = self.transform(dirty_img)
        
        return dirty_img, clean_img


class DirtyDocumentsDataset_Test(Dataset):
    def __init__(self, test_dir, transform=None, img_size=(512, 512)):
        """
        初始化测试数据集
        参数:
            test_dir (str): 测试图像的目录路径
            transform (callable, optional): 应用于图像的转换
            img_size (tuple): 图像大小，默认为(512, 512)
        """
        # 获取所有图像文件路径
        self.img_files = sorted(glob.glob(os.path.join(test_dir, '*.png')) + 
                               glob.glob(os.path.join(test_dir, '*.jpg')))
        
        # 提取文件名
        self.filenames = [os.path.basename(f) for f in self.img_files]
        
        self.transform = transform
        self.H, self.W = img_size
        
        print(f"测试数据集: {len(self.img_files)}图像")

    def __len__(self):
        """返回数据集大小"""
        return len(self.img_files)
    
    def _load_image(self, image_path):
        """
        加载并预处理单个测试图像，转换为YUV格式。
        对于小于目标尺寸 (H, W) 的图像，不进行放大，只做必要的边缘填充以保证张量尺寸一致。
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 保存原始尺寸（用于后处理还原）
        original_h, original_w = img.shape[:2]
        
        # 检测图像类型（RGB或灰度）
        if len(img.shape) == 3 and img.shape[2] == 3:
            # RGB 图像：如果任一边大于目标尺寸，则缩小到 (W, H)，否则保持原尺寸
            if original_h > self.H or original_w > self.W:
                img_proc = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            else:
                img_proc = img

            # 使用 OpenCV 的标准 YUV 转换
            img_yuv_cv = cv2.cvtColor(img_proc, cv2.COLOR_BGR2YUV)
            # 归一化到 [0, 1] 范围
            img_yuv = img_yuv_cv.astype("float32") / 255.0
        else:
            # 灰度图像：如果任一边大于目标尺寸，则缩小到 (W, H)，否则保持原尺寸
            if len(img.shape) == 2:
                img_gray = img
            else:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if original_h > self.H or original_w > self.W:
                img_gray = cv2.resize(img_gray, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

            # 使用 OpenCV 的标准灰度到YUV转换（自动处理UV通道）
            img_yuv_cv = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2YUV)
            # 归一化到 [0, 1] 范围
            img_yuv = img_yuv_cv.astype("float32") / 255.0

        # 对于小于目标尺寸的图像，不放大，只在右侧和下侧做 0 填充到 (H, W)
        h, w = img_yuv.shape[:2]
        pad_bottom = max(0, self.H - h)
        pad_right = max(0, self.W - w)
        if pad_bottom > 0 or pad_right > 0:
            img_yuv = np.pad(
                img_yuv,
                ((0, pad_bottom), (0, pad_right), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )
        
        return img_yuv, (original_h, original_w)

    def __getitem__(self, idx):
        """获取单个测试样本"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 加载图像（返回YUV格式和原始尺寸）
        img_path = self.img_files[idx]
        img_yuv, original_size = self._load_image(img_path)
        
        # 将NumPy数组转换为PyTorch张量，并调整通道顺序 [H, W, 3] -> [3, H, W]
        img = torch.from_numpy(img_yuv.transpose((2, 0, 1)).copy()).float()
        
        # 应用任何变换
        if self.transform:
            img = self.transform(img)
        
        return img, self.filenames[idx], original_size
