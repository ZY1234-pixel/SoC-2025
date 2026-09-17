import collections
import json
import os

from PIL import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
from torch.utils.data import Dataset, DataLoader
import hdf5storage as h5
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.interpolate import griddata
class Warp_DataSet(Dataset):
    def __init__(self, input_size, data_root, split, is_aug=False):
        assert os.path.exists(data_root), 'Images folder does not exist'
        self.split = split
        self.is_aug = is_aug
        self.files = collections.defaultdict(list)
        self.data_root = data_root
        path = os.path.join(self.data_root, split + '.txt')
        file_list = tuple(open(path, 'r').read().splitlines())
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[split] = file_list

        self.img_size = input_size

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        try:
            img_filename = self.files[self.split][index]

            # 只读取 im_path (二值掩码图像) 和 bm_path (真实形变场)
            im_path = os.path.join(self.data_root, 'img', img_filename + '.jpg')
            h_im_path = os.path.join(self.data_root, 'h_mask', img_filename + '.png')
            v_im_path = os.path.join(self.data_root, 'v_mask', img_filename + '.png')
            edge_path = os.path.join(self.data_root, 'edge', img_filename + '_mask.png')
            # grid_dir=os.path.join(self.data_root,'grid2d')
            # meta_dir=os.path.join(self.data_root,'metadata_sample')
            # bm_path, _ = self.infer_mat_from_metadata(img_filename,grid_dir,meta_dir)
            bm_path = os.path.join(self.data_root, 'grid2d', img_filename + '.mat')

            # 读取二值掩码图像作为输入
            im = np.array(Image.open(im_path).convert("RGB"))
            h_mask = cv2.imread(h_im_path, cv2.IMREAD_GRAYSCALE)
            v_mask = cv2.imread(v_im_path, cv2.IMREAD_GRAYSCALE)
            edge_mask = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            # print("im:" ,img_filename ,"grid2d shape:", h5.loadmat(bm_path)['grid2d'].shape)
            # 读取形变场作为标签
            # bm = np.transpose(h5.loadmat(bm_path)['grid2d'], (2, 1, 0))
            bm=h5.loadmat(bm_path)['grid2d']
            if bm.shape[0]==2:
                bm = np.transpose(bm, (2, 1, 0))
            h_img, v_img, lbl, img , edge= self.transform_new(h_mask, v_mask, bm,im,edge_mask)

            lbl = lbl.permute((2, 0, 1))  # HWC -> CHW
        except Exception as e:
            print(f"Failed to read: {self.files[self.split][index]}")
            return self[index + 1]

        return h_img, v_img, lbl, img, edge

    def infer_mat_from_metadata(self, img_stem, grid_dir, meta_dir):
        """从 metadata 推断对应的 .mat 文件"""
        meta_path = os.path.join(meta_dir, f"{img_stem}.json")
        if not os.path.isfile(meta_path):
            return None, None

        with open(meta_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        geom_name = info.get("geom_name") if isinstance(info, dict) else None
        if not geom_name:
            return None, None

        mat_path = os.path.join(grid_dir, f"{geom_name}.mat")
        if not os.path.isfile(mat_path):
            return None, geom_name

        return mat_path, geom_name

    def rotate(self, img, h_mask, v_mask, edge_mask, bm):
        """
        随机旋转图像、掩码和 backward map (BM)
        BM 定义：dst(x,y) <- src(bm_x, bm_y)
        """
        angle = random.uniform(-30, 30)
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)

        # 正向旋转矩阵（用于旋转图像和 mask）
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 计算旋转后画布尺寸
        cos = np.abs(rot_matrix[0, 0])
        sin = np.abs(rot_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        rot_matrix[0, 2] += (new_w / 2) - center[0]
        rot_matrix[1, 2] += (new_h / 2) - center[1]

        # ---------- 1. 旋转 RGB 图像 ----------
        img_rot = cv2.warpAffine(
            img, rot_matrix, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(128, 128, 128)
        )

        # ---------- 2. 旋转 Mask（最近邻） ----------
        h_mask_rot = cv2.warpAffine(
            h_mask, rot_matrix, (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        v_mask_rot = cv2.warpAffine(
            v_mask, rot_matrix, (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        edge_mask_rot = cv2.warpAffine(
            edge_mask, rot_matrix, (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # ---------- 3. 正确旋转 Backward Map ----------
        # 构造旋转后图像的网格
        yy, xx = np.meshgrid(
            np.arange(new_h),
            np.arange(new_w),
            indexing='ij'
        )

        # 齐次坐标
        coords = np.stack([
            xx.ravel(),
            yy.ravel(),
            np.ones_like(xx.ravel())
        ], axis=0).astype(np.float32)

        # 逆旋转矩阵：dst -> src
        inv_rot_matrix = cv2.invertAffineTransform(rot_matrix).astype(np.float32)

        # 反推在原 BM 上的采样位置
        src_coords = inv_rot_matrix @ coords
        src_x = src_coords[0, :].reshape(new_h, new_w)
        src_y = src_coords[1, :].reshape(new_h, new_w)

        # 使用 remap 从原 BM 中采样
        bm_rot = np.zeros((new_h, new_w, 2), dtype=np.float32)
        for i in range(2):
            bm_rot[..., i] = cv2.remap(
                bm[..., i],
                src_x,
                src_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        return img_rot, h_mask_rot, v_mask_rot, edge_mask_rot, bm_rot

    def transform_new(self, h_mask, v_mask, grid2d, im, edge_mask):
        """
        h_mask / v_mask: 二值掩码 (H, W)
        bm: grid2d (Hc, Wc, 2)，backward map
        im: RGB 图像 (H, W, 3)
        """
        # print("grid2d:",grid2d.shape,grid2d)
        # print(grid2d[:, :, 0].min(), grid2d[:, :, 0].max())
        # print(grid2d[:, :, 1].min(), grid2d[:, :, 1].max())
        # print(im.shape)

        H, W, _ = im.shape

        # 随机旋转增强
        # ===============================
        # if self.is_aug and random.random() > 0.5:
        #     im, h_mask, v_mask, edge_mask, grid2d = self.rotate(
        #         im, h_mask, v_mask, edge_mask, grid2d
        #     )
        #     # 更新旋转后的尺寸
        #     H, W = im.shape[:2]

        if self.is_aug:
            if random.random() > 0.8:
                im = color_jitter(im, 0.2, 0.2, 0.6, 0.6)

        # ===============================
        # 1. Grid2D 本身就是 backward map
        # ===============================
        # grid2d[...,0] : x (W 方向)
        # grid2d[...,1] : y (H 方向)

        gx = grid2d[:, :, 0].astype(np.float32)
        gy = grid2d[:, :, 1].astype(np.float32)

        # 1. 先在稀疏点上完成坐标系切换
        gx = gx * (self.img_size / W)
        gy = gy * (self.img_size / H)

        # 2. densify
        gx = cv2.resize(gx, (self.img_size, self.img_size),
                        interpolation=cv2.INTER_LINEAR)
        gy = cv2.resize(gy, (self.img_size, self.img_size),
                        interpolation=cv2.INTER_LINEAR)

        # ===============================
        # 3. 构造 label（pixel coord）
        # ===============================
        lbl = torch.from_numpy(
            np.stack([gx, gy], axis=-1)
        ).float()

        # ===============================
        # 4. mask / img
        # ===============================
        h_img = cv2.resize(h_mask, (self.img_size, self.img_size))
        h_img = torch.from_numpy(h_img / 255.0).float().unsqueeze(0)

        v_img = cv2.resize(v_mask, (self.img_size, self.img_size))
        v_img = torch.from_numpy(v_img / 255.0).float().unsqueeze(0)

        img = cv2.resize(im, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        edge = cv2.resize(edge_mask, (self.img_size, self.img_size))
        edge = torch.from_numpy(edge / 255.0).float().unsqueeze(0)

        return h_img, v_img, lbl, img, edge

class Warp_DataSet_my(Dataset):
    def __init__(self, input_size, data_root, split, is_aug=False):
        assert os.path.exists(data_root), 'Images folder does not exist'
        self.split = split
        self.is_aug = is_aug
        self.files = collections.defaultdict(list)
        self.data_root = data_root
        path = os.path.join(self.data_root, split + '.txt')
        file_list = tuple(open(path, 'r').read().splitlines())
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[split] = file_list

        self.img_size = input_size

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        try:
            img_filename = self.files[self.split][index]

            # 只读取 im_path (二值掩码图像) 和 bm_path (真实形变场)
            im_path = os.path.join(self.data_root, 'img', img_filename + '.jpg')
            h_im_path = os.path.join(self.data_root, 'h_mask', img_filename + '.png')
            v_im_path = os.path.join(self.data_root, 'v_mask', img_filename + '.png')
            # grid_dir=os.path.join(self.data_root,'grid2d')
            # meta_dir=os.path.join(self.data_root,'metadata_sample')
            # bm_path, _ = self.infer_mat_from_metadata(img_filename,grid_dir,meta_dir)
            bm_path = os.path.join(self.data_root, 'grid2d', img_filename + '.mat')

            # 读取二值掩码图像作为输入
            im = np.array(Image.open(im_path).convert("RGB"))
            h_mask = cv2.imread(h_im_path, cv2.IMREAD_GRAYSCALE)
            v_mask = cv2.imread(v_im_path, cv2.IMREAD_GRAYSCALE)
            # print("im:" ,img_filename ,"grid2d shape:", h5.loadmat(bm_path)['grid2d'].shape)
            # 读取形变场作为标签
            # bm = np.transpose(h5.loadmat(bm_path)['grid2d'], (2, 1, 0))
            bm=h5.loadmat(bm_path)['grid2d']
            if bm.shape[0]==2:
                bm = np.transpose(bm, (2, 1, 0))
            h_img, v_img, lbl, img , edge= self.transform_new(h_mask, v_mask, bm,im)

            lbl = lbl.permute((2, 0, 1))  # HWC -> CHW
        except Exception as e:
            print(f"Failed to read: {self.files[self.split][index]},{e}")
            return self[index + 1]

        return h_img, v_img, lbl, img, edge

    def infer_mat_from_metadata(self, img_stem, grid_dir, meta_dir):
        """从 metadata 推断对应的 .mat 文件"""
        meta_path = os.path.join(meta_dir, f"{img_stem}.json")
        if not os.path.isfile(meta_path):
            return None, None

        with open(meta_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        geom_name = info.get("geom_name") if isinstance(info, dict) else None
        if not geom_name:
            return None, None

        mat_path = os.path.join(grid_dir, f"{geom_name}.mat")
        if not os.path.isfile(mat_path):
            return None, geom_name

        return mat_path, geom_name

    def transform_new(self, h_mask, v_mask, grid2d, im):
        """
        h_mask / v_mask: 二值掩码 (H, W)
        bm: grid2d (Hc, Wc, 2)，backward map
        im: RGB 图像 (H, W, 3)
        """
        # print("grid2d:",grid2d.shape,grid2d)
        # print(grid2d[:, :, 0].min(), grid2d[:, :, 0].max())
        # print(grid2d[:, :, 1].min(), grid2d[:, :, 1].max())
        # print(im.shape)

        H, W, _ = im.shape
        edge = gradient(im)
        edge = edge[:, :, np.newaxis]

        if self.is_aug:
            if random.random() > 0.8:
                im = color_jitter(im, 0.2, 0.2, 0.6, 0.6)

        # ===============================
        # 1. Grid2D 本身就是 backward map
        # ===============================
        # grid2d[...,0] : x (W 方向)
        # grid2d[...,1] : y (H 方向)

        gx = grid2d[:, :, 0].astype(np.float32)
        gy = grid2d[:, :, 1].astype(np.float32)

        # 1. 先在稀疏点上完成坐标系切换
        gx = gx * (self.img_size / W)
        gy = gy * (self.img_size / H)

        # 2. densify
        gx = cv2.resize(gx, (self.img_size, self.img_size),
                        interpolation=cv2.INTER_LINEAR)
        gy = cv2.resize(gy, (self.img_size, self.img_size),
                        interpolation=cv2.INTER_LINEAR)

        # ===============================
        # 3. 构造 label（pixel coord）
        # ===============================
        lbl = torch.from_numpy(
            np.stack([gx, gy], axis=-1)
        ).float()

        # ===============================
        # 4. mask / img
        # ===============================
        h_img = cv2.resize(h_mask, (self.img_size, self.img_size))
        h_img = torch.from_numpy(h_img / 255.0).float().unsqueeze(0)

        v_img = cv2.resize(v_mask, (self.img_size, self.img_size))
        v_img = torch.from_numpy(v_img / 255.0).float().unsqueeze(0)

        img = cv2.resize(im, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        edge = cv2.resize(edge.astype(np.uint8), (self.img_size, self.img_size))
        edge = edge.astype(np.float64) / 255.
        edge = torch.from_numpy(edge).float().unsqueeze(0)

        return h_img, v_img, lbl, img, edge

def gradient(image):
    # 将彩色图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 对灰度图像进行高斯滤波
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 计算水平和垂直方向上的梯度值
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # 合并水平和垂直方向上的梯度值
    gradient = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 对梯度幅值进行归一化处理
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return gradient

def color_jitter(im, brightness=0, contrast=0, saturation=0, hue=0):
    im = im / 255.
    f = random.uniform(-brightness, brightness)
    im = np.clip(im + f, 0., 1.).astype(np.float32)

    f = random.uniform(1 - contrast, 1 + contrast)
    im = np.clip(im * f, 0., 1.)

    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    f = random.uniform(-hue, hue)
    hsv[0] = np.clip(hsv[0] + f * 360, 0., 360.)

    f = random.uniform(-saturation, saturation)
    hsv[2] = np.clip(hsv[2] + f, 0., 1.)
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    im = np.clip(im, 0., 1.)
    return im * 255.


if __name__ == '__main__':
    data_path = '/raid/lh/dataset/dewarp_Horizontal_Vertical/'
    dataset_train = Warp_DataSet(448, data_path, 'train', is_aug=True)  # Doc3d_train1
    train_loader = DataLoader(dataset_train, batch_size=8, shuffle=False, num_workers=1)
    for img, lbl in train_loader:
        print("-" * 50)
        print("img.shape: ", img.shape)
        print("lbl.shape: ", lbl.shape)
        print("-" * 50)
