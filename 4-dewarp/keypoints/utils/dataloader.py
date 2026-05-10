import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class DeeplabDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path, keypoint_json_path=None, num_keypoints=0):
        super(DeeplabDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

        self.keypoint_json_path = keypoint_json_path   # e.g. "VOCdevkit/VOC2007/Keypoints"
        self.num_keypoints = num_keypoints
        self.keypoint_dict = None
        if self.keypoint_json_path and os.path.exists(self.keypoint_json_path):
            with open(self.keypoint_json_path, 'r') as f:
                self.keypoint_dict = json.load(f)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        jpg = Image.open(os.path.join(self.dataset_path, "JPEGImages", name + ".jpg"))
        png = Image.open(os.path.join(self.dataset_path, "SegmentationClass", name + ".png"))

        # 读取关键点
        keypoints = self.load_keypoints(name)  # 返回 (4,2) 或 None

        # 增强（注意 get_random_data 内部要支持 keypoints）
        jpg, png, keypoints = self.get_random_data(jpg, png, keypoints, self.input_shape, random=self.train)

        # 生成热力图真值
        if keypoints is not None and self.num_keypoints > 0:
            from utils.keypoint_utils import generate_heatmap
            kpt_heatmap = generate_heatmap(self.input_shape, keypoints, sigma=3.0)
        else:
            kpt_heatmap = np.zeros((self.num_keypoints, self.input_shape[0], self.input_shape[1]), dtype=np.float32)

        # 图像预处理
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels, kpt_heatmap

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, keypoints, input_shape, jitter=0.1, hue=.1, sat=0.7, val=0.3, random=False):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', (w, h), (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))

            if keypoints is not None:
                keypoints[:, 0] = keypoints[:, 0] * (nw / iw) + (w - nw) // 2
                keypoints[:, 1] = keypoints[:, 1] * (nh / ih) + (h - nh) // 2
            return new_image, new_label, keypoints

        # 随机缩放+扭曲
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.9, 1.1)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        if keypoints is not None:
            keypoints[:, 0] *= (nw / iw)
            keypoints[:, 1] *= (nh / ih)

        # 翻转
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            if keypoints is not None:
                keypoints[:, 0] = nw - keypoints[:, 0]
                # 水平翻转后，左右角点交换通道：0<->1, 2<->3
                keypoints[0], keypoints[1] = keypoints[1].copy(), keypoints[0].copy()
                keypoints[2], keypoints[3] = keypoints[3].copy(), keypoints[2].copy()

        # 随机偏移 pad
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        if keypoints is not None:
            keypoints[:, 0] += dx
            keypoints[:, 1] += dy
        image = new_image
        label = new_label

        # 转换为 numpy
        image_data = np.array(image, np.uint8)

        # 高斯模糊
        blur = self.rand() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        # 旋转
        rotate = self.rand() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            if keypoints is not None:
                ones = np.ones((keypoints.shape[0], 1))
                pts = np.concatenate([keypoints, ones], axis=1)
                keypoints = np.dot(pts, M.T)
            label = Image.fromarray(label)

        # HSV 色域变换
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label, keypoints

    def load_keypoints(self, image_name):
        if self.keypoint_dict is None or self.num_keypoints == 0:
            return None
        # 尝试匹配: image_name, image_name.png, image_name.jpg
        for key in (image_name, image_name + ".png", image_name + ".jpg"):
            if key in self.keypoint_dict:
                pts = np.array(self.keypoint_dict[key], dtype=np.float32)
                if pts.shape[0] == self.num_keypoints:
                    return pts
        return None


# DataLoader中collate_fn使用
def deeplab_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    kpt_heatmaps = []
    for img, png, labels, kpt_hm in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
        kpt_heatmaps.append(kpt_hm)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    kpt_heatmaps = torch.from_numpy(np.array(kpt_heatmaps)).type(torch.FloatTensor)
    return images, pngs, seg_labels, kpt_heatmaps
