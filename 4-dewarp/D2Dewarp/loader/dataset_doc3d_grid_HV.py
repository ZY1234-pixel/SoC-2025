import collections
import multiprocessing
import os
from time import time

from skimage import measure

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
from torch.utils.data import Dataset, DataLoader
import hdf5storage as h5
import random
from PIL import ImageFilter, Image
import torchvision.transforms as transforms


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

            im_path = os.path.join(self.data_root, 'warp_img', img_filename + '.png')
            bm_path = os.path.join(self.data_root, 'bm', img_filename.split('.')[0][:-4] + '_ann0001.mat')
            wc_path = os.path.join(self.data_root, 'wc', img_filename.split('.')[0][:-4] + '_ann0001.exr')
            h_line_path = os.path.join(self.data_root, 'alb_h', img_filename.split('.')[0][:-4] + '_ann0001.png')
            v_line_path = os.path.join(self.data_root, 'alb_v', img_filename.split('.')[0][:-4] + '_ann0001.png')

            wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            h, w, c = wc.shape

            img = np.array(Image.open(im_path).convert("RGB"))
            h, w, c = img.shape

            bm = h5.loadmat(bm_path)['bm']
            h, w, c = bm.shape

            h_line = cv2.imread(h_line_path)
            h, w, c = h_line.shape
            v_line = cv2.imread(v_line_path)
            h, w, c = v_line.shape

            img, lbl, h_line, v_line, edge = self.transform_new(wc, bm, img, h_line, v_line)

            lbl = lbl.permute((2, 0, 1))  # HWC -> CHW
        except Exception as e:
            print(f"Failed to read: {self.files[self.split][index]}")
            return self[index + 1]
        return img, lbl, h_line, v_line, edge

    def tight_crop(self, wc, img, h_line, v_line, edge):
        msk = ((wc[:, :, 0] != 0) & (wc[:, :, 1] != 0) & (wc[:, :, 2] != 0)).astype(np.uint8)
        size = msk.shape
        [y, x] = (msk).nonzero()
        crop_random = random.random()
        if crop_random > 0.55:
            minx = min(x) // 2
            maxx = size[1] - (size[1] - max(x)) // 2
            miny = min(y) // 2
            maxy = size[0] - (size[0] - max(y)) // 2

            s = 0
        else:
            minx = min(x)
            maxx = max(x)
            miny = min(y)
            maxy = max(y)
            s = random.randint(7, 25)

        wc = wc[miny: maxy + 1, minx: maxx + 1, :]
        img = img[miny: maxy + 1, minx: maxx + 1, :]
        h_line = h_line[miny: maxy + 1, minx: maxx + 1, :]
        v_line = v_line[miny: maxy + 1, minx: maxx + 1, :]
        edge = edge[miny: maxy + 1, minx: maxx + 1, :]

        wc = np.pad(wc, ((s, s), (s, s), (0, 0)), 'constant')
        img = np.pad(img, ((s, s), (s, s), (0, 0)), 'constant')
        h_line = np.pad(h_line, ((s, s), (s, s), (0, 0)), 'constant')
        v_line = np.pad(v_line, ((s, s), (s, s), (0, 0)), 'constant')
        edge = np.pad(edge, ((s, s), (s, s), (0, 0)), 'constant')

        t = miny - s# + cy1
        b = size[0] - maxy - s# + cy2
        l = minx - s# + cx1
        r = size[1] - maxx - s# + cx2

        return wc, img, h_line, v_line, edge, t, b, l, r

    def transform_new(self, wc, bm, img, h_line, v_line):
        edge = gradient(img)
        edge = edge[:, :, np.newaxis]

        if self.is_aug:
            if random.random() > 0.55:
                img = color_line(img, bm)
        if random.random() > 0.1:
            wc, img, h_line, v_line, edge, t, b, l, r = self.tight_crop(wc, img, h_line, v_line, edge)
        else:
            t, b, l, r = 0, 0, 0, 0

        # msk = ((wc[:, :, 0] != 0) & (wc[:, :, 1] != 0) & (wc[:, :, 2] != 0)).astype(np.uint8) * 255

        bm = bm.astype(float)
        bm[:, :, 1] = bm[:, :, 1] - t
        bm[:, :, 0] = bm[:, :, 0] - l
        bm = bm / np.array([512. - l - r, 512. - t - b])

        if self.is_aug:
            if random.random() > 0.8:
                img = color_jitter(img, 0.2, 0.2, 0.6, 0.6)

        bm0 = cv2.resize(bm[:, :, 0], (self.img_size, self.img_size))
        bm1 = cv2.resize(bm[:, :, 1], (self.img_size, self.img_size))
        bm = np.stack([bm0, bm1], axis=-1)

        bm = bm * self.img_size
        lbl = torch.from_numpy(bm).float()

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float64)

        if img.shape[2] == 4:
            img = img[:, :, :3]

        edge = cv2.resize(edge.astype(np.uint8), (self.img_size, self.img_size))
        edge = edge.astype(np.float64) / 255.
        edge = torch.from_numpy(edge).float().unsqueeze(0)

        img = img.astype(float) / 255.0  # 使用normalize归一化
        img = img.transpose(2, 0, 1)  # NHWC -> NCHW
        img = torch.from_numpy(img).float()

        h_line = cv2.resize(h_line, (self.img_size, self.img_size))
        h_line = h_line.astype(float) / 255.
        h_line = h_line.transpose(2, 0, 1)  # NHWC -> NCHW
        h_line = h_line[[0], :, :]
        h_line = torch.from_numpy(h_line).float()

        v_line = cv2.resize(v_line, (self.img_size, self.img_size))
        v_line = v_line.astype(float) / 255.
        v_line = v_line.transpose(2, 0, 1)  # NHWC -> NCHW
        v_line = v_line[[0], :, :]
        v_line = torch.from_numpy(v_line).float()

        return img, lbl, h_line, v_line, edge


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


def color_line(im, bm):
    # im = im# * 255
    # bm = bm * 448
    chance = random.random()
    if chance < 0.8:
        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        t = bm[random.randint(2, 18):random.randint(20, 40), :, :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)
        if random.random() > 0.7:
            c = np.array([1, 1, 1]) * 255.0
            t = bm[:random.randint(1, 10), :, :].reshape([-1, 2])
            for j in range(len(t)):
                cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)
    elif chance > 0.3 and chance < 0.6:
        cc = random.randint(2, 18)
        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        t = bm[:, random.randint(2, 18):random.randint(20, 40), :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)

        if random.random() > 0.7:
            c = np.array([1, 1, 1]) * 255.0
            t = bm[:, :random.randint(1, 10), :].reshape([-1, 2])
            for j in range(len(t)):
                cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)

    chance = random.random()
    if chance < 0.8:
        c = np.array([0, 0, 0]) * 255.0
        t = bm[25:random.randint(30, 40), :random.randint(112, 224), :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)

        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        t = bm[:, :10, :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, [255, 255, 255], thickness=1)

    chance = random.random()
    if chance < 0.1:
        im[:, :, 0] = random.random() * 255.0
    elif chance < 0.2 and chance > 0.1:
        im[:, :, 1] = random.random() * 255.0

    elif chance < 0.6 and chance > 0.4:
        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        t = bm[:random.randint(20, 45), :, :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)
    elif chance < 0.8 and chance > 0.6:
        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        t = bm[:, :random.randint(20, 45), :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)

    chance = random.random()
    if random.random() > 0.4:
        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        t = bm[:, :random.randint(1, 20), :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)

    chance = random.random()
    if random.random() > 0.4:
        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        t = bm[:random.randint(1, 20), :, :].reshape([-1, 2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)

    chance = random.random()
    if chance > 0.4:
        c = np.array([random.random(), random.random(), random.random()]) * 255.0
        num = int(random.random() * 20)
        cc = random.randint(10, 15)
        for m in range(30):
            t = bm[num:num + 20, 50 + cc * m:cc * m + 57, :].reshape([-1, 2])
            for j in range(len(t)):
                cv2.circle(im, (int(t[j, 0]), int(t[j, 1])), 1, c, thickness=1)

    chance = random.random()
    if chance > 0.9:
        im = 255 - im
    elif chance > 0.85:
        im[:, :, 0] = 255
    elif chance > 0.8:
        im[:, :, 0] = 255
    elif chance > 0.75:
        im[:, :, 0] = 0
    elif chance > 0.7:
        im[:, :, 1] = 255
    elif chance > 0.65:
        im[:, :, 1] = 0
    elif chance > 0.6:
        im[:, :, 2] = 255
    elif chance > 0.55:
        im[:, :, 2] = 0

    return im# / 255


def img_flip(im, bm, fore_mask, text_line):
    bm = bm * 2 - 1  # [-1, 1]
    chance = random.random()
    if chance > 0.8:
        im = cv2.flip(im, 0)
        text_line = cv2.flip(text_line, 0)
        fore_mask = cv2.flip(fore_mask, 0)
        bm[:, :, 0] = cv2.flip(bm[:, :, 0], 0)
        bm[:, :, 1] = -cv2.flip(bm[:, :, 1], 0)

    elif 0.6 < chance <= 0.8:
        im = cv2.flip(im, 1)
        text_line = cv2.flip(text_line, 1)
        fore_mask = cv2.flip(fore_mask, 1)
        bm[:, :, 0] = -cv2.flip(bm[:, :, 0], 1)
        bm[:, :, 1] = cv2.flip(bm[:, :, 1], 1)

    elif 0.4 < chance <= 0.6:
        im = cv2.flip(im, 0)
        im = cv2.flip(im, 1)
        text_line = cv2.flip(text_line, 0)
        text_line = cv2.flip(text_line, 1)

        fore_mask = cv2.flip(fore_mask, 0)
        fore_mask = cv2.flip(fore_mask, 1)

        bm[:, :, 0] = cv2.flip(bm[:, :, 0], 0)
        bm[:, :, 1] = -cv2.flip(bm[:, :, 1], 0)
        bm[:, :, 0] = -cv2.flip(bm[:, :, 0], 1)
        bm[:, :, 1] = cv2.flip(bm[:, :, 1], 1)

    bm = (bm + 1) / 2  # [0, 1]
    return im, bm, fore_mask, text_line


def create_mask(fore_bg_mask, width, height):
    mask_new = []

    # 右侧
    mask_right = fore_bg_mask.copy()  #

    mask_right_x = random.randint(width - np.random.randint(30, 80), width)
    mask_right[:, mask_right_x:width] = np.random.rand(height, width - mask_right_x)

    # 左侧
    mask_left = fore_bg_mask.copy()
    mask_left_x = random.randint(width - np.random.randint(30, 80), width)
    mask_left[:, 0:width - mask_left_x] = np.random.rand(height, width - mask_left_x)

    # 下侧
    mask_bottom = fore_bg_mask.copy()
    mask_bottom_y = random.randint(height - np.random.randint(30, 80), height)
    mask_bottom[mask_bottom_y:height, :] = np.random.rand(height - mask_bottom_y, width)

    # 上侧
    mask_top = fore_bg_mask.copy()
    mask_top_y = random.randint(height - np.random.randint(30, 80), height)
    mask_top[0:height - mask_top_y, :] = np.random.rand(height - mask_top_y, width)

    mask_new.append(mask_right)
    mask_new.append(mask_bottom)
    mask_new.append(mask_top)
    mask_new.append(mask_left)

    return random.choice(mask_new)


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


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def compute_connected_components(image):
    # 读入图像
    # image = color.rgb2gray(image)
    # 二值化操作
    # threshold = np.mean(image)
    # binary = image > threshold

    # 进行连通域分割
    labels = measure.label(image)
    regions = measure.regionprops(labels)
    # 剔除面积较小的连通域
    clean_regions = []
    for region in regions:
        if region.area > 75:  # 75
            clean_regions.append(region)
    # 得到分割后的图像
    mask = np.zeros_like(image)
    for region in clean_regions:
        for coord in region.coords:
            mask[coord[0], coord[1]] = 1
    # result = morphology.binary_dilation(mask, morphology.square(2))
    # result.astype(int)
    return mask + 0


def pred_textline(data_path, alb_root, img_filename, unet):

    alb_path = os.path.join(alb_root, img_filename + '.png')
    input = cv2.imread(alb_path) / 255.
    binary_threshold = 0.2

    w, h, c = input.shape
    # input = cv2.resize(input, (512, 512))
    print(input.shape)
    input = torch.from_numpy(input).permute(2, 0, 1).unsqueeze(0).cuda()
    pred_mask = unet(input.float())
    pred_mask = (pred_mask > binary_threshold)
    pred_mask = ((pred_mask.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0))).astype(np.uint8)
    # pred_mask = compute_connected_components(pred_mask)
    # pred_mask = cv2.resize(pred_mask, (896, 896))

    save_dir_h = data_path + "textline/" + img_filename.split('/', 1)[0]
    if not os.path.exists(save_dir_h):
        os.makedirs(save_dir_h)
    #
    save_name_h = os.path.join(save_dir_h, img_filename.split('/', 1)[1] + '.png')
    #
    pred_mask = cv2.resize(pred_mask, (w, h))
    cv2.imwrite(save_name_h, pred_mask * 255.)


def worker(img_file_lst, data_path, alb_root, unet):

    # pool = multiprocessing.Pool(processes=16)
    for img_filename in img_file_lst:
        # pool.apply_async(func=pred_textline, args=(data_path, alb_root, img_filename, unet, f_crop_tlbr))
        pred_textline(data_path, alb_root, img_filename, unet)

    # pool.close()
    # pool.join()


if __name__ == '__main__':
    data_path = '/raid/lh/dataset/dewarp_Horizontal_Vertical/'
    dataset_train = Warp_DataSet(448, data_path, 'train', is_aug=True)  # Doc3d_train1
    train_loader = DataLoader(dataset_train, batch_size=8, shuffle=False, num_workers=1)
    for img, lbl, h_line, v_line, edge in train_loader:
        start = time()
        print("-" * 50)
        print("img.shape: ", img.shape)
        print("lbl.shape: ", lbl.shape)
        print("h_line.shape: ", h_line.shape)
        print("v_line.shape: ", v_line.shape)
        print("edge.shape: ", edge.shape)

        print("---time: ", time() - start)
        print("-" * 50)
    # -----------------
    # predict textline
    # unet = UNet(n_channels=3, n_classes=1)
    # unet.load_state_dict(torch.load('../pretrianed_models/unet.pth', map_location='cpu'))  # unet_v_30000_iter
    # unet = unet.cuda()
    # unet.eval()
    # img_file_lst = open(data_path + 'train.txt', 'r', encoding='utf-8').read().splitlines()
    # print("le