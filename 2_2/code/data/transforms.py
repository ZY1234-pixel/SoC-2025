import cv2
import random
import numpy as np

def mod_crop(img, scale):
    """修剪图像，用于测试阶段。
    参数：
        img (ndarray)：输入图像。
        scale (int)：缩放因子。
    返回值：
        ndarray：结果图像。
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img

def paired_random_crop(img_gts, img_lqs, lq_patch_size, scale, gt_path):
    """成对随机裁剪
    该函数裁剪包含lq图像和gt图像及其对应位置的列表。
    参数：
        img_gts (list[ndarray] | ndarray)：GT图像。注意所有图像
            应具有相同形状。若输入为ndarray，则将其转换为包含自身的列表。
        img_lqs (list[ndarray] | ndarray)：低质量图像。注意所有图像
            应具有相同形状。若输入为ndarray，则转换为包含自身的列表。
        lq_patch_size (int): 低质量图像裁剪尺寸。
        scale (int): 缩放因子。
        gt_path (str): 地面实测数据路径。
    返回值：
        list[ndarray] | ndarray: 地面实测图像与低质量图像。若返回结果仅含单个元素，则直接返回ndarray。
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    gt_patch_size = int(lq_patch_size * scale)

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

def paired_random_crop_DP(img_lqLs, img_lqRs, img_gts, gt_patch_size, scale, gt_path):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqLs, list):
        img_lqLs = [img_lqLs]
    if not isinstance(img_lqRs, list):
        img_lqRs = [img_lqRs]

    h_lq, w_lq, _ = img_lqLs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqLs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqLs
    ]

    img_lqRs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqRs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqLs) == 1:
        img_lqLs = img_lqLs[0]
    if len(img_lqRs) == 1:
        img_lqRs = img_lqRs[0]
    return img_lqLs, img_lqRs, img_gts


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """增强操作：水平翻转或旋转（0, 90, 180, 270度）。
    采用垂直翻转与转置实现旋转功能。
    列表中所有图像均采用相同的增强方案。
    参数：
        imgs (list[ndarray] | ndarray)：待增强的图像。若输入为 ndarray，将转换为列表。
        hflip (bool)：水平翻转。默认值：True。
        rotation (bool)：旋转。默认值：True。
        flows (list[ndarray])：待增强的流。若输入为 ndarray，将转换为列表。
            维度为 (h, w, 2)。默认值：无。
        return_status (bool)：返回翻转和旋转的状态。
            默认值：False。
    返回值：
        list[ndarray] | ndarray：增强后的图像和流。若返回结果仅含一个元素，则直接返回 ndarray。
    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """旋转图像
    参数：
        img (ndarray)：待旋转的图像。
        angle (float)：旋转角度（单位：度）。正值表示
            逆时针旋转。
        中心点 (tuple[int]): 旋转中心点。若中心点为 None，则初始化为图像中心。默认值：None。
        缩放因子 (float)：各向同性缩放因子。默认值：1.0。
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img

def data_augmentation(image, mode):
    """对输入图像执行数据增强
    输入：
        image：cv2（OpenCV）图像
        mode：整数。选择对图像应用的变换方式
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0,7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out
