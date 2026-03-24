import os
import cv2
import numpy as np

def imfrombytes(content, flag='color', float32=False):
    """从字节读取图像。
    参数：
        content (bytes)：从文件或其他流获取的图像字节。
        标志 (str): 指定加载图像颜色类型的标志，
            可选值为 `color`、`grayscale` 和 `unchanged`。
        float32 (bool): 是否转换为 float32 类型。若为 True，将同时归一化至 [0, 1] 范围。
            默认值：False。
    返回值：
        ndarray: 加载后的图像数组。
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    if img_np is None:
        raise Exception('None .. !!!')
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def padding(img_lq, img_gt, gt_size):
    h, w, _ = img_lq.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)

    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    if img_gt.ndim == 2:
        img_gt = np.expand_dims(img_gt, axis=2)
    return img_lq, img_gt


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy数组转Tensor。
    参数：
        imgs (list[ndarray] | ndarray)：输入图像。
        bgr2rgb (bool)：是否将BGR转换为RGB。
        float32 (bool)：是否转换为float32类型。

    返回值：
        list[tensor] | tensor：张量图像。若返回结果仅含
            一个元素，则直接返回 tensor。
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        if float32:
            img = img.astype("float32")
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """将图像写入文件。
    参数：
        img (ndarray)：待写入的图像数组。
        file_path (str): 图像文件路径。
        params (None or list): 与 opencv 的 :func:`imwrite` 接口相同。
        auto_mkdir (bool): 若 `file_path` 的父文件夹不存在，
            是否自动创建。
    返回值：
        bool: 操作是否成功。
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """将Torch张量转换为图像numpy数组。
    在限制到[min, max]后，数值将归一化为[0, 1]。
    参数：
        tensor (Tensor or list[Tensor])：支持以下形状：
            1) 4D 小批量张量，形状为 (B x 3/1 x H x W)；
            2) 3D 张量，形状为 (3/1 x H x W)；
            3) 2D 张量，形状为 (H x W)。
            张量通道应遵循RGB顺序。
        rgb2bgr (bool)：是否将rgb转换为bgr。
        out_type (numpy type)：输出类型。若为``np.uint8``，则将输出转换为
            uint8类型[0, 255]；否则转换为浮点类型[0, 1]。默认值：``np.uint8``。
        min_max (tuple[int])：用于限制的最小值和最大值。

    返回值：
        (Tensor or list)：形状为 (H x W x C) 的 3D ndarray 或形状为 (H x W) 的 2D ndarray。通道顺序为 BGR。
    """

    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).detach().clip(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()
        if n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result