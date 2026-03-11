import numpy as np

from utils.matlab_functions import bgr2ycbcr


def reorder_image(img, input_order='HWC'):
    """将图像重新排序为HWC顺序。
    若输入顺序为(h, w)，则返回(h, w, 1)；
    若输入顺序为(c, h, w)，则返回(h, w, c)；
    若输入顺序为(h, w, c)，则保持原样返回。
    参数：
        img (ndarray)：输入图像。
        input_order (str)：输入顺序为'HWC'或'CHW'。
            若输入图像形状为(h, w)，input_order将无效。
            默认值：‘HWC’。
    返回值：
        ndarray：重新排序后的图像。
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """切换至YCbCr的Y通道。
    参数：
        img (ndarray)：图像范围为[0, 255]。
    返回值：
        (ndarray)：图像范围为[0, 255]（float），不进行四舍五入。
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.
