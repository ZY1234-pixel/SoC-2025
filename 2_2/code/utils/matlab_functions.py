import math
import numpy as np
import paddle


def cubic(x):
    """cubic function used for calculate_weights_indices."""
    absx = paddle.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx +
                                      2) * (((absx > 1) *
                                             (absx <= 2)).type_as(absx))


def calculate_weights_indices(in_length, out_length, scale, kernel,
                              kernel_width, antialiasing):
    """计算权重和索引，用于图像缩放函数。
    参数：
        in_length (int)：输入长度。
        out_length (int)：输出长度。
        scale (float)：缩放因子。
        kernel_width (int)：卷积核宽度。
        antialisaing (bool)：下采样时是否应用抗锯齿处理。
    """

    if (scale < 1) and antialiasing:
        # Use a modified kernel (larger kernel width) to simultaneously
        # interpolate and antialias
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = paddle.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5 + scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = paddle.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    p = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.reshape([out_length, 1]).expand([out_length, p]) + paddle.linspace(
        0, p - 1, p).reshape([1, p]).expand([out_length, p])

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.reshape([out_length, 1]).expand([out_length, p]) - indices

    # apply cubic kernel
    if (scale < 1) and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)

    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = paddle.sum(weights, 1).reshape([out_length, 1])
    weights = weights / weights_sum.expand([out_length, p])

    # If a column in weights is all zero, get rid of it. only consider the
    # first and last column.
    weights_zero_tmp = paddle.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


@paddle.no_grad()
def imresize(img, scale, antialiasing=True):
    """
    参数：
        img (Tensor | Numpy array)：
            Tensor：输入图像，形状为(c, h, w)，取值范围[0, 1]。
            Numpy：输入图像形状为(h, w, c)，取值范围[0, 1]。
        scale (float)：缩放因子。高度与宽度采用相同比例。
        antialisaing (bool)：下采样时是否启用抗锯齿。
            默认值：True。
    返回值：
        Tensor：输出图像，形状为 (c, h, w)，取值范围 [0, 1]，不进行四舍五入。
    """
    if type(img).__module__ == np.__name__:  # numpy type
        numpy_type = True
        img = paddle.to_tensor(img.transpose(2, 0, 1).astype('float32'))
    else:
        numpy_type = False

    in_c, in_h, in_w = img.shape
    out_h, out_w = math.ceil(in_h * scale), math.ceil(in_w * scale)
    kernel_width = 4
    kernel = 'cubic'

    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = calculate_weights_indices(
        in_h, out_h, scale, kernel, kernel_width, antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = calculate_weights_indices(
        in_w, out_w, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(img)

    sym_patch = img[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(
                0, 1).mv(weights_h[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :,
                                       idx:idx + kernel_width].mv(weights_w[i])

    if numpy_type:
        out_2 = out_2.numpy().transpose(1, 2, 0)
    return out_2


def rgb2ycbcr(img, y_only=False):
    """将RGB图像转换为YCbCr图像。
    参数：
        img (ndarray)：输入图像。接受以下类型：
            1. np.uint8 类型，取值范围 [0, 255]；
            2. np.float32 类型，取值范围 [0, 1]。
        y_only (bool)：是否仅返回Y通道。默认值：False。
    返回值：
        ndarray：转换后的YCbCr图像。输出图像与输入图像具有相同的类型和取值范围。
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = np.matmul(
            img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                  [24.966, 112.0, -18.214]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def bgr2ycbcr(img, y_only=False):
    """将BGR图像转换为YCbCr图像。
    参数：
        img (ndarray)：输入图像。支持以下类型：
            1. np.uint8 类型，取值范围 [0, 255]；
            2. np.float32 类型，取值范围 [0, 1]。
        y_only (bool)：是否仅返回Y通道。默认值：False。
    返回值：
        ndarray：转换后的YCbCr图像。输出图像与输入图像具有相同的类型和取值范围。
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2rgb(img):
    """将YCbCr图像转换为RGB图像。
    参数：
        img (ndarray)：输入图像。支持以下类型：
            1. np.uint8 类型，取值范围 [0, 255]；
            2. np.float32 类型，取值范围 [0, 1]。
    返回值：
        ndarray：转换后的 RGB 图像。输出图像与输入图像具有相同的类型和取值范围。
    """
    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621],
                              [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [
                                  -222.921, 135.576, -276.836
                              ]  # noqa: E126
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2bgr(img):
    """将YCbCr图像转换为BGR图像。
    参数：
        img (ndarray)：输入图像。支持以下类型：
            1. np.uint8 类型，取值范围 [0, 255]；
            2. np.float32 类型，取值范围 [0, 1]。
    返回值：
        ndarray：转换后的 BGR 图像。输出图像与输入图像具有相同的类型和取值范围。
    """
    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621],
                              [0.00791071, -0.00153632, 0],
                              [0, -0.00318811, 0.00625893]]) * 255.0 + [
                                  -276.836, 135.576, -222.921
                              ]  # noqa: E126
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def _convert_input_type_range(img):
    """转换输入图像的类型和范围。
    该函数将输入图像转换为 np.float32 类型，范围为 [0, 1]。
    主要用于在颜色空间转换函数（如 rgb2ycbcr 和 ycbcr2rgb）中预处理输入图像。
    参数：
        img (ndarray)： 输入图像。支持以下格式：
            1. np.uint8 类型，范围 [0, 255]；
            2. np.float32 类型，范围 [0, 1]。
    返回值：
        (ndarray)：转换后的图像，类型为 np.float32，范围为[0, 1]。
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, '
                        f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """根据目标类型将图像的类型和范围进行转换。
    该操作将图像转换为指定类型和范围。若 `dst_type` 为 np.uint8，
    则图像将转换为 np.uint8 类型，范围为 [0, 255]。若`dst_type` 为 np.float32，则图像将转换为 np.float32 类型，[0, 1]。
    主要用于在色彩空间转换函数（如 rgb2ycbcr 和 ycbcr2rgb）中对图像进行后处理。
    参数：
        img (ndarray)：待转换的图像，转换后为 np.float32 类型且范围为 [0, 255]。
        dst_type (np.uint8 | np.float32)：若dst_type为np.uint8，则
            将图像转换为np.uint8类型，范围[0, 255]。若dst_type为np.float32，则将图像转换为np.float32类型，范围[0, 1]。
    返回值：
        (ndarray)：转换后具有指定类型和范围的图像。
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, '
                        f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)
