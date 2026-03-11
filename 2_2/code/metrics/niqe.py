import cv2
import math
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.special import gamma

from metrics.metric_util import reorder_image, to_y_channel


def estimate_aggd_param(block):
    """估计AGGD（非对称广义高斯分布）参数。
    参数：
        block (ndarray)：二维图像块。
    返回值：
        tuple：AGGD分布的alpha（float）、beta_l（float）和beta_r（float）参数
    """
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (
        gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0]**2))
    right_std = np.sqrt(np.mean(block[block > 0]**2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block)))**2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) *
                (gammahat + 1)) / ((gammahat**2 + 1)**2)
    array_position = np.argmin((r_gam - rhatnorm)**2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)


def compute_feature(block):
    """计算特征。
    参数：
        block (ndarray)：二维图像块。
    返回值：
        list：长度为18的特征向量。
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    # 畸变会扰乱自然图像中较为规则的结构。
    # 这种偏差可通过分析沿水平、垂直及对角方向计算的
    # 相邻系数对乘积的样本分布来捕捉。
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = np.roll(block, shifts[i], axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def niqe(img,
         mu_pris_param,
         cov_pris_param,
         gaussian_window,
         block_size_h=96,
         block_size_w=96):
    """计算NIQE（自然图像质量评估器）指标。
    参数：
        img (ndarray)：需计算质量的输入图像。该图像必须为灰度或Y通道（来自YCbCr）图像，形状为(h, w)。
            取值范围[0, 254]，浮点类型。
        mu_pris_param (ndarray)：基于原始数据集计算的预定义多元高斯模型的均值。
        cov_pris_param (ndarray)：基于原始数据集计算的预定义多元高斯模型协方差。
        gaussian_window (ndarray)：用于图像平滑的7x7高斯窗。
        block_size_h (int)：图像划分的块高度。
            default：96。
        block_size_w (整数)：图像划分的块宽度。
            default：96。
    """
    assert img.ndim == 2, (
        'Input image must be a gray or Y (of YCbCr) image with shape (h, w).')
    # crop image
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        mu = convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(
            np.abs(
                convolve(np.square(img), gaussian_window, mode='nearest') -
                np.square(mu)))
        # normalize, as in Eq. 1 in the paper
        img_nomalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process ecah block
                block = img_nomalized[idx_h * block_size_h //
                                      scale:(idx_h + 1) * block_size_h //
                                      scale, idx_w * block_size_w //
                                      scale:(idx_w + 1) * block_size_w //
                                      scale]
                feat.append(compute_feature(block))

        distparam.append(np.array(feat))

        if scale == 1:
            h, w = img.shape
            img = cv2.resize(
                img / 255., (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
            img = img * 255.

    distparam = np.concatenate(distparam, axis=1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = np.nanmean(distparam, axis=0)
    # use nancov. ref: https://ww2.mathworks.cn/help/stats/nancov.html
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(
        np.matmul((mu_pris_param - mu_distparam), invcov_param),
        np.transpose((mu_pris_param - mu_distparam)))
    quality = np.sqrt(quality)

    return quality


def calculate_niqe(img, crop_border, input_order='HWC', convert_to='y'):
    """计算NIQE（自然图像质量评估器）指标。
    参数：
        img (ndarray)：需计算质量的输入图像。输入图像必须为浮点/整数类型，数值范围[0, 255]。
            图像输入顺序可为'HW'、‘HWC'或'CHW’。(BGR序列)
            若输入序列为'HWC'或'CHW'，将根据``convert_to``参数转换为灰度或Y通道(YCbCr中的Y通道)图像。
        crop_border (int)：图像每条边界裁剪的像素数。这些像素不参与指标计算。
        input_order (str)：输入序列类型为'HW'、‘HWC'或'CHW’。
            默认值：‘HWC’。
        convert_to (str)：是否转换为YCbCr中的'y'或'gray'。
            默认值：‘y’。
    返回值：
        float：NIQE计算结果。
    """

    niqe_pris_params = np.load('basicsr/metrics/niqe_pris_params.npz')
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']

    img = img.astype(np.float32)
    if input_order != 'HW':
        img = reorder_image(img, input_order=input_order)
        if convert_to == 'y':
            img = to_y_channel(img)
        elif convert_to == 'gray':
            img = cv2.cvtColor(img / 255., cv2.COLOR_BGR2GRAY) * 255.
        img = np.squeeze(img)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]

    niqe_result = niqe(img, mu_pris_param, cov_pris_param, gaussian_window)

    return niqe_result
