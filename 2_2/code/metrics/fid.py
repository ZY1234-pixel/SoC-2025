import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from tqdm import tqdm

from models.archs.inception import InceptionV3


def load_patched_inception_v3(device='cuda',
                              resize_input=True,
                              normalize_input=False):
    # we may not resize the input, but in [rosinality/stylegan2-pytorch] it
    # does resize the input.
    inception = InceptionV3([3],
                            resize_input=resize_input,
                            normalize_input=normalize_input)
    inception = nn.DataParallel(inception).eval().to(device)
    return inception


@torch.no_grad()
def extract_inception_features(data_generator,
                               inception,
                               len_generator=None,
                               device='cuda'):
    """提取Inception特征
    参数：
        data_generator (generator)：数据生成
        inception (nn.Module)：Inception模型
        len_generator (int)：用于显示进度条的数据生成长度
            默认值：None。
        device (str)：设备。默认值：cuda
    返回值：
        Tensor：提取的特征
    """
    if len_generator is not None:
        pbar = tqdm(total=len_generator, unit='batch', desc='Extract')
    else:
        pbar = None
    features = []

    for data in data_generator:
        if pbar:
            pbar.update(1)
        data = data.to(device)
        feature = inception(data)[0].view(data.shape[0], -1)
        features.append(feature.to('cpu'))
    if pbar:
        pbar.close()
    features = torch.cat(features, 0)
    return features


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy实现的Frechet距离。
    两个多元高斯分布 X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    参数：
        mu1 (np.array)：激活值的样本均值。
        sigma1 (np.array)：生成样本的激活值协方差矩阵。
        mu2 (np.array)：激活值的样本均值，基于代表性数据集预先计算。
        sigma2 (np.array)：激活值的协方差矩阵，基于代表性数据集预先计算。
        sigma2 (np.array)：激活值协方差矩阵，
            基于代表性数据集预先计算。
    返回值：
        float：Frechet距离。
    """
    assert mu1.shape == mu2.shape, 'Two mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, (
        'Two covariances have different dimensions')

    cov_sqrt, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Product might be almost singular
    if not np.isfinite(cov_sqrt).all():
        print('Product of cov matrices is singular. Adding {eps} to diagonal '
              'of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))
            raise ValueError(f'Imaginary component {m}')
        cov_sqrt = cov_sqrt.real

    mean_diff = mu1 - mu2
    mean_norm = mean_diff @ mean_diff
    trace = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov_sqrt)
    fid = mean_norm + trace

    return fid
