import paddle
from paddle import nn as nn
from paddle.nn import functional as F
import numpy as np

from models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Layer):
    """L1（均方根误差，MAE）损失。
    参数：
        loss_weight (float)：L1损失的权重系数。默认值：1.0。
        reduction (str)：指定应用于输出的缩减方式。
            支持选项为 ‘none’ | ‘mean’ | ‘sum’。默认值：‘mean’。
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        参数：
            pred (tensor)：形状为 (N, C, H, W)。预测tensor。
            target (tensor)：形状为 (N, C, H, W)。真实值tensor。
            weight (tensor，可选)：形状为 (N, C, H, W)。元素级权重。
                默认值：无。
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Layer):
    """均方误差（L2）损失。
    参数：
        loss_weight (float)：均方误差损失的权重系数。默认值：1.0。
        reduction (str)：指定应用于输出的缩减方式。
            支持选项为 ‘none’ | ‘mean’ | ‘sum’。默认值：‘mean’。
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Layer):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = paddle.to_tensor(np.array([65.481, 128.553, 24.966])).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.shape) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef
                self.first = False

            pred = (pred * self.coef).sum(axis=1).unsqueeze(axis=1) + 16.
            target = (target * self.coef).sum(axis=1).unsqueeze(axis=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.shape) == 4

        return self.loss_weight * self.scale * paddle.log(((pred - target) ** 2).mean(axis=(1, 2, 3)) + 1e-8).mean()

def gaussian1d(window_size, sigma):
    ###window_size = 11
    x = paddle.arange(window_size,dtype='float32')
    x = x - window_size//2
    gauss = paddle.exp(-x ** 2 / float(2 * sigma ** 2))
    # print('gauss.size():', gauss.size())
    ### torch.Size([11])
    return gauss / gauss.sum()

def create_window(window_size, sigma, channel):
    _1D_window = gaussian1d(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    # print('2d',_2D_window.shape)
    # print(window_size, sigma, channel)
    return _2D_window.expand([channel,1,window_size,window_size])

def _ssim(img1, img2, window, window_size, channel=3 ,data_range = 255.,size_average=True,C=None):
    # size_average for different channel

    padding = window_size // 2

    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
    # print(mu1.shape)
    # print(mu1[0,0])
    # print(mu1.mean())
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
    if C ==None:
        C1 = (0.01*data_range) ** 2
        C2 = (0.03*data_range) ** 2
    else:
        C1 = (C[0]*data_range) ** 2
        C2 = (C[1]*data_range) ** 2
    # l = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    # ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    sc = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    lsc = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1))*sc

    if size_average:
        ### ssim_map.mean()是对这个tensor里面的所有的数值求平均
        return lsc.mean()
    else:
        # ## 返回各个channel的值
        return lsc.flatten(2).mean(-1),sc.flatten(2).mean(-1)

def ms_ssim(
    img1, img2,window, data_range=255, size_average=True, window_size=11, channel=3, sigma=1.5, weights=None, C=(0.01, 0.03)
):

    r""" ms-ssim接口
    参数：
        img1 (torch.Tensor)：图像批次，(N,C,[T,]H,W)
        img2 (torch.Tensor)：图像批次，(N,C,[T,]H,W)
        data_range (float or int, 可选)：输入图像的值域（通常为1.0或255）
        size_average (bool，可选)：若size_average=True，则所有图像的ssim值将取平均值作为标量
        win_size: (int，可选)：高斯核的尺寸
        win_sigma：（float，可选）：正态分布的标准差
        win（torch.Tensor，可选）：一维高斯核。若为None，将根据win_size和win_sigma创建新核
        weights（list，可选）：不同层级的权重
        K（list or tuple，可选）：标量常量（K1, K2）。若结果为负值或NaN，请尝试增大K2常数（例如0.4）。
    返回值：
        torch.Tensor：ms-ssim计算结果
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images should have the same dimensions.")

    # for d in range(len(img1.shape) - 1, 1, -1):
    #     img1 = img1.squeeze(dim=d)
    #     img2 = img2.squeeze(dim=d)

    if not img1.dtype == img2.dtype:
        raise ValueError("Input images should have the same dtype.")

    if len(img1.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(img1.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {img1.shape}")

    smaller_side = min(img1.shape[-2:])

    assert smaller_side > (window_size - 1) * (2 ** 4), "Image size should be larger than %d due to the 4 downsamplings " \
                                                        "with window_size %d in ms-ssim" % ((window_size - 1) * (2 ** 4),window_size)

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = paddle.to_tensor(weights)

    if window is None:
        window = create_window(window_size, sigma, channel)
    assert window.shape == [channel, 1, window_size, window_size], " window.shape error"

    levels = weights.shape[0] # 5
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs =  _ssim(img1, img2, window=window, window_size=window_size,
                                       channel=3, data_range=data_range,C=C, size_average=False)
        if i < levels - 1:
            mcs.append(F.relu(cs))
            padding = [s % 2 for s in img1.shape[2:]]
            img1 = avg_pool(img1, kernel_size=2, padding=padding)
            img2 = avg_pool(img2, kernel_size=2, padding=padding)

    ssim_per_channel = F.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = paddle.stack(mcs + [ssim_per_channel], axis=0)  # (level, batch, channel) 按照等级堆叠
    ms_ssim_val = paddle.prod(mcs_and_ssim ** weights.reshape([-1, 1, 1]), axis=0) # level 相乘
    #print(ms_ssim_val.shape)
    if size_average:
        return ms_ssim_val.mean()
    else:
        # 返回各个channel的值
        return ms_ssim_val.flatten(2).mean(1)

class SSIMLoss(paddle.nn.Layer):
   """
   1. 继承paddle.nn.Layer
   """
   def __init__(self, window_size=11, channel=3, data_range=255., sigma=1.5):
       """
       2. 构造函数根据自己的实际算法需求和使用需求进行参数定义即可
       """
       super(SSIMLoss, self).__init__()
       self.data_range = data_range
       self.C = [0.01, 0.03]
       self.window_size = window_size
       self.channel = channel
       self.sigma = sigma
       self.window = create_window(self.window_size, self.sigma, self.channel)
       # print(self.window_size,self.window.shape)
   def forward(self, input, label):
       """
       3. 实现forward函数，forward在调用时会传递两个参数：input和label
           - input：单个或批次训练数据经过模型前向计算输出结果
           - label：单个或批次训练数据对应的标签数据
           接口返回值是一个Tensor，根据自定义的逻辑加和或计算均值后的损失
       """
       # 使用Paddle中相关API自定义的计算逻辑
       # output = xxxxx
       # return output
       return self.loss_weight * (1-_ssim(input, label,data_range = self.data_range,
                      window = self.window, window_size=self.window_size, channel=3,
                      size_average=True,C=self.C))



class MS_SSIMLoss(paddle.nn.Layer):
   """
   1. 继承paddle.nn.Layer
   """
   def __init__(self, loss_weight, reduction='mean', data_range=255., channel=3, window_size=11, sigma=1.5):
       """
       2. 构造函数根据自己的实际算法需求和使用需求进行参数定义即可
       """
       super(MS_SSIMLoss, self).__init__()
       self.data_range = data_range
       self.C = [0.01, 0.03]
       self.window_size = window_size
       self.channel = channel
       self.sigma = sigma
       self.window = create_window(self.window_size, self.sigma, self.channel)
       self.loss_weight = loss_weight
       # print(self.window_size,self.window.shape)
   def forward(self, input, label):
       """
       3. 实现forward函数，forward在调用时会传递两个参数：input和label
           - input：单个或批次训练数据经过模型前向计算输出结果
           - label：单个或批次训练数据对应的标签数据
           接口返回值是一个Tensor，根据自定义的逻辑加和或计算均值后的损失
       """
       # 使用Paddle中相关API自定义的计算逻辑
       # output = xxxxx
       # return output
       return self.loss_weight * (1-ms_ssim(input, label, data_range=self.data_range,
                      window = self.window, window_size=self.window_size, channel=self.channel,
                      size_average=True,  sigma=self.sigma,
                      weights=None, C=self.C))

class CharbonnierLoss(nn.Layer):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = paddle.mean(paddle.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class CompositeLoss(nn.Layer):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CompositeLoss, self).__init__()
        self.SSIMLoss = MS_SSIMLoss(loss_weight=1.0)
        self.L2Loss = CharbonnierLoss()

    def forward(self, x, y):
        loss = self.SSIMLoss(x, y) + self.L2Loss(x, y)
        return loss

