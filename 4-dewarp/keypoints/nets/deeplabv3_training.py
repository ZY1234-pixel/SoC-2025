import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


def compute_distance_map(target, gamma=3, theta=10):
    """
        计算边界权重图：离边界越近，权重越高
        target: [B, H, W]
        """
    weights = []
    target_np = target.cpu().numpy().astype(np.uint8)

    for i in range(target_np.shape[0]):
        m = target_np[i]

        # 前景内部距离边界
        dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        dist_in = np.clip(dist_in, 0, 300)
        w_in = np.exp(-dist_in / theta) * gamma

        # 背景外部距离边界
        dist_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
        dist_out = np.clip(dist_out, 0, 10)
        w_out = np.exp(-dist_out / theta) * gamma * 0.3

        # 合并
        w = 1.0 + w_in + w_out
        weights.append(w)

    return torch.from_numpy(np.stack(weights)).float().to(target.device)

# def compute_distance_map(target, gamma=3, theta=10):
#     """
#     GPU 版本，带 clip 限制
#     target: [B, H, W] 0/1
#     返回: [B, H, W] float
#     前景内部距离 clip 0-300
#     背景外部距离 clip 0-10
#     """
#     device = target.device
#     target = target.float().unsqueeze(1)  # [B,1,H,W]
#
#     # 1. 使用 max pooling 近似距离 transform
#     # 背景->前景距离
#     inv_target = 1 - target
#     dist_out = inv_target.clone()
#     for i in range(10):  # clip 0-10
#         dist_out = F.max_pool2d(dist_out, kernel_size=3, stride=1, padding=1)
#     dist_out = dist_out.squeeze(1).clamp(0, 10)
#
#     # 前景->背景距离
#     dist_in = target.clone()
#     for i in range(300):  # clip 0-300
#         dist_in = F.max_pool2d(dist_in, kernel_size=3, stride=1, padding=1)
#     dist_in = dist_in.squeeze(1).clamp(0, 300)
#
#     # 权重
#     w_in = gamma * torch.exp(-dist_in / theta)
#     w_out = gamma * 0.3 * torch.exp(-dist_out / theta)
#     weights = 1.0 + w_in + w_out
#
#     return weights


def Boundary_Loss(pred, target):
    # pred: [B, C, H, W], target: [B, H, W]
    # 1. 计算基础的交叉熵损失 (不进行 reduction)
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    # 2. 计算边界权重
    # 离边界越近的点，权重越大
    weights = compute_distance_map(target)
    # probs = F.softmax(pred, dim=1)[:,1,:,:]
    # background_mask = (target==0).float()
    # penalty = (probs * weights * background_mask).mean()
    # 3. 加权平均
    loss = (ce_loss * weights).sum()/weights.sum()
    return loss


def Tversky_loss(inputs, target):
    alpha = 0.7
    beta = 0.3
    smooth = 1e-6
    # inputs: [B, C, H, W], target: [B, H, W]
    n, c, h, w = inputs.size()
    if target.dim() == 4:
        target = torch.argmax(target, dim=-1)
    nt, ht, wt = target.size()

    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    inputs = F.softmax(inputs, dim=1)

    # 重点关注书本类 (假设 index=1)
    p1 = inputs[:, 1, :, :].reshape(n, -1)
    g1 = (target == 1).float().reshape(n, -1)

    tp = torch.sum(p1 * g1, dim=1)
    fn = torch.sum(g1 * (1 - p1), dim=1)
    fp = torch.sum((1 - g1) * p1, dim=1)

    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky)


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
