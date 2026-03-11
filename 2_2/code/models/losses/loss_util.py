import functools


def reduce_loss(loss, reduction):
    """减少损失
    参数：
        loss (tensor)：元素级损失张量。
        reduction (str)：可选值为 ‘none’、'mean' 和 ‘sum’。
    返回值：
        tensor：经过减少的损失张量。
    """
    # none: 0, elementwise_mean:1, sum: 2
    if reduction == None:
        return loss
    elif reduction == 'mean':
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """应元素级权重并减少损失。
    参数：
        loss (tensor)：元素级损失值。
        weight (tensor)：元素级权重。默认值：无。
        reduction (str)：与PyTorch内置损失函数相同。可选值为
            ‘none’、'mean' 和 ‘sum’。默认值：‘mean’。

    返回值：
        tensor：损失值。
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss


def weighted_loss(loss_func):
    """创建给定损失函数的加权版本。
    损失函数必须具有如下：
    `loss_func(pred, target, **kwargs)`。该函数只需计算元素级损失，无需任何归约操作。
    将为函数添加权重和归约参数。处理后的函数将变为：
    `loss_func(pred, target, weight=None, reduction=‘mean’, **kwargs)`。

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper
