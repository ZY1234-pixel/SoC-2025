import math
from paddle.optimizer.lr import LRScheduler


def get_position_from_periods(iteration, cumulative_period):
    """从周期列表中获取位置。
    该函数将返回周期列表中右侧最近数字的索引。
    例如，当 cumulative_period = [100, 200, 300, 400] 时：
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.
    参数：
        iteration (int)：当前迭代次数。
        cumulative_period (list[int])：累积周期列表。

    返回值：
        int：周期列表中右侧最近数字的位置。
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i

class CosineAnnealingRestartCyclicLR(LRScheduler):
    """ 余弦退火学习
    配置示例：
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7
    该方案包含四个周期，每个周期包含10次迭代。在第10、20、30次迭代时，调度器将使用restart_weights中的权重值重启。
    参数：
        optimizer (torch.nn.optimizer)：Torch优化器。
        periods (list)：每个余弦退火周期的迭代次数。
        restart_weights (list): 每次重启迭代时使用的重启权重。
            默认值: [1]。
        eta_min (float): 最小学习率。默认值: 0。
        last_epoch (int): 用于_LRScheduler。默认值: -1。
    """

    def __init__(self,
                 learning_rate,
                 periods,
                 restart_weights=(1, ),
                 eta_mins=(0, ),
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartCyclicLR, self).__init__(learning_rate, last_epoch)
        
    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return eta_min + current_weight * 0.5 * (self.base_lr - eta_min) * \
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))

