import logging
import time
import datetime

initialized_logger = {}

def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    """获取根日志。
    若日志器尚未初始化，则将进行初始化。默认情况下会添加一个
    流处理器。若指定了`log_file`，则还会添加一个文件处理器。
    参数：
        logger_name (str)：根日志器名称。默认值：‘basicsr’。
        log_file (str | None)： 日志文件名。若指定，将向根日志器添加FileHandler。
        log_level (int)：根日志器级别。注意仅影响
            进程序号为0的进程，其他进程将设置级别为
            “Error”并保持静默状态。
    返回值：
        logging.Logger：根日志器。
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    if log_file is not None:
        logger.setLevel(log_level)
        # add code handler
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger


class MessageLogger():
    """用于打印的消息记录器。
    参数：
        opt (dict)：配置。包含以下：
            name (str)：实验名称。
            logger (dict)：包含记录器间隔参数 ‘print_freq’ (str)。
            train (dict)：包含总迭代次数参数 ‘total_iter’ (int)。
            use_tb_logger (bool): 使用TensorBoard日志器。
        start_iter (int): 起始迭代次数。默认值：1。
        tb_logger (obj:`tb_logger`): TensorBoard日志器实例。默认值：None。
    """

    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    def __call__(self, log_vars):
        """格式化日志消息。
        参数：
            log_vars (dict)：包含以下：
                epoch (int)： epoch 编号。
                iter (int)：当前迭代次数。
                lrs (list)：学习率列表。

                time (float)：迭代时间。
                data_time (float)：每次迭代的数据时间。
        """
        # epoch, iter, learning rates
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')

        message = (f'epoch:{epoch}, ' f'iter:{current_iter}, lr: ')
        for v in lrs:
            message += f'{v:.6f}'
        message += ' '

        # other items, especially losses
        # for k, v in log_vars.items():
        loss = log_vars['l_pix']
        message += f'loss: {loss:.6f} '
        # time and estimated time
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')

            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f' eta: {eta_str}, '
            message += f'time (data): {iter_time:.3f}'

        self.logger.info(message)

