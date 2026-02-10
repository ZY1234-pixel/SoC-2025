import numpy as np
import os
import random
import time
import paddle
from os import path as osp

from .logger import get_root_logger


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdir_and_rename(path):
    """mkdirs。若路径已存在，则将其重命名为包含时间戳的新名称并创建新文件夹。
    参数：
        path (str)：文件夹路径。
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)

def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        mkdir_and_rename(path_opt.pop('experiments_root'))
    else:
        mkdir_and_rename(path_opt.pop('results_root'))
    for key, path in path_opt.items():
        if ('strict_load' not in key) and ('pretrain_network'
                                           not in key) and ('resume'
                                                            not in key):
            os.makedirs(path, exist_ok=True)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """扫描目录以查找目标文件。
    参数：
        dir_path (str)：目录路径。
        suffix (str | tuple(str), 可选)：目标文件后缀。
            默认值：无。
        recursive (bool，可选)：若设为True，则递归扫描目录。
            默认值：False。
        full_path (bool, 可选): 设为 True 时包含目录路径。
            默认值: False。
    返回值:
        包含所有目标文件相对路径的生成器。
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def scandir_SIDD(dir_path, keywords=None, recursive=False, full_path=False):
    """扫描目录以查找感兴趣的文件。
    参数：
        dir_path (str)：目录路径。
        keywords (str | tuple(str)，可选)：感兴趣的文件关键词。
            默认值：无。
        recursive (bool，可选)：若设为True，则递归扫描目录。
            默认值：False。
        full_path (bool, 可选): 设为 True 时包含目录路径。
            默认值: False。
    返回值:
        包含所有目标文件相对路径的生成器。
    """

    if (keywords is not None) and not isinstance(keywords, (str, tuple)):
        raise TypeError('"keywords" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, keywords, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if keywords is None:
                    yield return_path
                elif return_path.find(keywords) > 0:
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, keywords=keywords, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, keywords=keywords, recursive=recursive)

def check_resume(opt, resume_iter):
    """检查恢复状态和预训练网络路径。
    参数：
        opt (dict)：选项。
        resume_iter (int)：恢复迭代次数。
    """
    logger = get_root_logger()
    if opt['path']['resume_state']:
        # get all the networks
        networks = [key for key in opt.keys() if key.startswith('network_')]
        flag_pretrain = False
        for network in networks:
            if opt['path'].get(f'pretrain_{network}') is not None:
                flag_pretrain = True
        if flag_pretrain:
            logger.warning(
                'pretrain_network path will be ignored during resuming.')
        # set pretrained model paths
        for network in networks:
            name = f'pretrain_{network}'
            basename = network.replace('network_', '')
            if opt['path'].get('ignore_resume_networks') is None or (
                    basename not in opt['path']['ignore_resume_networks']):
                opt['path'][name] = osp.join(
                    opt['path']['models'], f'net_{basename}_{resume_iter}.pth')
                logger.info(f"Set {name} to {opt['path'][name]}")


def sizeof_fmt(size, suffix='B'):
    """获取可读的文件大小。
    参数：
        size (int)：文件大小。
        suffix (str)：后缀。默认值：‘B’。
    返回值：
        str：格式化的文件大小。
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} Y{suffix}'
