import argparse
import random
import math
import logging
import os.path as osp
import time
import datetime
import os

import numpy as np
import paddle
from paddle.io import DataLoader

from utils.options import dict2str, parse
from utils.logger import get_root_logger, MessageLogger
from utils.misc import get_time_str, make_exp_dirs, mkdir_and_rename
from data.dataset import Dataset_GaussianDenoising, Dataset_PairedImage
from models.image_restoration_model import ImageCleanModel


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML code.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings

    opt['dist'] = False
    print('Disable distributed.', flush=True)

    opt['rank'] = 0
    opt['world_size'] = 1
    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed

    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    return opt, args


def init_loggers(opt):
    os.makedirs(opt['path']['log'], exist_ok=True)
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)

    logger.info(dict2str(opt))

    return logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    local_rank = paddle.distributed.ParallelEnv().local_rank
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = Dataset_PairedImage(dataset_opt)
            batch_sampler = paddle.io.DistributedBatchSampler(
                train_set, batch_size=dataset_opt['batch_size_per_gpu'], shuffle=True, drop_last=True)
            train_loader = DataLoader(dataset=train_set,
                                      batch_sampler=batch_sampler,
                                      num_workers=4)

            num_iter_per_epoch = math.ceil(len(train_loader) * dataset_enlarge_ratio)
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            if local_rank == 0:
                logger.info(
                    'Training statistics:'
                    f'\n\tNumber of train images: {len(train_set)}'
                    f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                    f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                    f'\n\tWorld size (gpu number): {opt["world_size"]}'
                    f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                    f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = Dataset_PairedImage(dataset_opt)
            batch_sampler = paddle.io.DistributedBatchSampler(
                val_set, batch_size=1, shuffle=False, drop_last=False)
            val_loader = DataLoader(dataset=val_set,
                                    batch_sampler=batch_sampler,
                                    num_workers=0)
            if local_rank == 0:
                logger.info(
                    f'Number of val images/folders in {dataset_opt["name"]}: '
                    f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, val_loader, total_epochs, total_iters, num_iter_per_epoch


def main():
    opt, args = parse_options(is_train=True)

    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()
        opt['world_size'] = nranks
    # mkdir for experiments and logger

    # if local_rank == 0:
    #     make_exp_dirs(opt)
    #     if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
    #             'name'] and opt['rank'] == 0:
    #         mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger = init_loggers(opt)

    model = ImageCleanModel(opt)
    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, val_loader, total_epochs, total_iters, num_iter_per_epoch = result


    start_epoch = 0
    current_iter = 0

    if args.resume is not None:
        state_dict = paddle.load(args.resume+".pdparams")
        logger.info(
            f'load resume state_dict: {args.resume}')
        model.net_g.set_state_dict(state_dict)
        # state_dict = paddle.load(args.resume + ".pdopt")
        # model.optimizer_g.set_state_dict(state_dict)
        # start_epoch = args.resume.split('/')[-1].split('_')[0]
        start_epoch = 0
        start_epoch = int(start_epoch)
        current_iter = start_epoch * num_iter_per_epoch

    msg_logger = MessageLogger(opt, current_iter)
    # training
    if local_rank == 0:
        logger.info(
            f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()
    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)

    scale = opt['scale']

    epoch = start_epoch
    best_metric = 0
    while current_iter <= total_iters:
        for idx, train_data in enumerate(train_loader):
            current_iter += 1
            if current_iter > total_iters:
                break
            ### ------Progressive learning ---------------------
            j = ((current_iter > groups) != True).nonzero()[0]
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]

            if logger_j[bs_j] and local_rank == 0:
                logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(mini_gt_size,
                                                                                          mini_batch_size))
                logger_j[bs_j] = False

            lq = train_data['lq']
            gt = train_data['gt']

            if mini_batch_size < batch_size:
                indices = random.sample(range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]

            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:, :, x0:x1, y0:y1]
                gt = gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]
            ###-------------------------------------------

            model.feed_train_data({'lq': lq, 'gt': gt})
            model.optimize_parameters(current_iter)

            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0 and local_rank == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
            data_time = time.time()
            iter_time = time.time()
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0 and local_rank == 0:
                # save models and training states
                logger.info(f'Saving models and training states on epoch {epoch}.')
                model.save()
                logger.info(f'Saving models and training states done.')
                # validation
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                current_metric = model.validation(val_loader, current_iter,
                                opt['val']['save_img'], rgb2bgr, use_image)
                if current_metric > best_metric:
                    best_metric = current_metric
                    logger.info(f'Saving best models and training states on epoch {epoch}.')
                    model.save(prefix_name='best')
        epoch += 1

    # end of epoch
    if local_rank == 0:
        consumed_time = str(
            datetime.timedelta(seconds=int(time.time() - start_time)))
        logger.info(f'End of training. Time consumed: {consumed_time}')
        logger.info('Save the latest model.')
        model.save()  # -1 stands for the latest
    if opt.get('val') is not None and local_rank == 0:
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        use_image = opt['val'].get('use_image', True)
        model.validation(val_loader, current_iter,
                         opt['val']['save_img'],
                         rgb2bgr=rgb2bgr,
                         use_image=use_image)


if __name__ == '__main__':
    main()
