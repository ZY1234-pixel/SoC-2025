import importlib
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import paddle

from models.archs.restormer_arch import Restormer
from models.archs.nafnet_arch import NAFNet
from models.base_model import BaseModel
from utils.logger import get_root_logger
from utils.img_utils import imwrite, tensor2img

loss_module = importlib.import_module('models.losses')
metric_module = importlib.import_module('metrics')

import os
import random
import numpy as np
import cv2
import paddle.nn.functional as F
from functools import partial

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity):

        self.use_identity = use_identity
        self.mixup_beta = mixup_beta
        self.augments = [self.mixup]

    def mixup(self, target, input_):
        # lam = self.mixup_beta.sample([1, 1])
        lam = np.random.beta(self.mixup_beta, self.mixup_beta, [1, 1])
        lam = paddle.to_tensor(lam.astype("float32"))
    
        r_index = paddle.randperm(target.shape[0])

        target = lam * target + (1-lam) * target[r_index]
        input_ = lam * input_ + (1-lam) * input_[r_index]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity)

        # self.net_g = Restormer(inp_channels=3,
        #                        out_channels=3,
        #                        dim=48,
        #                        num_blocks=[4,6,6,8],
        #                        num_refinement_blocks=4,
        #                        heads=[1,2,4,8],
        #                        ffn_expansion_factor=2.66,
        #                        bias=False,
        #                        LayerNorm_type='BiasFree',
        #                        dual_pixel_task=False)
        self.net_g = NAFNet(img_channel=3, width=24, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 10], dec_blk_nums=[1, 1, 1, 1])
        if self.is_train:
            self.init_training_settings()

        nranks = paddle.distributed.ParallelEnv().nranks
        local_rank = paddle.distributed.ParallelEnv().local_rank
        if nranks > 1:
            # Initialize parallel environment if not done.
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
            ):
                paddle.distributed.init_parallel_env()
                self.net_g = paddle.DataParallel(self.net_g)
            else:
                self.net_g = paddle.DataParallel(self.net_g)

        if local_rank == 0:
            self.print_network(self.net_g)


    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt'])
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_schedulers()
        self.setup_optimizers()


    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if not v.stop_gradient:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        grad_clip = None
        if self.opt['train']['use_grad_clip']:
            grad_clip = paddle.nn.ClipGradByNorm(0.01)
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            lr = self.schedulers[0]
            self.optimizer_g = paddle.optimizer.Adam(learning_rate=lr, parameters=optim_params, grad_clip=grad_clip, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            lr = self.schedulers[0]
            self.optimizer_g = paddle.optimizer.AdamW(learning_rate=lr, parameters=optim_params, grad_clip=grad_clip,  **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq']
        if 'gt' in data:
            self.gt = data['gt']

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq']
        if 'gt' in data:
            self.gt = data['gt']

    def optimize_parameters(self, current_iter):

        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)

        loss_dict['l_pix'] = l_pix

        l_pix.backward()

        self.optimizer_g.step()
        self.optimizer_g.clear_gradients()
        self.schedulers[0].step()

        self.log_dict = self.reduce_loss_dict(loss_dict)


    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.shape
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.shape
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with paddle.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with paddle.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def nondist_validation(self, dataloader, current_iter,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger=None):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, prefix_name="last"):
        current_path = "code/output/model"
        os.makedirs(current_path, exist_ok=True)
        paddle.save(self.net_g.state_dict(), os.path.join(current_path, f"{prefix_name}_model.pdparams"))
        paddle.save(self.optimizers[0].state_dict(), os.path.join(current_path, f"{prefix_name}_model.pdopt"))
