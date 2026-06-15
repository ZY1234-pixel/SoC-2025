"""
Official Code Implementation of:
"D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping"
"""

import argparse
import copy
import os
import random
import time

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import math

from loader.data_prefetcher import DataPrefetcher
from loader.dataset_doc3d_grid_HV import Warp_DataSet

from model_utils.lr_scheduler import WarmupCosineLR
from networks.d2dewarp_model import D2DewarpModel


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = float(args.lr) * epoch / args.warmup_epochs
    else:
        lr = float(args.min_lr) + (float(args.lr) - float(args.min_lr)) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', default=448, type=int, help='image size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='min learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save model interval between each train')
    parser.add_argument('--show_iter', type=int, default=50, help='Show log interval between each train')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    parser.add_argument('--exp_name', default='dewarp_HV_100K', help='Where to store logs and models')
    parser.add_argument('--data_path', default='/dataset/dewarp/dewarp_Horizontal_Vertical',
                        help='dataset path')

    parser.add_argument('--pre_trained_path', default=None,
                        help='pre trained model path')
    parser.add_argument('--parallel', default=True, type=str2bool, help='Set to True to train on parallel GPUs')

    parser.add_argument('--beta1', type=float, default=0.9, help='Beta Values for Adam Optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta Values for Adam Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for optim.')

    parser.add_argument('--log', default=True, type=str2bool, help='Set to False to stop logging')
    parser.add_argument('--save_path', default="./model_save/2025_dewarp_HV_100K", help='Save Model')
    parser.add_argument('--seed', type=int, default=24610, help='random seed')
    parser.add_argument('--warmup_epochs', type=int, default=80, help='epochs to warmup LR')
    parser.add_argument('--warmup_steps', type=int, default=10000, help='steps to warmup LR')

    """ Model """
    parser.add_argument('--hv_out_chans', type=int, default=1, help='h line and v line output channels')
    parser.add_argument('--d_model', type=int, default=448, help='last layer dim in UNet and all layers in attention')
    parser.add_argument('--in_chans', type=int, default=4, help='input channels')

    return parser.parse_args()


def clear():
    if os.name == 'nt':
        _ = os.system('cls')

    else:
        _ = os.system('clear')


def line_loss(gt, pred):
    n, _, h, w = gt.shape
    Non_index = torch.nonzero(gt)
    total_pix_count = n * h * w
    pos_pix_count = Non_index.shape[0]
    neg_pix_count = total_pix_count - pos_pix_count

    gt_pos_pix = gt[Non_index[:, 0], Non_index[:, 1], Non_index[:, 2], Non_index[:, 3]]
    pred_pos_pix = pred[Non_index[:, 0], Non_index[:, 1], Non_index[:, 2], Non_index[:, 3]]

    loss_pos = torch.sum((gt_pos_pix - pred_pos_pix) ** 2) / pos_pix_count
    loss_neg = (torch.sum((gt - pred) ** 2) - torch.sum((gt_pos_pix - pred_pos_pix) ** 2)) / neg_pix_count

    loss = (loss_pos * neg_pix_count + loss_neg * pos_pix_count) / total_pix_count
    return loss


def train(model, batch_size, epochs, start_epoch, optimizer, lr_scheduler, train_loader,
          log, save_path):
    if save_path[-1] != '/':
        save_path = save_path + '/'

    loss_train = []
    loss_val = []

    writer = SummaryWriter(os.path.join(save_path, 'log'))
    iter_num = 0
    val_iter_num = 0

    for epoch in range(start_epoch, epochs + 1):
        print("Epoch ", epoch)
        model.train()

        with open(save_path + 'train_log.txt', 'a') as f:
            f.write('-------training-------\n')
        f.close()
        print("Learning Rate", optimizer.param_groups[0]['lr'])

        train_prefetcher = DataPrefetcher(train_loader)
        train_iter = 0
        train_samples = len(train_loader.dataset)
        loss_sum_train = 0
        train_batches = int(len(train_loader.dataset) / batch_size)
        with tqdm(total=train_batches) as pbar:

            # for data_iter_step, (img, lbl, fore_mask, text_line, uv, wc, imgs_name) in enumerate(train_loader):
            batch = train_prefetcher.next()
            while batch is not None:
                train_iter += 1
                if train_iter >= train_batches:
                    break

                img, lbl, h_line, v_line, edge = batch[0], batch[1], batch[2], batch[3], batch[4]

                pred_h_lst, pred_v_lst, pred_bm = model(torch.cat((img.float(), edge.float()), dim=1))

                loss_h = 0.0
                for i in range(len(pred_h_lst)):
                    loss_h_line = line_loss(h_line.float(), pred_h_lst[i].float())
                    loss_h_bce = 1.0 / (len(pred_h_lst) * 2 - i) * nn.BCELoss() \
                        (torch.sigmoid(pred_h_lst[i].float()), h_line.float())
                    loss_h = loss_h + (loss_h_line + loss_h_bce)

                loss_v = 0.0
                for i in range(len(pred_v_lst)):
                    loss_v_line = line_loss(v_line.float(), pred_v_lst[i].float())
                    loss_v_bce = 1.0 / (len(pred_v_lst) * 2 - i) * nn.BCELoss() \
                        (torch.sigmoid(pred_v_lst[i].float()), v_line.float())
                    loss_v = loss_v + (loss_v_line + loss_v_bce)

                loss_bm = nn.L1Loss()(lbl.float(), pred_bm.float())
                loss = 5. * loss_bm + (loss_h + loss_v)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum_train += float(loss)
                iter_num = iter_num + 1
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("info/lr", lr, iter_num)
                writer.add_scalar("info/train_loss", loss, iter_num)
                writer.add_scalar("info/train_loss_bm", loss_bm, iter_num)
                writer.add_scalar("info/train_loss_h", loss_h, iter_num)
                writer.add_scalar("info/train_loss_v", loss_v, iter_num)

                if iter_num % parser.save_interval == 0:
                    torch.save({'epoch': epoch, 'step': iter_num, 'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': lr_scheduler.state_dict()},
                               save_path + 'epoch_' + str(epoch) + '_' + str(train_iter) + '_iter.pt')

                with open(save_path + 'train_log.txt', 'a') as f:
                    if iter_num % parser.show_iter == 0:
                        str_log = '[{}] - Epoch [{}/{}] | iter [{}/{}] | lr: {:.8f} | loss: {:.8f} | ' \
                                  'loss_bm: {:.8f} | loss_h: {:.8f} | loss_v: {:.8f}'.format(
                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            str(epoch), str(epochs),
                            str(train_iter),
                            str(train_batches),
                            lr, loss.item() / batch_size,
                                loss_bm.item() / batch_size,
                                loss_h.item() / batch_size,
                                loss_v.item() / batch_size)
                        print(str_log)
                        f.write(str_log + '\n')
                pbar.update(1)
                batch = train_prefetcher.next()

                lr_scheduler.step()

        torch.save(
            {'epoch': epoch, 'step': iter_num, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': lr_scheduler.state_dict()}, save_path + str(epoch) + '.pt')

        loss_train.append(loss_sum_train / train_samples)
        print('[{}] - Epoch {}, Training Loss: {:.12f}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                                               str(epoch), (loss_sum_train / train_samples)))
        with open(save_path + 'train_log.txt', 'a') as f:
            f.write('-------Evaluation-------\n')
        f.close()
        writer.add_scalar("info/loss_train", (loss_sum_train / train_samples), epoch)

        if log:
            with open(save_path + 'loss.txt', 'a') as f:
                str1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " - Epoch  " + str(
                    epoch) + "  Training Loss" + str((loss_sum_train / train_samples)) + '\n'
                f.write(str1)


if __name__ == '__main__':
    parser = get_args()
    seed_torch(seed=parser.seed)
    parser.exp_name += f'_input{parser.input_size}-Seed{parser.seed}'
    save_path = f'{parser.save_path}/{parser.exp_name}'
    os.makedirs(save_path, exist_ok=True)
    print("----------[seed]----------: ", parser.seed)

    clear()
    data_path = parser.data_path
    assert data_path and os.path.isdir(data_path), 'Wrong Data path'

    model = D2DewarpModel(img_size=parser.input_size, in_chans=parser.in_chans, hv_out_chans=parser.hv_out_chans,
                          d_model=parser.d_model).cuda()

    optimizer = optim.AdamW(model.parameters(), lr=float(parser.lr),
                            betas=(float(parser.beta1), float(parser.beta2)),
                            weight_decay=parser.weight_decay)

    print("----------[model]----------")
    print(model)

    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('Model Summary: %g parameters, %g gradients\n' % (n_p, n_g))

    print("----------[config]----------")
    print(parser)

    print("----------[optimizer]----------")
    print(optimizer)

    dataset_train = Warp_DataSet(parser.input_size, data_path, split='DocDewarpHV', is_aug=True)
    train_loader = DataLoader(dataset_train, batch_size=parser.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    max_iterations = parser.epochs * len(train_loader)
    lr_scheduler = WarmupCosineLR(optimizer, parser.min_lr, parser.lr, parser.warmup_steps, max_iterations, 0.1)

    start_epoch = 0

    if parser.pre_trained_path:
        pre = 1
        assert os.path.exists(parser.pre_trained_path), 'Wrong path for pre-trained model'
        weights = torch.load(parser.pre_trained_path)
        model.load_state_dict(weights['state_dict'])  # , strict=True
        optimizer.load_state_dict(weights['optimizer'])
        lr_scheduler.load_state_dict(weights['scheduler'])

        start_epoch = weights['epoch'] + 1
        lr_scheduler.last_epoch = start_epoch
        print(
            f'model {parser.pre_trained_path} loaded!, Current epoch is {start_epoch}, lr_scheduler: {lr_scheduler.state_dict()}')

    train(model, int(parser.batch_size), int(parser.epochs), start_epoch, optimizer,
          lr_scheduler, train_loader, parser.log, save_path)
