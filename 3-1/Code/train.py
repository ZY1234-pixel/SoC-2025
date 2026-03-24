"""Training entry for ESDNet-Lite."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from tqdm import tqdm

from dataset import UHDMTrainDataset, UHDMValDataset
from model import build_model
from utils import ensure_dir, load_checkpoint_state_dict, set_random_seed


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(weights='IMAGENET1K_V1').features
        blocks = [vgg[:4].eval(), vgg[4:9].eval(), vgg[9:16].eval()]
        for block in blocks:
            for parameter in block.parameters():
                parameter.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, prediction, target, layers=(0, 1, 2)):
        prediction = (prediction - self.mean) / self.std
        target = (target - self.mean) / self.std
        prediction = F.interpolate(prediction, size=(224, 224), mode='bilinear', align_corners=False)
        target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        loss = 0.0
        pred_feature = prediction
        target_feature = target
        for index, block in enumerate(self.blocks):
            pred_feature = block(pred_feature)
            target_feature = block(target_feature)
            if index in layers:
                loss += F.l1_loss(pred_feature, target_feature)
        return loss


class MultiScaleLoss(nn.Module):
    def __init__(self, lam: float = 1.0, lam_p: float = 1.0):
        super().__init__()
        self.lam = lam
        self.lam_p = lam_p
        self.perceptual = VGGPerceptualLoss()

    def forward(self, out1, out2, out3, target):
        target_2 = F.interpolate(target, scale_factor=0.5, mode='bilinear', align_corners=False)
        target_3 = F.interpolate(target, scale_factor=0.25, mode='bilinear', align_corners=False)
        loss = 0.0
        for prediction, label in ((out1, target), (out2, target_2), (out3, target_3)):
            loss += self.lam * F.l1_loss(prediction, label)
            loss += self.lam_p * self.perceptual(prediction, label)
        return loss


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    psnr_sum = 0.0
    count = 0
    for batch in val_loader:
        image = batch['in_img'].to(device)
        target = batch['label'].to(device)
        _, _, height, width = image.shape
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='replicate')
        output = model(image)[0][:, :, :height, :width].clamp(0, 1)
        mse = F.mse_loss(output, target, reduction='mean').item()
        if mse > 0:
            psnr_sum += 10 * math.log10(1.0 / mse)
        count += 1
    model.train()
    return psnr_sum / max(count, 1)


def load_teacher(teacher_path: str | None, device):
    if not teacher_path:
        return None
    teacher = build_model('full')
    teacher.load_state_dict(load_checkpoint_state_dict(teacher_path), strict=True)
    teacher.eval().to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def build_arg_parser():
    parser = argparse.ArgumentParser(description='训练 ESDNet-Lite（直接参数模式）')
    parser.add_argument('--train_dir', required=True, help='UHDM 训练集路径')
    parser.add_argument('--val_dir', required=True, help='UHDM 验证集路径')
    parser.add_argument('--save_dir', default='test-result/train_runs', help='训练输出目录')
    parser.add_argument('--exp_name', default='uhdm_lite_s', help='实验名称')
    parser.add_argument('--model_preset', choices=['lite-s', 'lite-xs'], default='lite-s')
    parser.add_argument('--teacher_path', default=None, help='Teacher checkpoint 路径')
    parser.add_argument('--resume', default=None, help='恢复训练的 checkpoint 路径')
    parser.add_argument('--gpu_id', type=int, default=0, help='单卡训练时使用的 GPU ID')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--val_num', type=int, default=50)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--lam_p', type=float, default=1.0)
    parser.add_argument('--lam_kd', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--eta_min', type=float, default=1e-6)
    parser.add_argument('--t0', type=int, default=50)
    parser.add_argument('--t_mult', type=int, default=1)
    return parser


def main():
    args = build_arg_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_random_seed(args.seed)

    save_dir = ensure_dir(Path(__file__).resolve().parent.parent / args.save_dir / args.exp_name)
    ckpt_dir = ensure_dir(save_dir / 'checkpoints')

    student = build_model(args.model_preset).to(device)
    print(f'[i] Student 参数量: {sum(p.numel() for p in student.parameters()) / 1e6:.4f} M')

    teacher = load_teacher(args.teacher_path, device)
    lam_kd = args.lam_kd if teacher is not None else 0.0
    if teacher is not None:
        print(f'[i] Teacher 已加载: {args.teacher_path}')

    train_loader = data.DataLoader(
        UHDMTrainDataset(args.train_dir, crop_size=args.crop_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = data.DataLoader(
        UHDMValDataset(args.val_dir, max_num=args.val_num),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    loss_fn = MultiScaleLoss(lam=args.lam, lam_p=args.lam_p).to(device)
    optimizer = optim.Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.t0, T_mult=args.t_mult, eta_min=args.eta_min
    )

    start_epoch = 1
    steps = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        student.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        steps = checkpoint.get('iters', 0)
        print(f'[i] 恢复训练: epoch={start_epoch}, iters={steps}')

    best_psnr = 0.0
    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        epoch_loss = 0.0
        for batch_index, batch in enumerate(progress):
            image = batch['in_img'].to(device)
            target = batch['label'].to(device)
            out1, out2, out3 = student(image)
            loss = loss_fn(out1, out2, out3, target)
            if teacher is not None and lam_kd > 0:
                with torch.no_grad():
                    teacher_out = teacher(image)[0]
                loss += lam_kd * F.l1_loss(out1, teacher_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps += 1
            epoch_loss += loss.item()
            progress.set_postfix(loss=f'{epoch_loss / (batch_index + 1):.5f}', lr=f"{optimizer.param_groups[0]['lr']:.7f}")

        scheduler.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            psnr = validate(student, val_loader, device)
            print(f'[Val] Epoch {epoch}: PSNR={psnr:.3f} dB')
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save({'state_dict': student.state_dict()}, ckpt_dir / 'best.pth')
                print(f'[i] best.pth 已更新: {best_psnr:.3f} dB')

        latest = {
            'epoch': epoch,
            'iters': steps,
            'state_dict': student.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(latest, ckpt_dir / 'latest.pth')
        if epoch % 10 == 0:
            torch.save(latest, ckpt_dir / f'epoch_{epoch:04d}.pth')

    print(f'[✓] 训练完成，最佳 PSNR: {best_psnr:.3f} dB')
    print(f'[i] checkpoint 目录: {ckpt_dir}')


if __name__ == '__main__':
    main()
