import argparse
import gc
import os
import sys

import torch
import torch.nn.functional as F

import data_doc3D
import data_UVDoc
import model
import utils
from data_mixDataset import mixDataset
from ddcp.Source.loss import Losses

train_mse = 0.0
losscount = 0
# 各损失分量（带权重）的逐 epoch 累加器，用于日志输出
loss2D_acc = 0.0    # Σ α·L_grid2D
loss3D_acc = 0.0    # Σ β·L_grid3D
lossrecon_acc = 0.0 # Σ γ·L_recon
losscross_acc = 0.0 # Σ cross_w·L_cross
losstotal_acc = 0.0 # Σ 总损失 netLoss
gamma_w = 0.0

def _grid_to_pixel(grid, img_size):
    """Map a normalized [-1, 1] grid2D to pixel coordinates.

    UVDoc normalizes grid2D to [-1, 1] (see ``data_utils.crop_tight``), while
    ddcp's differential/interval losses were tuned for pixel-scale control points
    (~1024x960). The quadratic terms (local / edge / rectangle) are therefore
    ~1e5x too weak unless evaluated on pixel-scale coordinates. This is a plain
    linear remap: channel 0 (x) -> [0, W], channel 1 (y) -> [0, H].

    grid: (B, 2, H, W) in [-1, 1].
    """
    W, H = img_size
    out = grid.clone()
    out[:, 0] = (out[:, 0] + 1.0) * (W / 2.0)
    out[:, 1] = (out[:, 1] + 1.0) * (H / 2.0)
    return out


def single_scale_cross_loss(pred, target, loss_fn, alpha=0.1, img_size=utils.IMG_SIZE):
    """Single-scale differential-coordinate (cross) loss on grid2D.

    Fix for the [-1, 1] normalization used by UVDoc (problem #1): ddcp's weights
    (alpha=0.1, 0.001/0.01 for local/edge/rectangle) were tuned for pixel-scale
    coordinates. We evaluate the quadratic differential / interval terms on
    pixel-scale coordinates (via ``_grid_to_pixel``) so they regain their intended
    strength relative to the position L1, but we keep the position-L1 term at the
    normalized scale (``l1_norm``) so it does not re-dominate the existing
    ``alpha_w * L1`` supervision in the total loss.

    The combination mirrors ddcp/Source/train.py:
        l1 + alpha*local + 0.001*edge + 0.01*rectangle
    where ``alpha`` == ddcp's lambda_loss_a (=0.1) and 0.001/0.01 == lambda_loss_b/c.
    """
    l1_norm = F.smooth_l1_loss(pred, target, reduction="mean")
    pred_p = _grid_to_pixel(pred, img_size)
    target_p = _grid_to_pixel(target, img_size)
    _, local, edge, rectangle = loss_fn(pred_p, target_p, size_average=True)
    return l1_norm + alpha * local + 0.001 * edge + 0.01 * rectangle


def multi_scale_cross_loss(pred_list, target_native, loss_fn, alpha=0.1, img_size=utils.IMG_SIZE):
    """Multi-scale cross (differential-coordinate) loss.

    This is a TRUE multi-scale loss: the cross-consistency term is enforced
    separately at several real down-sampling scales of the network. ``pred_list``
    is a list of 2D-grid predictions ``(B, 2, h_i, w_i)`` coming from different
    decoder / down-sampling stages (e.g. 1/4, 1/8 and the finest 1/16 of the
    input resolution), NOT a bilinearly-downsampled copy of the final output.

    ``target_native`` is the ground-truth grid2D at the finest native resolution.
    For each prediction scale it is bilinearly resampled to that prediction's
    spatial size, so the differential-coordinate / cross consistency (ddcp's
    ``loss_fn4_v5_r_4`` -> ``(l1 + alpha*local)``) is applied at every scale.

    The grid is remapped to pixel scale before the loss (see ``_grid_to_pixel``)
    so the quadratic terms keep the intended strength under UVDoc's [-1, 1] norm.

    pred_list: list of (B, 2, h_i, w_i) tensors at distinct resolutions.
    target_native: (B, 2, H, W) ground-truth at the finest resolution.
    """
    assert pred_list, "pred_list must contain at least one prediction."
    total = torch.zeros((), device=pred_list[0].device)
    H, W = target_native.shape[-2:]
    for p in pred_list:
        _, _, h, w = p.shape
        if h == H and w == W:
            t = target_native
        else:
            t = F.interpolate(target_native, size=(h, w), mode="bilinear", align_corners=True)
        p_p = _grid_to_pixel(p, img_size)
        t_p = _grid_to_pixel(t, img_size)
        l1, local, _, _ = loss_fn(p_p, t_p, size_average=True)
        total = total + (l1 + alpha * local)
    return total


def setup_data(args):
    """
    Returns train and validation dataloader.
    """
    doc3D = data_doc3D.doc3DDataset
    UVDoc = data_UVDoc.UVDocDataset
    traindata = "train"
    valdata = "val"

    # Training data (only instantiate the datasets actually needed)
    t_doc3D_data = None
    t_UVDoc_data = None
    if args.data_to_use in ("both", "doc3d"):
        t_doc3D_data = doc3D(
            data_path=args.data_path_doc3D,
            split=traindata,
            appearance_augmentation=args.appearance_augmentation,
        )
    if args.data_to_use in ("both", "uvdoc"):
        t_UVDoc_data = UVDoc(
            data_path=args.data_path_UVDoc,
            appearance_augmentation=args.appearance_augmentation,
            geometric_augmentations=args.geometric_augmentationsUVDoc,
        )

    if args.data_to_use == "both":
        t_mix_data = mixDataset(t_doc3D_data, t_UVDoc_data)
        trainloader = torch.utils.data.DataLoader(
            t_mix_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True
        )
    elif args.data_to_use == "doc3d":
        trainloader = torch.utils.data.DataLoader(
            t_doc3D_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True
        )
    elif args.data_to_use == "uvdoc":
        trainloader = torch.utils.data.DataLoader(
            t_UVDoc_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True
        )
    else:
        raise ValueError(f"data_to_use should be either both, doc3d or uvdoc, provided {args.data_to_use}.")

    # Validation data (doc3D by default, UVDoc when training uvdoc only)
    if args.data_to_use == "uvdoc":
        v_data = UVDoc(
            data_path=args.data_path_UVDoc,
            appearance_augmentation=[],
            geometric_augmentations=[],
        )
    else:
        v_data = doc3D(data_path=args.data_path_doc3D, split=valdata, appearance_augmentation=[])
    valloader = torch.utils.data.DataLoader(
        v_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True
    )

    return trainloader, valloader


def get_scheduler(optimizer, args, epoch_start):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        args               -- stores all the experiment flags
        epoch_start        -- the epoch number we started/continued from
    We keep the same learning rate for the first <args.n_epochs> epochs
    and linearly decay the rate to zero over the next <args.n_epochs_decay> epochs.
    """

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + epoch_start - args.n_epochs) / float(args.n_epochs_decay + 1)
        return lr_l

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    return scheduler


def update_learning_rate(scheduler, optimizer):
    """Update learning rates; called at the end of every epoch"""
    old_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]
    print("learning rate update from %.7f -> %.7f" % (old_lr, lr))
    return lr


def write_log_file(log_file_name, entry, epoch, lrate, phase):
    """记录训练/验证损失。

    entry 为字典：
      Train 需含 total/l2d/l3d/lrecon/lcross/mse；
      Val   仅需 mse（重建 MSE）。
    """
    with open(log_file_name, "a") as f:
        if phase == "Train":
            f.write(
                "\n{} LRate: {:.7f} Epoch: {} | Total: {:.6f} | "
                "L2D: {:.6f} | L3D: {:.6f} | Lrecon: {:.6f} | Lcross: {:.6f} | "
                "ReconMSE: {:.6f}".format(
                    phase, lrate, epoch,
                    entry["total"], entry["l2d"], entry["l3d"], entry["lrecon"], entry["lcross"], entry["mse"],
                )
            )
        else:
            f.write(
                "\n{} LRate: {:.7f} Epoch: {} | ReconMSE: {:.6f}".format(
                    phase, lrate, epoch, entry["mse"]
                )
            )


def main_worker(args):
    # setup training data
    trainloader, valloader = setup_data(args)

    device = torch.device("cuda:0")
    UVDocnet = model.UVDocnet(num_filter=32, kernel_size=5)
    UVDocnet.to(device)

    # define loss functions
    criterionL1 = torch.nn.L1Loss()
    criterionMSE = torch.nn.MSELoss()

    # ddcp cross loss (reused, not redefined): multi-scale differential-coordinate loss on grid2D
    ddcp_losses = Losses(classify_size_average=True, args_gpu=0)
    cross_loss_fn = ddcp_losses.loss_fn4_v5_r_4

    global gamma_w
    epoch_start = 0
    ft_lr = args.lr  # 续训/从头训练时使用 args.lr；微调时改为 args.finetune_lr

    if args.resume is not None:
        # ---- 续训：加载模型 + 优化器状态，沿用 epoch 与 LR 调度 ----
        if os.path.isfile(args.resume):
            print("Continue training from checkpoint '{}' (model + optimizer)".format(args.resume))
            checkpoint = torch.load(args.resume)
            # strict=False：容忍旧权重缺失多尺度 aux 头（随机初始化）
            UVDocnet.load_state_dict(checkpoint["model_state"], strict=False)
            optimizer = torch.optim.Adam(UVDocnet.parameters(), lr=args.lr, betas=(0.9, 0.999))
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            epoch_start = checkpoint["epoch"]
            print("Resumed from '{}' (epoch {})".format(args.resume, epoch_start))
            if epoch_start >= args.ep_gamma_start:
                gamma_w = args.gamma_w
        else:
            print("No checkpoint found at '{}', training from scratch.".format(args.resume))
            optimizer = torch.optim.Adam(UVDocnet.parameters(), lr=args.lr, betas=(0.9, 0.999))
    elif args.finetune is not None:
        # ---- 微调：只加载权重，重置优化器与 epoch=0，使用新的（更小的）LR 调度 ----
        if os.path.isfile(args.finetune):
            print("Fine-tuning from weights '{}' (fresh optimizer, epoch starts at 0)".format(args.finetune))
            checkpoint = torch.load(args.finetune)
            state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
            # strict=False：若已有权重来自旧模型（无 aux 头），aux_head_s4/s8 随机初始化，
            # 微调时由多尺度 cross 损失进一步训练；反之新权重精确加载。
            missing, unexpected = UVDocnet.load_state_dict(state, strict=False)
            if missing:
                print(f"[finetune] missing keys (kept random init): {missing}")
            if unexpected:
                print(f"[finetune] unexpected keys (ignored): {unexpected}")
            ft_lr = args.finetune_lr
            epoch_start = 0
        else:
            print("No checkpoint found at '{}', training from scratch with finetune_lr.".format(args.finetune))
        optimizer = torch.optim.Adam(UVDocnet.parameters(), lr=ft_lr, betas=(0.9, 0.999))
    else:
        # ---- 从头训练 ----
        optimizer = torch.optim.Adam(UVDocnet.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # ---- 微调策略：冻结骨干，只训练输出头与多尺度 aux 头（可选）----
    # 已有权重提供了强先验，冻结 encoder/bridges 可稳定微调、防止灾难性遗忘，
    # 仅更新 2D/3D 输出头与新加的 aux 头。通常与 --finetune 配合使用。
    if args.freeze_backbone:
        trainable = ["out_point_positions2D", "out_point_positions3D", "aux_head_s4", "aux_head_s8"]
        for name, p in UVDocnet.named_parameters():
            p.requires_grad = any(t in name for t in trainable)
        n_train = sum(p.numel() for p in UVDocnet.parameters() if p.requires_grad)
        print("Backbone frozen; training only {} ({} params).".format(trainable, n_train))

    # initialize learning rate schedulers
    scheduler = get_scheduler(optimizer, args, epoch_start)

    # Log file:
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    experiment_name = (
        "params"
        + str(args.batch_size)
        + "_lr="
        + str(args.lr)
        + "_nepochs"
        + str(args.n_epochs)
        + "_nepochsdecay"
        + str(args.n_epochs_decay)
        + "_alpha"
        + str(args.alpha_w)
        + "_beta"
        + str(args.beta_w)
        + "_gamma="
        + str(args.gamma_w)
        + "_crossW="
        + str(args.cross_w)
        + "_gammastartep"
        + str(args.ep_gamma_start)
        + "_data"
        + args.data_to_use
    )
    if args.resume:
        experiment_name = "RESUME" + experiment_name

    log_file_name = os.path.join(args.logdir, experiment_name + ".txt")
    if os.path.isfile(log_file_name):
        log_file = open(log_file_name, "a")
    else:
        log_file = open(log_file_name, "w+")

    log_file.write("\n---------------  " + experiment_name + "  ---------------\n")
    log_file.close()

    exp_log_dir = os.path.join(args.logdir, experiment_name, "")
    if not os.path.exists(exp_log_dir):
        os.makedirs(exp_log_dir)

    global losscount
    global train_mse
    global loss2D_acc, loss3D_acc, lossrecon_acc, losscross_acc, losstotal_acc

    # Run training
    for epoch in range(epoch_start, args.n_epochs + args.n_epochs_decay + 1):
        print(f"\n----- Epoch {epoch} -----")
        if epoch >= args.ep_gamma_start:
            gamma_w = args.gamma_w
            print("epoch ", epoch, "gamma_w is now", gamma_w)

        train_mse = 0.0
        best_val_mse = 99999.0
        losscount = 0
        loss2D_acc = loss3D_acc = lossrecon_acc = losscross_acc = losstotal_acc = 0.0

        # Train
        UVDocnet.train()

        for batch in trainloader:
            (
                imgs_doc3D_,
                imgs_unwarped_doc3D_,
                grid2D_doc3D_,
                grid3D_doc3D_,
                imgs_UVDoc_,
                imgs_unwarped_UVDoc_,
                grid2D_UVDoc_,
                grid3D_UVDoc_,
            ) = (None, None, None, None, None, None, None, None)

            if args.data_to_use == "both":
                (
                    imgs_doc3D_,
                    imgs_unwarped_doc3D_,
                    grid2D_doc3D_,
                    grid3D_doc3D_,
                ) = batch[0]
                (
                    imgs_UVDoc_,
                    imgs_unwarped_UVDoc_,
                    grid2D_UVDoc_,
                    grid3D_UVDoc_,
                    uvmap_UVDoc_,
                ) = batch[1]
            elif args.data_to_use == "doc3d":
                (
                    imgs_doc3D_,
                    imgs_unwarped_doc3D_,
                    grid2D_doc3D_,
                    grid3D_doc3D_,
                ) = batch
            elif args.data_to_use == "uvdoc":
                (
                    imgs_UVDoc_,
                    imgs_unwarped_UVDoc_,
                    grid2D_UVDoc_,
                    grid3D_UVDoc_,
                    uvmap_UVDoc_,
                ) = batch

            train_doc3d = args.data_to_use in ("both", "doc3d")
            train_uvdoc = args.data_to_use in ("both", "uvdoc")

            # ---- Doc3D step ----
            if train_doc3d:
                imgs_doc3D = imgs_doc3D_.to(device, non_blocking=True)
                unwarped_GT_doc3D = imgs_unwarped_doc3D_.to(device, non_blocking=True)
                grid2D_GT_doc3D = grid2D_doc3D_.to(device, non_blocking=True)
                grid3D_GT_doc3D = grid3D_doc3D_.to(device, non_blocking=True)

                # grid2D_pred_doc3D, grid3D_pred_doc3D, grid2D_aux_doc3D = UVDocnet(imgs_doc3D, return_aux=True)
                grid2D_pred_doc3D, grid3D_pred_doc3D = UVDocnet(imgs_doc3D)

                unwarped_pred_doc3D = utils.bilinear_unwarping(imgs_doc3D, grid2D_pred_doc3D, utils.IMG_SIZE)

                optimizer.zero_grad(set_to_none=True)

                recon_loss = criterionL1(unwarped_pred_doc3D, unwarped_GT_doc3D)
                loss_grid2D = criterionL1(grid2D_pred_doc3D, grid2D_GT_doc3D)
                loss_grid3D = criterionL1(grid3D_pred_doc3D, grid3D_GT_doc3D)
                cross_loss = single_scale_cross_loss(
                    grid2D_pred_doc3D, grid2D_GT_doc3D, cross_loss_fn, alpha=args.cross_alpha
                )

                netLoss = (
                    args.alpha_w * loss_grid2D
                    + args.beta_w * loss_grid3D
                    + gamma_w * recon_loss
                    + args.cross_w * cross_loss
                )
                netLoss.backward()
                optimizer.step()

                # 累计各项（带权重）损失分量
                loss2D_acc += float(args.alpha_w * loss_grid2D)
                loss3D_acc += float(args.beta_w * loss_grid3D)
                lossrecon_acc += float(gamma_w * recon_loss)
                losscross_acc += float(args.cross_w * cross_loss)
                losstotal_acc += float(netLoss)

                tmp_mse = criterionMSE(unwarped_pred_doc3D, unwarped_GT_doc3D)
                train_mse += float(tmp_mse)
                losscount += 1

            # ---- UVDoc step (dataset returns a 5-tuple: img, unwarped, grid2D, grid3D, uvmap) ----
            if train_uvdoc:
                imgs_UVDoc = imgs_UVDoc_.to(device, non_blocking=True)
                unwarped_GT_UVDoc = imgs_unwarped_UVDoc_.to(device, non_blocking=True)
                grid2D_GT_UVDoc = grid2D_UVDoc_.to(device, non_blocking=True)
                grid3D_GT_UVDoc = grid3D_UVDoc_.to(device, non_blocking=True)

                # grid2D_pred_UVDoc, grid3D_pred_UVDoc, grid2D_aux_UVDoc = UVDocnet(imgs_UVDoc, return_aux=True)
                grid2D_pred_UVDoc, grid3D_pred_UVDoc = UVDocnet(imgs_UVDoc)
                
                unwarped_pred_UVDoc = utils.bilinear_unwarping(imgs_UVDoc, grid2D_pred_UVDoc, utils.IMG_SIZE)

                optimizer.zero_grad(set_to_none=True)

                recon_loss = criterionL1(unwarped_pred_UVDoc, unwarped_GT_UVDoc)
                loss_grid2D = criterionL1(grid2D_pred_UVDoc, grid2D_GT_UVDoc)
                loss_grid3D = criterionL1(grid3D_pred_UVDoc, grid3D_GT_UVDoc)
                cross_loss = single_scale_cross_loss(
                    grid2D_pred_UVDoc, grid2D_GT_UVDoc, cross_loss_fn, alpha=args.cross_alpha
                )

                netLoss = (
                    args.alpha_w * loss_grid2D
                    + args.beta_w * loss_grid3D
                    + gamma_w * recon_loss
                    + args.cross_w * cross_loss
                )
                netLoss.backward()
                optimizer.step()

                # 累计各项（带权重）损失分量
                loss2D_acc += float(args.alpha_w * loss_grid2D)
                loss3D_acc += float(args.beta_w * loss_grid3D)
                lossrecon_acc += float(gamma_w * recon_loss)
                losscross_acc += float(args.cross_w * cross_loss)
                losstotal_acc += float(netLoss)

                tmp_mse = criterionMSE(unwarped_pred_UVDoc, unwarped_GT_UVDoc)
                train_mse += float(tmp_mse)
                losscount += 1
            gc.collect()

        train_mse = train_mse / max(1, losscount)
        curr_lr = update_learning_rate(scheduler, optimizer)
        # 平均到每个训练步，便于与总损失对照
        n = max(1, losscount)
        write_log_file(
            log_file_name,
            {
                "total": losstotal_acc / n,
                "l2d": loss2D_acc / n,
                "l3d": loss3D_acc / n,
                "lrecon": lossrecon_acc / n,
                "lcross": losscross_acc / n,
                "mse": train_mse,
            },
            epoch + 1,
            curr_lr,
            "Train",
        )

        # Evaluation
        UVDocnet.eval()

        with torch.no_grad():
            mse_loss_val = 0.0
            for imgs_val_, imgs_unwarped_val_, *_ in valloader:
                imgs_val = imgs_val_.to(device)
                unwarped_GT_val = imgs_unwarped_val_.to(device)

                grid2D_pred_val, grid3D_pred_val = UVDocnet(imgs_val)
                unwarped_pred_val = utils.bilinear_unwarping(imgs_val, grid2D_pred_val, utils.IMG_SIZE)

                loss_img_val = criterionMSE(unwarped_pred_val, unwarped_GT_val)
                mse_loss_val += float(loss_img_val)

            val_mse = mse_loss_val / len(valloader)
            write_log_file(log_file_name, {"mse": val_mse}, epoch + 1, curr_lr, "Val")

        # save best models
        if val_mse < best_val_mse or epoch == args.n_epochs + args.n_epochs_decay:
            best_val_mse = val_mse
            state = {
                "epoch": epoch + 1,
                "model_state": UVDocnet.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            model_path = exp_log_dir + f"ep_{epoch + 1}_{val_mse:.5f}_{train_mse:.5f}_best_model.pkl"
            torch.save(state, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")

    parser.add_argument(
        "--data_path_doc3D", nargs="?", type=str, default="./data/doc3D/", help="Data path to load Doc3D data."
    )
    parser.add_argument(
        "--data_path_UVDoc", nargs="?", type=str, default="./data/UVDoc/", help="Data path to load UVDoc data."
    )
    parser.add_argument(
        "--data_to_use",
        type=str,
        default="uvdoc",
        choices=["both", "doc3d", "uvdoc"],
        help="Dataset to use for training: 'doc3d' for Doc3D only, 'uvdoc' for UVDoc only, or 'both' for both.",
    )
    parser.add_argument("--batch_size", nargs="?", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--n_epochs",
        nargs="?",
        type=int,
        default=10,
        help="Number of epochs with initial (constant) learning rate.",
    )
    parser.add_argument(
        "--n_epochs_decay",
        nargs="?",
        type=int,
        default=30,
        help="Number of epochs to linearly decay learning rate to zero.",
    )
    parser.add_argument("--lr", nargs="?", type=float, default=0.0002, help="Initial learning rate.")
    parser.add_argument("--alpha_w", nargs="?", type=float, default=5.0, help="Weight for the 2D grid L1 loss.")
    parser.add_argument("--beta_w", nargs="?", type=float, default=0.5, help="Weight for the 3D grid L1 loss.")
    parser.add_argument(
        "--gamma_w", nargs="?", type=float, default=1.0, help="Weight for the image reconstruction loss."
    )
    parser.add_argument(
        "--ep_gamma_start",
        nargs="?",
        type=int,
        default=10,
        help="Epoch from which to start using image reconstruction loss.",
    )
    parser.add_argument(
        "--cross_w",
        nargs="?",
        type=float,
        default=0.0,
        help="Weight for the multi-scale cross loss (reused ddcp loss_fn4_v5_r_4) added on grid2D.",
    )
    parser.add_argument(
        "--cross_alpha",
        nargs="?",
        type=float,
        default=0.1,
        help="Weight of the local (differential-coordinate) term inside the ddcp cross loss (matches ddcp lambda_loss_a=0.1).",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        default=None,
        help="Continue an interrupted training: loads BOTH model and optimizer state and "
             "resumes the epoch count and LR schedule. Mutually exclusive with --finetune.",
    )
    parser.add_argument(
        "--finetune",
        nargs="?",
        type=str,
        default=None,
        help="Fine-tune from existing weights: loads ONLY the model weights (no optimizer), "
             "starts a fresh optimizer, resets epoch to 0 and uses a fresh LR schedule. "
             "Defaults to the provided pretrained weights ./model/best_model.pkl.",
    )
    parser.add_argument(
        "--finetune_lr",
        nargs="?",
        type=float,
        default=2e-5,
        help="LR used in --finetune mode (typically ~10x smaller than the original training "
             "lr, e.g. 2e-5 vs 2e-4). Ignored unless --finetune is set.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=False,
        help="Fine-tune strategy: freeze the pretrained encoder/bridges and only train the "
             "output heads (2D/3D grid) and the multi-scale auxiliary heads. Strong prior, "
             "fast convergence. Combine with --finetune.",
    )
    parser.add_argument("--logdir", nargs="?", type=str, default="./log/default", help="Path to store the logs.")
    parser.add_argument(
        "-a",
        "--appearance_augmentation",
        nargs="*",
        type=str,
        default=["visual", "noise", "color"],
        choices=["shadow", "blur", "visual", "noise", "color"],
        help="Appearance augmentations to use.",
    )
    parser.add_argument(
        "-gUVDoc",
        "--geometric_augmentationsUVDoc",
        nargs="*",
        type=str,
        default=["rotate"],
        choices=["rotate", "flip", "perspective"],
        help="Geometric augmentations to use for the UVDoc dataset.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to use for the dataloaders.")

    args = parser.parse_args()
    main_worker(args)