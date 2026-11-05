import argparse
import gc
import os

import torch

import data_doc3D
import data_UVDoc
import model
import utils
from data_mixDataset import mixDataset

train_mse = 0.0
losscount = 0
# 各损失分量（带权重）的逐 epoch 累加器，用于日志输出
loss2D_acc = 0.0    # Σ α·L_grid2D
loss3D_acc = 0.0    # Σ β·L_grid3D
lossrecon_acc = 0.0 # Σ γ·L_recon
lossal_acc = 0.0    # Σ δ·L_AL
losstotal_acc = 0.0 # Σ 总损失 netLoss
gamma_w = 0.0


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
      Train 需含 total/l2d/l3d/lrecon/lal/mse；
      Val   仅需 mse（重建 MSE）。
    """
    with open(log_file_name, "a") as f:
        if phase == "Train":
            f.write(
                "\n{} LRate: {:.7f} Epoch: {} | Total: {:.6f} | "
                "L2D: {:.6f} | L3D: {:.6f} | Lrecon: {:.6f} | LAL: {:.6f} | "
                "ReconMSE: {:.6f}".format(
                    phase, lrate, epoch,
                    entry["total"], entry["l2d"], entry["l3d"], entry["lrecon"], entry["lal"], entry["mse"],
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
    # 基础网(C-Net) + 自适应细化网(F-Net)
    net = model.UVDocIterative(num_filter=32, kernel_size=5)
    net.to(device)

    # define loss functions
    criterionL1 = torch.nn.L1Loss()
    criterionMSE = torch.nn.MSELoss()

    global gamma_w
    epoch_start = 0

    # ---------- 断点续训 ----------
    resume_optimizer_state = None
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            net.load_state_dict(checkpoint["model_state"])
            resume_optimizer_state = checkpoint["optimizer_state"]
            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
            epoch_start = checkpoint["epoch"]
            if epoch_start >= args.ep_gamma_start:
                gamma_w = args.gamma_w
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    # 从旧版 UVDocnet 权重初始化 C-Net 与 F-Net（joint 模式下的 seeding）
    if args.base_ckpt is not None and os.path.isfile(args.base_ckpt):
        print("Initializing C-Net/F-Net from base checkpoint '{}'".format(args.base_ckpt))
        base_ckpt = torch.load(args.base_ckpt)
        base_state = (
            base_ckpt["model_state"]
            if isinstance(base_ckpt, dict) and "model_state" in base_ckpt
            else base_ckpt
        )
        new_state = {}
        for k, v in base_state.items():
            new_state["base_net." + k] = v
            new_state["refine_net." + k] = v
        net.load_state_dict(new_state, strict=False)

    # ---------- 分阶段序贯训练（专利 CN121391677A [0135]-[0138]）----------
    #  cnet: 仅训练 C-Net（冻结 F-Net），得到收敛的 C-Net 权重；
    #  fnet: 用收敛的 C-Net 权重初始化 F-Net，冻结 C-Net，仅微调 F-Net。
    if args.train_stage == "cnet":
        for p in net.refine_net.parameters():
            p.requires_grad = False
        args.refine_iters = 0  # 仅 C-Net，不展开 F-Net 细化
        print("Stage 'cnet': training C-Net only (F-Net frozen, refine_iters=0).")
    elif args.train_stage == "fnet":
        cnet_ckpt = args.cnet_ckpt
        assert os.path.isfile(cnet_ckpt), f"Converged C-Net checkpoint not found: {cnet_ckpt}"
        print("Stage 'fnet': loading converged C-Net from '{}' and fine-tuning F-Net.".format(cnet_ckpt))
        ckpt = torch.load(cnet_ckpt)
        cnet_state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        # 收敛的 C-Net 权重即 net 的 base_net.* 子状态；若 ckpt 为完整 net 状态则去掉前缀，
        # 若为裸 UVDocnet 状态（如 model/best_model.pkl）则直接使用。
        uv_state = {k.replace("base_net.", ""): v for k, v in cnet_state.items() if k.startswith("base_net.")}
        if not uv_state:  # ckpt 本身是裸 UVDocnet 状态
            uv_state = cnet_state
        net.base_net.load_state_dict(uv_state, strict=True)
        net.refine_net.load_state_dict(uv_state, strict=True)
        for p in net.base_net.parameters():
            p.requires_grad = False
        print("Stage 'fnet': C-Net frozen, F-Net will be fine-tuned.")

        # 论文 F-Net 微调策略对齐（ArbDR / AADD）：
        #   AdamW + 初始 lr=1e-5 + 线性衰减（共 10 个 epoch）。
        # 进入 fnet 阶段即直接使用论文默认超参。
        args.optimizer = "adamw"
        args.weight_decay = 1e-4
        args.lr = 1e-5
        args.n_epochs = 0
        args.n_epochs_decay = 19  # 共 10 个 epoch，线性衰减到 ~0

    # initialize optimizers (only trainable parameters; respects frozen nets in staged training)
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "No trainable parameters! Check --train_stage / frozen nets."
    OptimizerCls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = OptimizerCls(
        trainable_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)

    # initialize learning rate schedulers
    scheduler = get_scheduler(optimizer, args, epoch_start)  # noqa

    # Log file:
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    experiment_name = (
        "params"
        + str(args.batch_size)
        + "_lr="
        + str(args.lr)
        + "_opt"
        + str(args.optimizer)
        + "_wd"
        + str(args.weight_decay)
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
        + "_gammastartep"
        + str(args.ep_gamma_start)
        + "_data"
        + args.data_to_use
        + "_refineIters"
        + str(args.refine_iters)
        + "_delta"
        + str(args.delta_w)
        + "_stage"
        + str(args.train_stage)
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
    global loss2D_acc, loss3D_acc, lossrecon_acc, lossal_acc, losstotal_acc

    # Run training
    for epoch in range(epoch_start, args.n_epochs + args.n_epochs_decay + 1):
        print(f"\n----- Epoch {epoch} -----")
        if epoch >= args.ep_gamma_start:
            gamma_w = args.gamma_w
            print("epoch ", epoch, "gamma_w is now", gamma_w)

        train_mse = 0.0
        best_val_mse = 99999.0
        losscount = 0
        loss2D_acc = loss3D_acc = lossrecon_acc = lossal_acc = losstotal_acc = 0.0

        # Train
        net.train()
        if args.train_stage == "fnet":
            net.base_net.eval()  # 冻结 C-Net 的 BN 统计量，使其在微调 F-Net 时真正固定

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

            # Train Doc3D step
            if train_doc3d:
                imgs_doc3D = imgs_doc3D_.to(device, non_blocking=True)
                unwarped_GT_doc3D = imgs_unwarped_doc3D_.to(device, non_blocking=True)
                grid2D_GT_doc3D = grid2D_doc3D_.to(device, non_blocking=True)
                grid3D_GT_doc3D = grid3D_doc3D_.to(device, non_blocking=True)

                # 展开 refine_iters 步自适应迭代细化（F-Net）；max_iter=0 时退化为仅 C-Net
                base_g2d, base_g3d, grids, imgs_r = net(imgs_doc3D, max_iter=args.refine_iters)
                unwarped_pred_doc3D = imgs_r[0]

                optimizer.zero_grad(set_to_none=True)

                loss_grid2D = criterionL1(grids[0], grid2D_GT_doc3D)
                loss_grid3D = criterionL1(base_g3d, grid3D_GT_doc3D)
                recon_loss = criterionL1(unwarped_pred_doc3D, unwarped_GT_doc3D)
                loss_al = utils.axis_alignment_loss(grids[0], grid2D_GT_doc3D)
                netLoss = (
                    args.alpha_w * loss_grid2D
                    + args.beta_w * loss_grid3D
                    + gamma_w * recon_loss
                    + args.delta_w * loss_al
                )
                # 各细化步：F-Net 不加 3D 损失（仅 2D 网格 + 重建 + 轴对齐正则）
                for m in range(1, len(grids)):
                    netLoss = netLoss + (
                        args.alpha_w * criterionL1(grids[m], grid2D_GT_doc3D)
                        + gamma_w * criterionL1(imgs_r[m], unwarped_GT_doc3D)
                        + args.delta_w * utils.axis_alignment_loss(grids[m], grid2D_GT_doc3D)
                    )

                netLoss.backward()
                optimizer.step()

                # 累计各项（带权重）损失分量
                loss2D_acc += float(args.alpha_w * loss_grid2D)
                loss3D_acc += float(args.beta_w * loss_grid3D)
                lossrecon_acc += float(gamma_w * recon_loss)
                lossal_acc += float(args.delta_w * loss_al)
                for m in range(1, len(grids)):
                    loss2D_acc += float(args.alpha_w * criterionL1(grids[m], grid2D_GT_doc3D))
                    lossrecon_acc += float(gamma_w * criterionL1(imgs_r[m], unwarped_GT_doc3D))
                    lossal_acc += float(args.delta_w * utils.axis_alignment_loss(grids[m], grid2D_GT_doc3D))
                losstotal_acc += float(netLoss)

                tmp_mse = criterionMSE(unwarped_pred_doc3D, unwarped_GT_doc3D)
                train_mse += float(tmp_mse)
                losscount += 1

            # Train UVDoc step
            if train_uvdoc:
                imgs_UVDoc = imgs_UVDoc_.to(device, non_blocking=True)
                unwarped_GT_UVDoc = imgs_unwarped_UVDoc_.to(device, non_blocking=True)
                grid2D_GT_UVDoc = grid2D_UVDoc_.to(device, non_blocking=True)
                grid3D_GT_UVDoc = grid3D_UVDoc_.to(device, non_blocking=True)
                uv_map_GT_UVDoc = uvmap_UVDoc_.to(device, non_blocking=True) if uvmap_UVDoc_ is not None else None

                base_g2d, base_g3d, grids, imgs_r = net(imgs_UVDoc, max_iter=args.refine_iters)
                unwarped_pred_UVDoc = imgs_r[0]

                optimizer.zero_grad(set_to_none=True)

                loss_grid2D = criterionL1(grids[0], grid2D_GT_UVDoc)
                loss_grid3D = criterionL1(base_g3d, grid3D_GT_UVDoc)
                recon_loss = criterionL1(unwarped_pred_UVDoc, unwarped_GT_UVDoc)
                loss_al = utils.axis_alignment_loss(grids[0], grid2D_GT_UVDoc, uv_map_gt=uv_map_GT_UVDoc)
                netLoss = (
                    args.alpha_w * loss_grid2D
                    + args.beta_w * loss_grid3D
                    + gamma_w * recon_loss
                    + args.delta_w * loss_al
                )
                for m in range(1, len(grids)):
                    netLoss = netLoss + (
                        args.alpha_w * criterionL1(grids[m], grid2D_GT_UVDoc)
                        + gamma_w * criterionL1(imgs_r[m], unwarped_GT_UVDoc)
                        + args.delta_w * utils.axis_alignment_loss(grids[m], grid2D_GT_UVDoc, uv_map_gt=uv_map_GT_UVDoc)
                    )

                netLoss.backward()
                optimizer.step()

                # 累计各项（带权重）损失分量
                loss2D_acc += float(args.alpha_w * loss_grid2D)
                loss3D_acc += float(args.beta_w * loss_grid3D)
                lossrecon_acc += float(gamma_w * recon_loss)
                lossal_acc += float(args.delta_w * loss_al)
                for m in range(1, len(grids)):
                    loss2D_acc += float(args.alpha_w * criterionL1(grids[m], grid2D_GT_UVDoc))
                    lossrecon_acc += float(gamma_w * criterionL1(imgs_r[m], unwarped_GT_UVDoc))
                    lossal_acc += float(args.delta_w * utils.axis_alignment_loss(grids[m], grid2D_GT_UVDoc, uv_map_gt=uv_map_GT_UVDoc))
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
                "lal": lossal_acc / n,
                "mse": train_mse,
            },
            epoch + 1,
            curr_lr,
            "Train",
        )

        # Evaluation
        net.eval()

        with torch.no_grad():
            mse_loss_val = 0.0
            for imgs_val_, imgs_unwarped_val_, *_ in valloader:
                imgs_val = imgs_val_.to(device)
                unwarped_GT_val = imgs_unwarped_val_.to(device)

                # 验证时仅评估 C-Net 初始矫正（max_iter=0），与推理自适应细化解耦
                _, _, grids_val, imgs_r_val = net(imgs_val, max_iter=0)
                unwarped_pred_val = imgs_r_val[0]

                loss_img_val = criterionMSE(unwarped_pred_val, unwarped_GT_val)
                mse_loss_val += float(loss_img_val)

            val_mse = mse_loss_val / len(valloader)
            write_log_file(log_file_name, {"mse": val_mse}, epoch + 1, curr_lr, "Val")

        # save best models
        if val_mse < best_val_mse or epoch == args.n_epochs + args.n_epochs_decay:
            best_val_mse = val_mse
            state = {
                "epoch": epoch + 1,
                "model_state": net.state_dict(),
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
        help="Dataset to use for training: 'both' for Doc3D and UVDoc, 'doc3d' for Doc3D only, 'uvdoc' for UVDoc only.",
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
        default=40,
        help="Number of epochs to linearly decay learning rate to zero.",
    )
    parser.add_argument("--lr", nargs="?", type=float, default=0.0002, help="Initial learning rate.")
    parser.add_argument(
        "--optimizer", nargs="?", type=str, default="adam",
        choices=["adam", "adamw"],
        help="Optimizer. Paper fine-tunes F-Net with AdamW (lr=1e-5, linear decay).",
    )
    parser.add_argument(
        "--weight_decay", nargs="?", type=float, default=0.0,
        help="Weight decay (used by AdamW). Set to 1e-4 for paper-aligned F-Net fine-tuning.",
    )
    parser.add_argument("--alpha_w", nargs="?", type=float, default=5.0, help="Weight for the 2D grid L1 loss.")
    parser.add_argument("--beta_w", nargs="?", type=float, default=5.0, help="Weight for the 3D grid L1 loss.")
    parser.add_argument(
        "--delta_w", nargs="?", type=float, default=2.0, help="Weight for the axis-aligned loss (L_AL, Eq.7 in AADD). Paper uses gamma=0.2."
    )
    parser.add_argument(
        "--refine_iters", nargs="?", type=int, default=3,
        help="Unrolled refinement steps (F-Net) during training. 0 = disable iterative refinement (C-Net only).",
    )
    parser.add_argument(
        "--base_ckpt", nargs="?", type=str, default=None,
        help="Path to a pretrained UVDocnet (.pkl) to initialize both C-Net and F-Net.",
    )
    parser.add_argument(
        "--train_stage", nargs="?", type=str, default="joint",
        choices=["joint", "cnet", "fnet"],
        help="Coarse-to-fine staged training. "
             "'joint': train C-Net and F-Net end-to-end (default). "
             "'cnet': train C-Net only (F-Net frozen) -> yields converged C-Net. "
             "'fnet': freeze C-Net, init F-Net from converged C-Net (--cnet_ckpt) and fine-tune it.",
    )
    parser.add_argument(
        "--cnet_ckpt", nargs="?", type=str, default="./model/best_model.pkl",
        help="Path to converged C-Net checkpoint used to seed F-Net in the 'fnet' stage.",
    )
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
        "--resume",
        nargs="?",
        type=str,
        default=None,
        help="Path to previous saved model to restart from.",
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
