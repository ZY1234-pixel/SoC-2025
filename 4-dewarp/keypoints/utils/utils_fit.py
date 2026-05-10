import os

import torch
from nets.deeplabv3_training import (CE_Loss, Dice_loss, Focal_Loss, Boundary_Loss, Tversky_loss,
                                     weights_init)
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                  epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda,
                  dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
                  save_period, save_dir, local_rank,
                  num_keypoints=0, kpt_loss_weight=1.0):
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        if num_keypoints > 0:
            imgs, pngs, labels, kpt_heatmaps = batch
        else:
            imgs, pngs, labels = batch
            kpt_heatmaps = None

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
                if kpt_heatmaps is not None:
                    kpt_heatmaps = kpt_heatmaps.cuda(local_rank)

        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            #----------------------#
            #   解析多任务输出
            #----------------------#
            if isinstance(outputs, tuple):
                seg_outputs, kpt_outputs = outputs
            else:
                seg_outputs = outputs
                kpt_outputs = None

            #----------------------#
            #   计算分割损失
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(seg_outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(seg_outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(seg_outputs, labels)
                loss      = loss + main_dice
            b_loss = Boundary_Loss(seg_outputs, pngs)
            loss = loss + 0.3 * b_loss

            #----------------------#
            #   计算关键点损失
            #----------------------#
            if num_keypoints > 0 and kpt_outputs is not None and kpt_heatmaps is not None:
                kpt_loss = torch.nn.functional.mse_loss(kpt_outputs, kpt_heatmaps)
                loss += kpt_loss_weight * kpt_loss

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(seg_outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   解析多任务输出
                #----------------------#
                if isinstance(outputs, tuple):
                    seg_outputs, kpt_outputs = outputs
                else:
                    seg_outputs = outputs
                    kpt_outputs = None

                #----------------------#
                #   计算分割损失
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(seg_outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(seg_outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Tversky_loss(seg_outputs, labels)
                    loss      = loss + main_dice
                b_loss = Boundary_Loss(seg_outputs, pngs)
                loss = loss + 0.3 * b_loss

                #----------------------#
                #   计算关键点损失
                #----------------------#
                if num_keypoints > 0 and kpt_outputs is not None and kpt_heatmaps is not None:
                    kpt_loss = torch.nn.functional.mse_loss(kpt_outputs, kpt_heatmaps)
                    loss += kpt_loss_weight * kpt_loss

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(seg_outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        if num_keypoints > 0:
            imgs, pngs, labels, kpt_heatmaps = batch
        else:
            imgs, pngs, labels = batch
            kpt_heatmaps = None

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
                if kpt_heatmaps is not None:
                    kpt_heatmaps = kpt_heatmaps.cuda(local_rank)

            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            if isinstance(outputs, tuple):
                seg_outputs, kpt_outputs = outputs
            else:
                seg_outputs = outputs
                kpt_outputs = None

            #----------------------#
            #   计算分割损失
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(seg_outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(seg_outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Tversky_loss(seg_outputs, labels)
                loss  = loss + main_dice
            b_loss = Boundary_Loss(seg_outputs, pngs)
            loss = loss + 0.3 * b_loss

            #----------------------#
            #   计算关键点损失 (仅记录，不反向传播)
            #----------------------#
            if num_keypoints > 0 and kpt_outputs is not None and kpt_heatmaps is not None:
                kpt_loss = torch.nn.functional.mse_loss(kpt_outputs, kpt_heatmaps)
                loss += kpt_loss_weight * kpt_loss

            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            _f_score    = f_score(seg_outputs, labels)

            val_loss    += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                    'f_score'   : val_f_score / (iteration + 1),
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))