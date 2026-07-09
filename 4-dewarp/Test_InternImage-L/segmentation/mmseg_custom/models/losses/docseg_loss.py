# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES


def _distance_weight_map(mask_np, theta0):
    mask = (mask_np > 0).astype(np.uint8)
    dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    return (1.0 + theta0 * (dist_in + dist_out)).astype(np.float32)


@LOSSES.register_module()
class DocSegCombinedLoss(nn.Module):
    """BCEWithLogits + SoftDice + optional distance-boundary BCE."""

    def __init__(self,
                 bce_weight=1.0,
                 dice_weight=1.0,
                 boundary_weight=0.5,
                 boundary_start_epoch=6,
                 boundary_theta0=3.0,
                 dice_smooth=1.0,
                 loss_weight=1.0,
                 loss_name='loss_docseg'):
        super(DocSegCombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.boundary_start_epoch = boundary_start_epoch
        self.boundary_theta0 = boundary_theta0
        self.dice_smooth = dice_smooth
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.register_buffer('current_epoch', torch.zeros((), dtype=torch.long))

    @property
    def loss_name(self):
        return self._loss_name

    def set_epoch(self, epoch):
        self.current_epoch.fill_(int(epoch))

    def _soft_dice_loss(self, pred_logit, target):
        pred = torch.sigmoid(pred_logit)
        pred = pred.reshape(pred.size(0), -1)
        target = target.float().reshape(target.size(0), -1)
        inter = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * inter + self.dice_smooth) / (union + self.dice_smooth)
        return 1.0 - dice.mean()

    def _boundary_loss(self, pred_logit, target):
        weights = []
        target_cpu = target.detach().cpu().numpy().astype(np.uint8)
        for mask_np in target_cpu:
            weights.append(_distance_weight_map(mask_np, self.boundary_theta0))
        phi = torch.from_numpy(np.stack(weights)).to(pred_logit.device)
        phi = phi.unsqueeze(1)
        bce_map = F.binary_cross_entropy_with_logits(
            pred_logit, target.unsqueeze(1).float(), reduction='none')
        return (bce_map * phi).mean()

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        if pred.size(1) != 1:
            raise ValueError('DocSegCombinedLoss expects one output channel.')

        target = (target > 0).long()
        bce = F.binary_cross_entropy_with_logits(
            pred, target.unsqueeze(1).float())
        dice = self._soft_dice_loss(pred, target)

        boundary = pred.new_tensor(0.0)
        if int(self.current_epoch.item()) >= self.boundary_start_epoch:
            boundary = self._boundary_loss(pred, target)

        total = (self.bce_weight * bce + self.dice_weight * dice +
                 self.boundary_weight * boundary)
        return self.loss_weight * total
