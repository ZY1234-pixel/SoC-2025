import torch
import torch.nn as nn


class ClsHeatmapLoss:
    """分类损失 + 热力图 MSE 损失（逐可见像素归一化）"""

    def __init__(self, num_keypoints=4, heatmap_weight=1.0, cls_weight=1.0):
        self.num_keypoints = num_keypoints
        self.heatmap_weight = heatmap_weight
        self.cls_weight = cls_weight
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def __call__(self, preds, batch):
        cls_logits, kpt_heatmap = preds

        # 分类损失
        cls_label = batch["cls_label"]
        if not isinstance(cls_label, torch.Tensor):
            cls_label = torch.tensor(cls_label, dtype=torch.long, device=cls_logits.device)
        else:
            cls_label = cls_label.long().to(cls_logits.device)
        loss_cls = self.ce(cls_logits, cls_label)

        # 热力图损失（验证阶段 batch 中没有 heatmap 时跳过）
        if "heatmap" in batch:
            target_heatmap = batch["heatmap"].to(kpt_heatmap.device)
            keypoints = batch["keypoints"]
            vis_mask = (keypoints[..., 2] > 0.5).unsqueeze(-1).unsqueeze(-1)
            diff = (kpt_heatmap - target_heatmap) ** 2
            # 分母为可见像素总数（vis 数 * H * W），保证是真正的逐像素 MSE
            H, W = kpt_heatmap.shape[2], kpt_heatmap.shape[3]
            loss_heatmap = (diff * vis_mask).sum() / ((vis_mask.sum() * H * W) + 1e-6)
        else:
            loss_heatmap = torch.tensor(0.0, device=cls_logits.device)

        total_loss = self.cls_weight * loss_cls + self.heatmap_weight * loss_heatmap
        loss_items = {
            "cls_loss": loss_cls.detach(),
            "heatmap_loss": loss_heatmap.detach(),
        }
        return total_loss, loss_items
