import torch.nn as nn
from ultralytics.nn.modules import Conv, Detect
from ultralytics.nn.tasks import DetectionModel


class HeatmapClsHead(nn.Module):
    """分类 + 热力图输出头：P3 上输出 K 通道热力图，P5 上接分类"""

    def __init__(self, nc=1, num_keypoints=4, ch=(128, 256, 512)):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.nc = nc
        self.reg_max = 16
        self.kpt_conv = nn.Sequential(
            Conv(ch[0], 128, 3),
            Conv(128, num_keypoints, 1),
        )
        self.cls_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(ch[-1], 128),
            nn.ReLU(),
            nn.Linear(128, nc),
        )

    def forward(self, x):
        kpt_heatmap = self.kpt_conv(x[0])
        cls_logits = self.cls_fc(x[-1])
        return cls_logits, kpt_heatmap


class HeatmapClsModel(DetectionModel):
    def __init__(self, cfg="yolov8s.yaml", ch=3, nc=1, num_keypoints=4):
        super().__init__(cfg, ch, nc)
        for i, layer in enumerate(self.model):
            if isinstance(layer, Detect):
                new_head = HeatmapClsHead(nc=nc, num_keypoints=num_keypoints, ch=(128, 256, 512))
                new_head.stride = layer.stride
                new_head.reg_max = getattr(layer, "reg_max", 16)
                new_head.f = getattr(layer, "f", -1)
                new_head.i = getattr(layer, "i", 0)
                new_head.save = getattr(layer, "save", False)
                self.model[i] = new_head
                break
        # yolov8s 结构中 P3/P4/P5 对应的层索引
        self.p3_idx = 15
        self.p4_idx = 18
        self.p5_idx = 21
        self.num_keypoints = num_keypoints
        self.nc = nc
        self.criterion = None

    def _forward_features(self, x):
        """遍历除输出头以外的所有层，返回多尺度特征 [P3, P4, P5]"""
        y = []
        for i, m in enumerate(self.model):
            if isinstance(m, HeatmapClsHead):
                break
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
        return [y[self.p3_idx], y[self.p4_idx], y[self.p5_idx]]

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            # 训练路径：dict -> 特征 -> 输出头 -> 损失
            img = x["img"]
            features = self._forward_features(img)
            preds = self.model[-1](features)
            if self.criterion is not None:
                return self.criterion(preds, x)
            return preds
        # 张量输入（stride 计算 / 推理）：走父类 forward
        return DetectionModel.forward(self, x, *args, **kwargs)
