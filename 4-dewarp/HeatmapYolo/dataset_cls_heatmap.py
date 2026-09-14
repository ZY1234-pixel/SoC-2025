import yaml
from pathlib import Path

import torch
from ultralytics.data.dataset import YOLODataset


class ClsHeatmapDataset(YOLODataset):
    """在 YOLODataset 基础上提取类别标签和第一个对象的关键点"""

    def __init__(self, *args, num_keypoints=4, **kwargs):
        self.num_keypoints = num_keypoints
        kwargs["task"] = "pose"  # 强制启用关键点解析
        data = kwargs.get("data")
        if isinstance(data, (str, Path)):
            with open(data, "r", encoding="utf-8") as f:
                kwargs["data"] = yaml.safe_load(f)
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)

        cls_labels = item.get("cls")
        if cls_labels is not None and cls_labels.numel() > 0:
            cls_label = int(cls_labels[0].item())
        else:
            cls_label = 0

        keypoints = item.get("keypoints")
        if keypoints is not None and keypoints.numel() > 0:
            kpt = keypoints[0].reshape(self.num_keypoints, 3).float()
        else:
            raise ValueError("图像中没有读取到关键点，请检查 data.yaml 的 kpt_shape: [4, 3]")

        item["cls_label"] = torch.tensor(cls_label, dtype=torch.long)
        item["keypoints"] = kpt
        return item
