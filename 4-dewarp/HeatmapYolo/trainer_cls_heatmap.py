import torch
from ultralytics.data import build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG_DICT

from dataset_cls_heatmap import ClsHeatmapDataset
from heatmap_cls_model import HeatmapClsModel
from heatmap_utils import generate_heatmaps
from loss_cls_heatmap import ClsHeatmapLoss


class DummyMetrics:
    @property
    def keys(self):
        return ["metrics/fitness", "metrics/cls_acc"]


class ClsHeatmapValidator(BaseValidator):
    """占位验证器：本任务只训练不评估 bbox，直接跳过官方验证逻辑"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args.task = "pose"
        self.metrics = DummyMetrics()

    def __call__(self, trainer=None, model=None):
        print("[Info] 验证阶段：无需边界框，跳过标准 NMS 评估...")
        return {"metrics/fitness": 0.0, "metrics/cls_acc": 0.0}


class ClsHeatmapTrainer(DetectionTrainer):
    def __init__(self, cfg, overrides, _callbacks):
        self.model_cfg = cfg
        self.num_keypoints = overrides.pop("num_keypoints", 4)
        self.heatmap_weight = overrides.pop("heatmap_weight", 10.0)
        self.cls_weight = overrides.pop("cls_weight", 1.0)

        if "model" not in overrides or overrides["model"] is None:
            overrides["model"] = overrides.get("pretrained", "yolov8s.pt")

        train_cfg = DEFAULT_CFG_DICT.copy()
        train_cfg.update(overrides)
        super().__init__(cfg=train_cfg, overrides={}, _callbacks=_callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = HeatmapClsModel(
            cfg=self.model_cfg,
            ch=3,
            nc=self.data.get("nc", 1),
            num_keypoints=self.num_keypoints,
        )
        if weights:
            model.load(weights)
        model.to(self.device)
        model.criterion = self.get_loss(model)
        return model

    def get_loss(self, model):
        return ClsHeatmapLoss(
            num_keypoints=self.num_keypoints,
            heatmap_weight=self.heatmap_weight,
            cls_weight=self.cls_weight,
        )

    def get_dataloader(self, dataset_path, batch_size, rank=0, mode="train"):
        dataset = ClsHeatmapDataset(
            img_path=dataset_path,
            imgsz=self.args.imgsz,
            batch_size=batch_size,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache,
            single_cls=self.args.single_cls,
            stride=int(self.model.stride.max()),
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            num_keypoints=self.num_keypoints,
            task="pose",
            data=self.args.data,
        )
        return build_dataloader(dataset, batch_size, self.args.workers, rank=rank)

    def preprocess_batch(self, batch):
        img = batch["img"].to(self.device)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        batch["img"] = img

        keypoints = batch["keypoints"].to(self.device)
        B = img.shape[0]
        keypoints = keypoints.reshape(B, self.num_keypoints, 3)

        # 保留增强产生的可见性标记（旋转/透视把角点移出画面时 ultralytics 会置 v=0），
        # 并把归一化坐标越界的点也标为不可见，避免生成越界/错位的热力图目标
        out = (
            (keypoints[..., 0] < 0.0)
            | (keypoints[..., 0] > 1.0)
            | (keypoints[..., 1] < 0.0)
            | (keypoints[..., 1] > 1.0)
        )
        keypoints[..., 2] = ((keypoints[..., 2] > 0.5) & (~out)).float()
        batch["keypoints"] = keypoints

        # 下采样 8 倍生成热力图（与输出头 P3 stride 一致）
        img_h, img_w = img.shape[2], img.shape[3]
        feat_h, feat_w = img_h // 8, img_w // 8
        heatmaps = generate_heatmaps(keypoints, img_shape=(feat_h, feat_w), sigma=3.0, device=self.device)
        batch["heatmap"] = heatmaps
        return batch

    def get_validator(self):
        self.loss_names = ("cls_loss", "heatmap_loss")
        return ClsHeatmapValidator(
            dataloader=self.test_loader,
            save_dir=self.save_dir,
            args=self.args,
            _callbacks=self.callbacks,
        )
