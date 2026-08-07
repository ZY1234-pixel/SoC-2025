import os

from trainer_cls_heatmap import ClsHeatmapTrainer


def main():
    model_cfg_path = os.path.join("configs", "yolov8s.yaml")

    overrides = {
        # ---- 必填项 ----
        "model": "yolov8s.pt",  # 预训练权重（不存在时会自动下载）
        "data": os.path.join("doc_4corners_heatmap.yaml"),
        "imgsz": 512,
        "batch": 32,
        "epochs": 100,
        "device": 0,
        "workers": 8,

        # ---- 优化器 ----
        "optimizer": "SGD",
        "lr0": 0.005,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,

        # ---- 增强（旋转 + 透视组合；角点被移出画面时自动标记不可见并屏蔽 loss）----
        "mosaic": 0.0,
        "close_mosaic": 0,
        "degrees": 10.0,
        "shear": 0.0,
        "perspective": 0.001,
        "scale": 0.1,
        "translate": 0.05,
        "fliplr": 0.5,
        "hsv_h": 0.02,
        "hsv_s": 0.02,
        "hsv_v": 0.02,
        "amp": True,

        # ---- 自定义参数 ----
        "num_keypoints": 4,
        # loss 为逐像素 MSE，此值 = 10 * (imgsz//8)^2，与原来的 10.0 梯度权重等价
        "heatmap_weight": 40960.0,
        "cls_weight": 1.0,

        # ---- 关键修复：防止热力图头通道塌缩 ----
        # 默认 warmup_bias_lr=0.1 会让头部的 BN/偏置以 0.1 的学习率起步，
        # 把头部参数打爆，导致某些/全部角点通道输出全 0，必须设为 0
        "warmup_bias_lr": 0.0,

        # ---- 其他 ----
        "patience": 0,
        "save": True,
        "plots": True,
        "project": "runs_heatmap",
        "name": "heatmap_v12_512_aug",
        "exist_ok": True,
    }

    print("启动【分类 + 热力图】训练...")
    trainer = ClsHeatmapTrainer(cfg=model_cfg_path, overrides=overrides, _callbacks={})
    trainer.train()

    # 训练完成标记
    done_path = os.path.join("runs_heatmap", "heatmap_v12_512_aug", "TRAINING_DONE.txt")
    with open(done_path, "w", encoding="utf-8") as f:
        f.write("训练完成\n")
    print("训练完成，权重保存在:", os.path.join("runs_heatmap", "heatmap_v12_512_aug", "weights"))


if __name__ == "__main__":
    main()
