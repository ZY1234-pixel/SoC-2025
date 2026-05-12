from ultralytics import YOLO

# 数据集路径
data_path = r"YoloV8_ClassFor6/cls_dataset"

# 预训练模型（yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt 等）
model_name = "yolov8n-cls.pt"

# 输入尺寸
imgsz = 256

# 训练轮数
epochs = 100

# 批次大小
batch_size = 32

# 实验名称（训练结果保存在 runs/classify/name 下）
name = "book_cls_6class"

# 在训练结束后对验证集进行评估
evaluate_after_train = True

def main():
    # 加载预训练模型
    model = YOLO(model_name)

    # 开始训练
    results = model.train(
        data=data_path,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch_size,
        name=name,
        project="YoloV8_ClassFor6",
        # 可选参数
        # patience=10,          # 早停轮数
        # lr0=0.01,             # 初始学习率
        # optimizer='Adam',     # 优化器
        # augment=True,         # 默认已开启数据增强
    )

    # 训练完成后评估
    if evaluate_after_train:
        print("\n开始验证集评估...")
        valid_results = model.val()  # 自动使用训练中保存的最佳权重
        print(f"Top-1 Accuracy: {valid_results.top1:.4f}")
        print(f"Top-5 Accuracy: {valid_results.top5:.4f}")

if __name__ == "__main__":
    main()