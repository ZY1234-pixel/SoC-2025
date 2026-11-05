# 书本中缝方向归一化与热力图细化

本目录是 `4-dewarp/keypoints/` 下独立的推理模块。先用 YOLOv8n 判断方向并预测中缝两端，再用局部热力图细化中缝。权重单独分发，不提交到 Git。

## 文件与权重

将单独提供的 `中缝热力图权重_20261105.zip` 解压到**本目录**，解压后的结构应为：

```text
spine_heatmap/
├── infer_spine_line_heatmap.py
├── infer_spine_auto_orientation.py
├── spine_line_heatmap_runtime.py
├── config.yaml
└── weights/
    ├── direction.pt  # 方向选择用 YOLOv8n
    ├── detector.pt   # 方向归一化后的粗中缝 YOLOv8n
    └── heatmap.pt    # 局部中缝热力图网络
```

不需要训练数据或训练时的 VGG16 预训练文件。已验证环境：Python 3.10、Ultralytics 8.0.180、PyTorch 2.5.1+cu118、torchvision 0.20.1+cu118、OpenCV 4.10.0、NumPy 1.26.3。

## 批量推理

在本目录执行：

```bash
python infer_spine_line_heatmap.py 输入图片文件夹 输出文件夹
```

支持 JPG、JPEG、PNG；有 CUDA 时优先使用 GPU，无 CUDA 时使用 CPU。输出预览图中红线为 YOLO 粗中缝，蓝线为热力图细化线。`predictions.csv` 同时保留两组端点坐标，方便特殊情况使用粗预测。推理入口也支持 `--router`、`--detector`、`--heatmap` 指定其他权重路径。

`config.yaml` 是本次模型的参数记录；当前程序从 Python 源码读取实际参数。修改输入尺寸或归一化参数时，需同步修改 `infer_spine_auto_orientation.py` 与 `spine_line_heatmap_runtime.py`。

## 效果与边界

现有 218 张验证图的平均端点误差由 1.15% 降至 0.61%；13 张横向/斜向补充图由 1.66% 降至 1.28%。热力图依赖 YOLO 先给出接近中缝的粗定位；粗定位漏检或偏差过大时不能保证修正。少数特殊图会退步，目前没有自动回退逻辑。13 张补充图曾用于此前方法选择，不是全新盲测。
