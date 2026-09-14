# Cloud-side Python Inference

本目录是云端文档分割推理交付代码，用于对多类型文档图像进行分割，包括小票、单页书本、屏幕拍摄内容以及其他常见文档。

云端侧使用较大的 DeepLabV3+ 分割模型，主要目标是提高复杂场景下的泛化能力和分割质量。

## 目录内容

```text
Cloud-side/
├── predict.py              # 批量推理入口
├── deeplab.py              # 模型加载、分割、mask/edge 输出逻辑
├── inference_config.py     # 推理路径、模型参数和输出开关
├── best_epoch_weights.pth  # 云端分割模型权重
├── nets/                   # DeepLabV3+ 网络结构
├── utils/                  # 图像预处理和通用工具
├── requirements.txt        # Python 依赖
└── img_out/                # 本地推理输出结果
```

## 与训练目录的关系

云端模型训练代码在项目外层：

```text
Semantic-Segmentation/Cloud_side_train/
```

`Cloud_side_train` 用于训练云端大模型、管理训练数据和生成 `.pth` 权重；本目录只保留推理交付需要的最小代码和权重。

训练完成后，本地运行时将权重放到本目录：

```text
best_epoch_weights.pth
```

该权重文件较大，不提交到 GitHub。

## 输出开关

推理路径、模型参数和输出开关统一在 `inference_config.py` 中配置。`predict.py` 和 `deeplab.py` 都从这里读取默认值：

```python
BACKBONE = "mobilenetv3"  # or "xception"
OUTPUT_TYPE = "mask"
EDGE_WIDTH = 2
MIX_TYPE = 0
```

取值说明：

```text
OUTPUT_TYPE = "mask"  输出文档区域分割 mask
OUTPUT_TYPE = "edge"  输出文档轮廓边缘

EDGE_WIDTH = 2        轮廓线宽，仅 edge 模式使用

MIX_TYPE = 0          原图和 mask 混合显示，仅 mask 模式使用
MIX_TYPE = 1          输出 0-255 黑白 mask/edge
```

当 `OUTPUT_TYPE = "edge"` 时，程序直接输出轮廓边缘图，不使用混合显示，所以 `MIX_TYPE` 不生效。

`BACKBONE` 必须和权重匹配：`best_hd95_epoch_85.pth` 使用 `mobilenetv3`，
`best_epoch_weights.pth` 使用 `xception`。如果组合不匹配，加载阶段会明确报错，避免用错误架构开始推理。

`best_hd95_epoch_85.pth` 对应训练输入为 `1024 × 1024`、`DOWNSAMPLE_FACTOR = 8`。
MobileNetV3 推理骨干与 `Semantic-Segmentation/Cloud_side_train/nets/deeplabv3_plus.py` 中的实现保持一致：
浅层特征取 `bneck[2]`，高层特征按配置转换为 8 或 16 倍下采样，包含残差分支及
5×5 卷积的步长、膨胀率和 padding。以上参数不包含在 checkpoint 中；即使权重
严格加载成功，使用未经转换的 32 倍下采样分类骨干也可能导致预测为空 mask。

## 输入输出路径

默认输入目录：

```text
../TEST_dewarp/
```

默认输出目录：

```text
img_out/
```

`predict.py` 会递归遍历 `../TEST_dewarp/` 下的图片，并将结果保存到 `img_out/`。

## 运行

在 `Cloud-side` 目录下运行：

```bash
python predict.py
```

输出文件统一保存为 `.png`。

## 提交说明

`../TEST_dewarp/`、`img_out/` 和 `best_epoch_weights.pth` 是本地测试数据、输出结果和本地权重文件，不提交到 GitHub。交付时重点保留：

```text
predict.py
deeplab.py
inference_config.py
nets/
utils/
requirements.txt
```

默认输入统一为 `4-dewarp/TEST_dewarp/`；输出中保留 `perspective/`、`instance/` 等子目录。
