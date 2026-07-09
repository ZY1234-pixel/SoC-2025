# Document Dewarp Segmentation Delivery

本目录中包含四套分割推理交付代码：`Cloud-side`、`Test_InternImage-L`、`Device-side` 和 `instance-seg`，以及汤、赵维护的相关代码。这几部分分割代码是当前交付和维护的重点，分别对应云端 DeepLab 推理、云端 InternImage-L 推理、端侧推理和 YOLO 实例分割推理。

除这四部分外，`4-dewarp` 下如果还有其他目录或文件夹更新，不属于当前说明和维护范围。

## 目录说明

```text
4-dewarp/
├── Cloud-side/          # 云端 Python DeepLab 语义分割推理交付
├── Test_InternImage-L/  # 云端 Python InternImage-L 文档分割推理交付
├── Device-side/         # 端侧 C++/NCNN 推理交付
└── instance-seg/        # YOLO 实例分割推理交付
```

## Cloud-side

`Cloud-side` 用于云端文档分割推理，覆盖类型更广，包括小票、单页书本、屏幕拍摄内容以及其他常见文档图像。

这一侧使用较大的 DeepLabV3+ 分割模型，优先保证泛化能力和分割质量。它的定位是：

```text
云端运行
Python 推理
更关注泛化能力和分割质量
```

推理结果可以输出为：

```text
mask    # 0-255 文档区域分割图
edge    # 文档轮廓边缘图
mix     # 原图和 mask 的混合可视化，仅 mask 模式使用
```

对应交付目录：

```text
4-dewarp/Cloud-side/
├── predict.py
├── deeplab.py
├── best_epoch_weights.pth
├── nets/
├── utils/
├── img/
└── img_out/
```

## Test_InternImage-L

`Test_InternImage-L` 用于云端 InternImage-L 文档前景分割推理。它是新的服务器 GPU 推理包，和训练侧保持一致：输入图像先按比例 letterbox 到 `1024x1024`，RGB 归一化到 `[0, 1]`，模型输出单通道 logit，经 sigmoid 后按默认阈值 `0.60` 二值化。

这一侧的定位是：

```text
云端运行
Python / PyTorch / MMCV / MMSeg 推理
InternImage-L + UPerHead
输入统一为 1024x1024 letterbox
输出文档前景 mask
更关注文档边界和几何形状质量
```

对应交付目录：

```text
4-dewarp/Test_InternImage-L/
├── infer.py
├── configs/
│   └── docseg_internimage_l_1024.py
├── checkpoints/
│   └── best_hd95_epoch_45.pth
├── segmentation/
│   ├── mmseg_custom/
│   └── ops_dcnv3/
├── img/
└── README.md
```

推理方式：

```bash
cd 4-dewarp/Test_InternImage-L
python infer.py
```

默认会递归读取 `img/` 下的图片，包括 `img/book/` 和 `img/comic/`，并自动使用 `checkpoints/` 下最新的 `best_hd95_epoch_*.pth` 权重。

推理结果输出到：

```text
outputs/
├── masks/       # 放回原图尺寸的 0-255 mask
├── masks_1024/  # 1024 letterbox 坐标下的 mask
└── overlays/    # 原图叠加 mask 的可视化
```

提交到 GitHub 时，`infer.py`、`configs/`、`segmentation/` 和 `README.md` 是推理必需文件；`img/` 是示例输入，`outputs/` 是本地推理结果，通常不需要提交。权重文件较大，如果不随仓库提交，需要单独提供并放到 `checkpoints/` 下。

## Instance-seg

`instance-seg` 用于 YOLO 实例分割推理，定位是从图像中检测并分割文档类目标实例。和 `Cloud-side` 的二分类语义分割不同，实例分割可以在同一张图里输出多条预测，每条预测对应一个文档实例。

当前默认权重为训练侧导出的 `best.pt`，推理目录只保留交付所需文件：

```text
4-dewarp/instance-seg/
├── predict.py
├── weights/
│   └── best.pt
├── img/
└── img_out/
```

推理方式：

```bash
cd 4-dewarp/instance-seg
python predict.py
```

## Device-side

`Device-side` 用于端侧部署，只面向翻开的书本这一类输入。模型更小，适合在设备侧运行。

端侧模型除输出书本区域分割结果外，还会同时输出书本中缝点，用于后续展开、矫正或几何处理。端侧交付代码使用 C++ 和 NCNN 运行。它的定位是：

```text
端侧运行
C++ / NCNN 推理
输出书本 mask 和关键点
更关注部署和运行稳定性
```

对应交付目录：

```text
4-dewarp/Device-side/
├── main.cpp
├── deeplabv3p_ncnn.cpp
├── deeplabv3p_ncnn.h
├── CornerLostProcess.cpp
├── run.sh
├── deeplabv3p.fp32.ncnn.param
├── deeplabv3p.fp32.ncnn.bin
├── img/
└── img_out/
```
