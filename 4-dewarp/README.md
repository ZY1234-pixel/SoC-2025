# Document Dewarp Segmentation Delivery

本目录包含文档分割相关的两套推理交付代码：云端推理和端侧推理。两套代码面向的场景、模型大小和输出内容不同。

## 目录说明

```text
4-dewarp/
├── Cloud-side/      # 云端 Python 推理交付
└── Device-side/     # 端侧 C++/NCNN 推理交付
```

开发训练目录位于本目录外侧：

```text
Cloud_side_train/    # 云端大模型训练代码和训练数据目录
Device_side_test/    # 端侧小模型 Python 训练、测试和推理代码
```

## Cloud-side

`Cloud-side` 用于云端文档分割推理，覆盖类型更广，包括小票、单页书本、屏幕拍摄内容以及其他常见文档图像。

这一侧使用较大的 DeepLabV3+ 分割模型，优先保证泛化能力和分割质量。推理结果可以输出为：

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

训练代码不放在交付目录中，云端模型训练使用：

```text
Cloud_side_train/
```

## Device-side

`Device-side` 用于端侧部署，只面向翻开的书本这一类输入。模型更小，适合在设备侧运行。

端侧模型除输出书本区域分割结果外，还会同时输出书本中缝点，用于后续展开、矫正或几何处理。端侧交付代码使用 C++ 和 NCNN 运行。

对应交付目录：

```text
4-dewarp/Device-side/
├── main.cpp
├── deeplabv3p_ncnn.cpp
├── deeplabv3p_ncnn.h
├── CornerLostProcess.cpp
├── run.sh
├── deeplabv3p.ncnn.param
├── img/
└── img_out/
```

端侧模型的 Python 训练、测试和推理代码不放在交付目录中，开发时使用：

```text
Device_side_test/
```

## 训练与交付关系

```text
Cloud_side_train/  -> 训练云端大模型 -> 4-dewarp/Cloud-side/
Device_side_test/  -> 训练和测试端侧小模型 -> 4-dewarp/Device-side/
```

`Cloud_side_train` 和 `Device_side_test` 用于训练、调试和生成模型文件；`4-dewarp/Cloud-side` 和 `4-dewarp/Device-side` 是最终提交和交付的推理代码。

## 数据与输出

测试图片和输出结果仅用于本地验证，通常不需要提交到 GitHub：

```text
img/
img_out/
```

模型权重和 NCNN 依赖需要和对应推理代码放在一起用于本地运行，但不提交到 GitHub：

```text
Device-side/deeplabv3p.ncnn.bin
Device-side/ncnn-20260113/
```

云端 Python 权重本地放置路径：

```text
Cloud-side/best_epoch_weights.pth
```
