# Document Dewarp Segmentation Delivery

本目录中包含两套分割推理交付代码：`Cloud-side` 和 `Device-side`以及汤、赵维护的相关代码。这两部分分割代码是当前交付和维护的重点，分别对应云端推理和端侧推理。

除这两部分外，`4-dewarp` 下如果还有其他目录或文件夹更新，不属于当前说明和维护范围。

## 目录说明

```text
4-dewarp/
├── Cloud-side/      # 云端 Python 推理交付
└── Device-side/     # 端侧 C++/NCNN 推理交付
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