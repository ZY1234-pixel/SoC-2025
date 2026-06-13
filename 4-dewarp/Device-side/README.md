# Device-side NCNN Inference

本目录是端侧部署推理交付代码，用于翻开书本这一类输入图像。

端侧模型使用较小的 DeepLabV3+ 网络，并通过 NCNN 在设备侧运行。相比云端模型，端侧模型覆盖的输入类型更窄，但运行更轻量，适合部署到本地设备。

端侧推理输出包括：

```text
1. 翻开书本区域的分割结果
2. 书本中缝点结果，用于后续展开、矫正或几何处理
```

## 目录内容

```text
Device-side/
├── main.cpp                  # C++ 程序入口，负责读图、批量处理和保存结果
├── deeplabv3p_ncnn.h          # 模型路径、输入输出路径和推理类声明
├── deeplabv3p_ncnn.cpp        # NCNN 推理、后处理和输出逻辑
├── CornerLostProcess.cpp      # 缺角补绘相关处理
├── run.sh                    # 编译并运行脚本
├── deeplabv3p.ncnn.param      # NCNN 模型结构
├── img/                       # 本地测试输入图片
└── img_out/                   # 本地推理输出结果
```

本地运行时还需要在本目录放置：

```text
deeplabv3p.ncnn.bin
ncnn-20260113/
```

其中 `deeplabv3p.ncnn.bin` 是模型权重，`ncnn-20260113/` 是本地 NCNN 依赖目录，二者不提交到 GitHub。

## 与训练目录的关系

端侧模型的 Python 训练、测试和推理代码在项目外层：

```text
Device_side_test/
```

`Device_side_test` 用于训练端侧小模型、验证 Python 推理结果、测试中缝点输出，并生成或辅助转换端侧部署所需的模型文件。

本目录只保留最终端侧部署需要的 C++/NCNN 推理代码和 NCNN 模型文件。

## 与云端模型的区别

```text
Cloud-side:
  面向小票、单页书本、屏幕等多种文档类型
  使用较大模型
  云端 Python 推理
  主要输出文档分割 mask 或边缘

Device-side:
  只面向翻开的书本
  使用较小模型
  端侧 C++/NCNN 推理
  同时输出书本分割结果和中缝点
```

## 路径配置

主要路径在 `deeplabv3p_ncnn.h` 顶部配置：

```cpp
static constexpr const char* kParamPath = "deeplabv3p.ncnn.param";
static constexpr const char* kBinPath = "deeplabv3p.ncnn.bin";
static constexpr const char* kDefaultInputPath = "img/";
static constexpr const char* kDefaultSavePath = "img_out/";
```

默认从 `img/` 读取图片，将推理结果保存到 `img_out/`。

## 输出模式

输出模式由 `deeplabv3p_ncnn.h` 中的 `output_type` 控制：

```cpp
int output_type = 0;
```

取值说明：

```text
0: 输出原图和分割结果的混合可视化图
1: 输出 0/255 分割 mask
```

缺角补绘由 `enable_corner_lost_process` 控制：

```cpp
bool enable_corner_lost_process = false;
```

取值说明：

```text
true:  开启缺角补绘，补绘生效时额外保存 *_filled.png
false: 关闭缺角补绘
```

## 编译并运行

在 `Device-side` 目录下运行：

```bash
bash run.sh
```

`run.sh` 会先调用 `g++` 编译生成 `main`，然后执行推理。

也可以直接运行已经编译好的程序：

```bash
./main
```

处理单张图片：

```bash
./main /path/to/input.jpg /path/to/output.png
```

处理一个目录：

```bash
./main /path/to/input_dir /path/to/output_dir
```

## 提交说明

`img/`、`img_out/`、`deeplabv3p.ncnn.bin` 和 `ncnn-20260113/` 是本地测试数据、输出结果、模型权重和本地依赖，不提交到 GitHub。交付时重点保留：

```text
main.cpp
deeplabv3p_ncnn.cpp
deeplabv3p_ncnn.h
CornerLostProcess.cpp
run.sh
deeplabv3p.ncnn.param
```
