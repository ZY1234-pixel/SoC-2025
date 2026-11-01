# ESDNet-Lite：轻量去屏纹模型

目前模型约有 **51.62 万参数**，量化后的权重约 **0.54 MB**。这里交接的是已经导出的 NCNN INT8 版本，使用 CPU 推理。使用时只运行 `infer.py`，它会调用同目录下的模型、分块和融合模块。
## 先跑一张图

建议使用 Linux 和 Python 3.12。在当前目录安装依赖，然后把命令里的图片路径换成自己的文件：

```bash
python -m pip install -r requirements.txt
python infer.py --input input.jpg --output output.png
```

程序会在终端显示处理进度，完成后保存一张与原图尺寸相同的 PNG。内部会把大图分块处理，再融合回完整画面，不需要提前裁图。

默认使用 8 个线程。如果设备核心较少，可以改成 4 个线程：

```bash
python infer.py --input input.jpg --output output.png --threads 4
```

## 目录结构

| 文件 | 用途 |
|---|---|
| `infer.py` | 唯一运行入口，负责读图、组织推理和保存 |
| `model.py` | NCNN模型加载和单块预测 |
| `tiling.py` | 图片补边、分块和重叠融合 |
| `fusion.py` | 接缝校正及保护式回退 |
| `weights/` | 配套的 `.param` 和 `.bin` 模型文件，两者都需要保留 |
| `requirements.txt` | 安装运行依赖 |
| `LICENSE.txt` | 保留的 PDNet 上游许可文本 |

如果后续接入自己的应用，可以先沿用这里的完整流程，确认结果一致后再调整。模型输出已经包含 RGB 残差相加，不需要再和原图叠加。

## 目前的效果与取舍

这个版本优先控制了模型体积，实际处理时间仍会受到图片大小、CPU 和线程设置的影响。它还不能把所有屏纹去干净，部分场景会有彩纹残留、颜色偏差或细节损失。评估时建议同时看整图和局部细节，尤其留意天空、渐变背景和细密纹理。

交接前已在独立目录验证运行，4K 输出与原项目默认流程逐像素一致。目前尚未在目标安卓平板上验证，设备上的速度和内存占用还需要实际测试。

## 来源

模型骨架来自 [ESDNet](https://github.com/CVMI-Lab/UHDM)，轻量化训练和部署整理在本项目中完成；分块融合沿用项目已有工程实现，相关研究参考 [PDNet](https://github.com/ChenJiaCong-1005/PDNet)。
