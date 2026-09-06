# 水印 Mask 检测模型

当前模型是 `DifferenceGateMaskNet`：输入有水印原图和去水印候选图，输出同分辨率的水印概率图与二值 Mask。

## 目录与权重

```text
mask-model/
├── handoff/README.md       # 使用说明
├── handoff/infer.py        # 生产推理：滑动窗口
├── handoff/visualize.ipynb # 可视化
├── models/                 # 模型结构
├── requirements.txt
└── weights/                # 权重单独部署
```

当前模型权重放在 `mask-model/handoff/weights/watermark_mask.pt`，权重文件不提交到代码仓库。

## 安装

建议 Python 3.10，并先安装与目标 CUDA 匹配的 PyTorch/torchvision：

```bash
python -m pip install -r mask-model/requirements.txt
```

始终在项目根目录执行命令。

## 单张图推理

```bash
python mask-model/handoff/infer.py \
  --source /path/to/watermarked.jpg \
  --candidate /path/to/clean_candidate.png \
  --checkpoint mask-model/handoff/weights/watermark_mask.pt \
  --output /path/to/output \
  --tile 512 --overlap 64 --threshold 0.5 --device auto
```

输出文件：

```text
output/*_probability.png  # 16 位概率图，像素值 / 65535 = 概率
output/*_mask.png         # 8 位二值 Mask，水印=255，背景=0
```

候选图尺寸可以与原图不同，脚本会缩放候选图后推理；输出保持原图分辨率。显存不足时将 `--batch-size` 调为 1。

## Python 调用

```python
import sys
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0, str(Path("mask-model").resolve()))
from infer import predict_full_resolution
from models import paired_model_from_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("mask-model/handoff/weights/watermark_mask.pt", map_location="cpu", weights_only=False)
model, architecture = paired_model_from_checkpoint(checkpoint)
model = model.to(device).eval()
with Image.open("watermarked.jpg") as source, Image.open("candidate.png") as candidate:
    probability = predict_full_resolution(model, source, candidate, device, tile=512, overlap=64)
mask = probability >= 0.5
```

## 可视化

打开 `mask-model/handoff/visualize.ipynb`，选择 Python 内核并按顺序运行。Notebook 自动加载 `best.pt`，不存在时使用 `last.pt`，展示：

```text
Watermarked source | Clean candidate | Mask probability | Overlay
```

默认每类测试 3 张；将 `MAX_IMAGES_PER_CATEGORY` 改为 `0` 可运行全部图片。结果保存在 `runs/external_notebook/`。

## 模型说明

- 共享 MobileNetV3-Large 编码器，约 3.17M 参数；
- 多尺度 source/candidate 差异与语义门控；
- RGB 和梯度差异细节分支；
- 训练包含候选图模糊、压缩、局部位移和水印残留增强。

该模型只负责输出水印 Mask，最终修复图由后续去水印模型生成并与原图融合。阈值可在目标数据上校准：漏检多时尝试 `0.35~0.45`，误检多时尝试 `0.6~0.7`。
