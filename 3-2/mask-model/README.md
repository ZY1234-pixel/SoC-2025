# 水印 Mask 检测模型

这份代码用来检测图片中的水印区域。输入一张 RGB 图片，模型会输出同尺寸的概率图和二值 Mask。

## 目录说明

```text
mask-model/
├── models/
│   └── network.py            # 模型结构
├── weights/
│   └── watermark_mask.pt     # 推理权重
├── infer_sliding.py              # 滑窗推理脚本
├── requirements.txt             # 依赖版本
├── train.py                     # 训练代码
├── evaluate_slbr_protocol.py    # CLWD 评测
└── visualize_results.ipynb      # 效果可视化
```

## 安装

先进入 `mask-model` 目录，然后创建一个新环境。

venv：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows 的激活命令是：

```powershell
.venv\Scripts\activate
```

也可以用 Conda，环境名字随意：

```bash
conda create -n watermark-mask python=3.10 -y
conda activate watermark-mask
python -m pip install -r requirements.txt
```

`requirements.txt` 写的是已经跑过的版本。如果目标机器的 CUDA 版本不同，先安装与驱动匹配的 PyTorch 和 torchvision，再安装其余依赖。

## 跑一张图

```bash
python infer_sliding.py \
  --input /path/to/image.jpg \
  --output /path/to/output
```

脚本默认读取 `weights/watermark_mask.pt`，不用额外传权重路径。

## 批量处理

`--input` 也可以是一个目录。脚本会递归扫描其中的图片，输出时保留原来的子目录结构。

```bash
python infer_sliding.py \
  --input /path/to/images \
  --output /path/to/output \
  --tile 512 \
  --overlap 64 \
  --batch-size 4 \
  --threshold 0.5
```

没有 GPU 时加上 `--device cpu`。如果机器有多张卡，可以用 `--device cuda:1` 指定其中一张。

## 输出文件

假设输入是 `example.jpg`，输出目录中会有两张图：

- `example_probability.png`：16 位概率图。像素值除以 65535 就是 `0~1` 的水印概率。
- `example_mask.png`：8 位二值 Mask。背景是 0，水印区域是 255。

两张图都保留原始分辨率。分块重叠处用 Hann 窗融合，用来减少接缝。

| 参数 | 默认值 | 怎么调 |
|---|---:|---|
| `--tile` | 512 | 一般不用改，训练时用的也是 512 |
| `--overlap` | 64 | 接缝明显时可以适当加大 |
| `--batch-size` | 4 | 显存不足就调小 |
| `--threshold` | 0.5 | 漏检多可以调低，误检多可以调高 |
| `--device` | auto | 自动选 GPU，也支持 `cpu` 和 `cuda:N` |

## 在 Python 中调用

```python
import torch
from PIL import Image

from infer_sliding import predict_full_resolution
from models import WatermarkMaskNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("weights/watermark_mask.pt", map_location="cpu", weights_only=False)

model = WatermarkMaskNet(pretrained=False).to(device).eval()
model.load_state_dict(state_dict)

with Image.open("example.jpg") as image:
    probability = predict_full_resolution(model, image, device)
```

`probability` 是 `float32` 的 NumPy 数组，形状为 `H x W`，值域是 `0~1`。

## 训练文件

`runs/` 里的 `best.pt` 和 `last.pt` 保留了优化器等训练状态，只在继续训练时需要。上线推理用 `weights/watermark_mask.pt` 就够了。

```bash
python train.py --epochs 20 --output runs/new_experiment
python evaluate_slbr_protocol.py
```
