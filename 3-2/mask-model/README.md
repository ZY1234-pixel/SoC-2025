# 水印 Mask 检测

输入 **有水印原图** + **去水印候选图**，输出与原图同分辨率的**水印 Mask**。

权重：`weights/watermark_mask.pt`（val IoU 0.7642）。

## 目录结构


```text
<mask-model>/
├── infer.py                # 推理入口（只依赖 models/ 和 weights/）
├── visualize.ipynb         # 可视化：水印图 / 原图 / 水印 mask / overlay
├── models/                 # 网络结构
│   ├── __init__.py
│   ├── network.py
│   └── difference_gate_network.py
├── weights/
│   └── watermark_mask.pt   # 权重，37 MB
├── requirements.txt
└── data/                   # 数据目录（自己放，见下）
```

## 安装

```bash
python -m pip install -r requirements.txt
```

先按目标 CUDA 装好 PyTorch（https://pytorch.org/get-started/locally/ ），
实测版本：torch 2.8.0 + CUDA 12.x。

## 运行推理

```bash
python infer.py \
  --source /path/to/watermarked.jpg \
  --candidate /path/to/clean_candidate.png \
  --output out/ \
  --threshold 0.35
```

输出两个文件：

```text
out/<name>_probability.png   # 16 位概率图，像素值 / 65535 = 概率
out/<name>_mask.png          # 8 位二值 mask，水印=255，背景=0
```

Python 调用：

```python
from infer import load_model, predict
from PIL import Image

model, device, info = load_model("weights/watermark_mask.pt")
source = Image.open("watermarked.jpg").convert("RGB")
candidate = Image.open("candidate.png").convert("RGB")

result = predict(model, source, candidate, device)
mask = result["probability"] >= 0.35
```

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--threshold` | 0.35 | 概率阈值。偏召回，适合「mask 外回退到原图」的融合方式 |
| `--checkpoint` | `weights/watermark_mask.pt` | 权重路径 |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |

推理内部固定做两件事：

- **门控融合**：以主 mask 头为主，只在大块分支自信处（>0.7）融合大块分支。
- **尺度匹配**：原图比候选图大 1.5 倍以上时，把原图缩到候选尺度推理，概率图再放回
  原图分辨率（**输出分辨率始终等于原图**）。不做这一步的话，候选图被硬放大后与
  原图的差异主要来自模糊，实测相机卡片区检出率会从 77.8% 掉到 13.5%。

## 运行可视化

数据目录：在 `visualize.ipynb` 第一个代码块顶部指定，里面放两个子目录：

```text
<data-dir>/
├── source/       # 有水印原图，例如 a.jpg
└── candidate/    # 去水印候选图，同名主干，例如 a.png
```

然后：

```bash
jupyter notebook visualize.ipynb
```

四栏展示：水印图 / 原图 / 水印 mask / overlay。
