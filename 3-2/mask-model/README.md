# 配对水印 Mask 检测

该模型根据含水印原图和去水印候选图，逐像素预测水印区域。输入图像应使用同一坐标系；候选图由去水印模型生成。

## 命令行

先安装与目标硬件匹配的 PyTorch / torchvision，再安装 `requirements.txt` 中的依赖。CUDA 部署使用对应 CUDA 版本的 PyTorch / torchvision。

CUDA 或 CPU 自动选择：

```bash
python infer.py \
  --source /path/to/watermarked.png \
  --candidate /path/to/clean_candidate.png \
  --output /path/to/output
```

Ascend NPU：

```bash
PIXRESTORE_DEVICE=npu python infer.py \
  --source /path/to/watermarked.png \
  --candidate /path/to/clean_candidate.png \
  --output /path/to/output
```

输出目录包含 16 位概率图 `*_probability.png` 和二值 `*_mask.png`。概率图像素值除以 65535 即为概率。默认阈值为 0.35；推理不做形态学闭合、填洞或膨胀。

## Python 调用

```python
from PIL import Image
from infer import load_model, predict

model, device, info = load_model()
with Image.open("watermarked.png") as f:
    source = f.convert("RGB")
with Image.open("clean_candidate.png") as f:
    candidate = f.convert("RGB")
result = predict(model, source, candidate, device)
probability = result["probability"]
mask = probability >= 0.35
```

## 可视化

打开 `visualize.ipynb` 并运行单元。默认读取项目的 `test_data` 配对清单；也可以修改 `DATA_DIR` 指向 JSONL 清单，或包含 `source/` 与 `candidate/` 子目录的数据目录。notebook 展示原图、候选图、Mask 和覆盖图。
