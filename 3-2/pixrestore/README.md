# PixRestore图片去水印模型

## 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## 命令行调用

```bash
python infer.py -i input -o output
```

`input` 可以是图片或目录。默认使用 `cuda:0`、4 步推理、FP16 混合精度、1024 分块。

修复结果保存到 `output/restored`。可用 `--steps` 和 `--precision` 调整步数与精度。

## Python 调用

```python
from restore import WatermarkRemover

model = WatermarkRemover(device="cuda:0")
result = model.restore("input.jpg")
result["restored"].save("result.png")
```

批量调用时复用同一个 `model` 实例。
