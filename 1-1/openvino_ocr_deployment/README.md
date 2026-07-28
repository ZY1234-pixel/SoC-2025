# PP-OCRv6 OpenVINO CPU 部署说明

## 1. 用途

该目录是独立 OCR 部署包，包含转换后的 OpenVINO 模型、字符字典以及 DET、裁剪、REC、解码的完整调用代码，输出为文字框、识别文字和置信度。

## 2. 方案

| 模块 | 模型 | OpenVINO 执行方式 | 说明 |
| --- | --- | --- | --- |
| DET | PP-OCRv6 small DET | FP32 | 长边上限 960，优先保证检测框完整 |
| REC | PP-OCRv6 small REC | 自动选择精度 | 当前测试 CPU 实际使用 BF16 |

模型权重仍以 FP32 保存在 XML/BIN 中，没有转成 INT8 或 FP16。REC 未强制指定计算精度，由 OpenVINO 根据 CPU 指令能力选择；不支持 BF16 的 CPU 仍可运行，但一般会回退到 FP32，速度需要重新测试。

## 3. 目录结构

```text
openvino_ocr_deployment/
├── models/
│   ├── det/model.xml + model.bin
│   └── rec/model.xml + model.bin + ppocrv6_dict.txt
├── openvino_ocr.py       # 推理和前后处理
├── example.py            # 单张图片示例
├── requirements.txt
└── README.md
```

模型文件合计约 30.0 MiB。

## 4. 安装

本次验证环境为 Python 3.9.25、OpenVINO 2025.3.0。

Windows PowerShell：

```powershell
cd openvino_ocr_deployment
py -3.9 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python example.py "D:\test\page.jpg"
```

Linux：

```bash
cd openvino_ocr_deployment
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python example.py /data/test/page.jpg
```

## 5. 项目调用

```python
from openvino_ocr import OpenVinoOCR

# 进程启动时创建一次，后续页面复用该实例。
ocr = OpenVinoOCR(
    cpu_threads=10,
    rec_batch_size=6,
    score_threshold=0.5,
)

# 正式处理前预热一次，预热结果不计入性能统计。
ocr.warmup("sample.jpg")

result = ocr.predict("page.jpg")
for line in result["lines"]:
    print(line["box"], line["text"], line["score"])
print(result["timings"])
```

`predict()` 可接收图片路径，也可接收 OpenCV BGR `numpy.ndarray`。返回示例：

```python
{
    "lines": [
        {
            "box": [[106, 45], [338, 45], [338, 83], [106, 83]],
            "text": "光学字符识别模块",
            "score": 0.998731
        }
    ],
    "timings": {
        "det_seconds": 0.052314,
        "rec_seconds": 0.611827,
        "total_seconds": 0.681942
    }
}
```

`box` 为原图坐标，顺序是左上、右上、右下、左下。`total_seconds` 包含 DET、裁剪、REC 和解码，不包含磁盘图片读取及模型首次加载。

不要每处理一页就重新创建 `OpenVinoOCR`，否则会重复执行模型编译，耗时和内存都不符合常驻部署的实际情况。

## 6. OmniDocBench 测试结果

### 测试环境

| 项目 | 配置 |
| --- | --- |
| CPU | AMD Ryzen 9 9900X；WSL2 可见 4 核 / 8 线程 |
| 内存 | WSL2 分配 15 GiB |
| 系统 | Ubuntu 24.04.4 LTS / WSL2 |
| Python / OpenVINO | 3.9.25 / 2025.3.0 |
| 推理参数 | `cpu_threads=10`，`rec_batch_size=6` |
| 实际精度 | DET FP32，REC BF16 |

### 测试口径

- 数据集：OmniDocBench。
- 模型只加载一次，并在正式计时前完成预热。
- 每页运行一次完整 OCR，计时包含预处理、DET、裁剪、REC 和解码，不包含模型加载和图片磁盘读取。

### 速度与效果

| 指标 | 结果 |
| --- | ---: |
| 有效页面数 | 972 页 |
| 完整 OCR 平均耗时 | **0.546 s/页** |
| DET 平均耗时 | 0.051 s/页 |
| REC 平均耗时 | 0.466 s/页 |
| 完整 OCR 中位数 | 0.412 s/页 |
| 完整 OCR P95 | 1.390 s/页 |
| 单页最大耗时 | 5.828 s |
| REC 总体识别准确率 | 99.072% |
| 中英混合完全匹配率 | 73.576% |

准确率和中英混合完全匹配率使用 OmniDocBench 标注框裁剪后单独评估 REC；完整 OCR 耗时则包含 DET 和 REC。两类指标的测试范围不同，表中同时列出是为了分别说明速度和识别效果。

`0.546 s/页` 是 OmniDocBench 平均值，不能代替打印驱动整条链路的耗时。P95 和最大值也表明复杂页面会明显慢于平均页面，集成完成后仍需在目标 Windows 机器上测试驱动 IPC、图片传输和 OCR 的端到端时间。

### 页面文本行数分布

这里的“文本行数”取完整 OCR 输出中的 `texts` 数量，即经过 DET、REC 和 `score_threshold=0.5` 过滤后，每页最终保留的文本条目数。

| 每页文本行数 | 页面数 | 页面占比 | 该区间平均 OCR 耗时 |
| ---: | ---: | ---: | ---: |
| 1-30 行 | 271 | 27.88% | 0.212 s/页 |
| 31-60 行 | 230 | 23.66% | 0.405 s/页 |
| 61-100 行 | 187 | 19.24% | 0.563 s/页 |
| 101-200 行 | 186 | 19.14% | 0.804 s/页 |
| 201 行及以上 | 98 | 10.08% | 1.283 s/页 |

### 内存占用

已有内存数据来自处理 30 张业务图片时的 Linux RSS 高水位：

| 阶段 | 累计峰值 RSS |
| --- | ---: |
| 模型加载后 | 165.5 MiB |
| 完成一次预热后 | 673.7 MiB |
| 30 张图片测试全程峰值 | **2009.0 MiB** |
