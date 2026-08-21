# 版面分析、OCR 和表格识别调用说明

目录内容：

```text
model_integration/
├── __init__.py    # 对外导出 OpenVINOModelRuntime
├── runtime.py     # 模型加载及四类推理接口
└── README.md      # 本文档
```

## 输入约定

- 输入类型：`numpy.ndarray`
- 颜色顺序：OpenCV BGR
- 图片维度：灰度二维数组或三通道图片
- 模型设备：CPU
- 同一个 `OpenVINOModelRuntime` 应复用，不要每处理一张图片就重新创建

模型文件应先按 [部署说明](../../doc/DEPLOYMENT.md#放置模型) 放入 `Code/models_openvino/`。

## 初始化

在仓库根目录启动程序时，可以直接导入统一运行接口：

```python
from Code.model_integration import OpenVINOModelRuntime

runtime = OpenVINOModelRuntime()
```

初始化时会加载 PP-DocLayoutV3、PP-OCRv6 Small DET 和 REC。表格模型在首次调用 `run_table()`，或整页检测到表格时再加载。

构造参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `layout_model_name` | `pp-doclayout-v3` | 版面模型名称，生产环境保持默认即可 |
| `table_engine` | `auto` | 自动选择有线或无线表格模型 |
| `full_page_table_fallback` | `False` | 未检测到表格时是否将整页作为表格识别 |
| `runtime_dir` | 系统临时目录 | 保存运行时字典和表格调试文件 |

## 1. 版面分析

```python
import cv2

image = cv2.imread("document.png")
result = runtime.run_layout(image)

regions = result["regions"]
elapsed = result["elapsed"]
```

`regions` 中每一项包含版面类别、坐标和置信度，例如：

```python
{
    "label": "text",
    "bbox": [120, 240, 1320, 680],
    "score": 0.96,
}
```

常见类别包括 `text`、`paragraph_title`、`table`、`image`、`header` 和 `footer`。

## 2. OCR

```python
result = runtime.run_ocr(image)

lines = result["lines"]
timing = result["timing"]
```

该接口执行完整 OCR 流程：DET 文本框检测、文本裁剪、REC 识别和字符解码。行级结果格式为：

```python
{
    "text_region": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "text": "识别文本",
    "confidence": 0.98,
}
```

`timing` 中的 `det` 和 `rec` 分别记录文本检测与识别耗时。

## 3. 表格识别

`run_table()` 接收一张已经裁剪好的表格图片：

```python
table_image = cv2.imread("table_crop.png")
result = runtime.run_table(table_image)
```

默认先用 Table CLS 判断有线表格或无线表格，再选择 UNet 或 LORE。也可以在调试时明确指定：

```python
wired = runtime.run_table(table_image, table_engine="wired_table_v2")
lineless = runtime.run_table(table_image, table_engine="lineless_table")
```

主要返回字段：

| 字段 | 含义 |
|---|---|
| `status` | `ok` 表示结构识别成功 |
| `table_type` | 实际使用的表格模型 |
| `html` | 表格 HTML |
| `bbox` | 单元格坐标 |
| `logic_points` | 单元格行列及跨行跨列关系 |
| `ocr_result` | 表格内部 OCR 行 |
| `elapsed_ms` | 表格完整流程耗时 |

## 4. 整页调用

需要同时执行版面分析、OCR 和表格识别时使用：

```python
result = runtime.run_document(
    image,
    page_index=0,
    table_output_dir="test-result/model-runtime",
)

regions = result["regions"]
timing = result["timing"]
```

流程如下：

1. PP-DocLayoutV3 检出版面区域。
2. PP-OCRv6 DET 和 REC 对整页文字进行识别。
3. 对 `table` 区域执行 RapidAI 表格结构识别。
4. 将表格 HTML、单元格结构和 OCR 结果写回对应区域。

`timing` 保留 `layout`、`det`、`rec`，并增加：

- `rapidai_table`：RapidAI 表格阶段耗时。
- `all_with_table`：从整页输入到最终区域结果的总耗时。

## 模型与调用关系

| 接口 | 使用模型 |
|---|---|
| `run_layout()` | PP-DocLayoutV3 |
| `run_ocr()` | PP-OCRv6 Small DET + REC |
| `run_table()` | Table CLS + UNet 或 LORE + PP-OCRv6 DET/REC + OCR 方向分类器 |
| `run_document()` | 上述模型按页面内容组合执行 |

## 集成注意事项

- `runtime` 创建后应长期复用，模型加载时间不应计入单页推理时间。
- 同一个实例不要被多个线程同时调用；多线程服务建议每个工作线程创建一个实例。
- `run_table()` 输入应尽量只包含表格区域；整页文档使用 `run_document()`。
- 返回区域中包含用于后续文档还原的图片数组，不建议直接对整个返回值执行 `json.dumps()`。
- 模型缺失时初始化会直接报告缺失路径，按部署文档补齐文件即可。
