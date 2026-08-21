# 部署说明

本文说明开发人员拿到 Git 仓库后的环境配置、模型放置和最小验收方法。项目不依赖开发者本机的 Conda 环境，以下命令均从仓库根目录执行。

## 运行环境

- Python 3.9
- Linux 或 Windows x86-64
- CPU 推理
- OpenVINO 2025.3
- 导出 PDF 时额外安装 LibreOffice

REC 在 CPU 支持 BF16 时使用 BF16/FP32 混合执行；不支持时自动回退 FP32。

## 安装依赖

Linux：

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Code/requirement.txt
```

Windows PowerShell：

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r Code\requirement.txt
```

`Code/test.py` 会直接加载仓库内的 `Code/docflow_src/` 和 PaddleOCR 前后处理代码，无需安装项目 wheel。

## 放置模型

模型不随 Git 仓库提交。

下载地址：**待项目负责人补充**

下载后保持以下目录结构：

```text
Code/models_openvino/
├── PP-DocLayoutV3_openvino/
│   ├── PP-DocLayoutV3.xml
│   └── PP-DocLayoutV3.bin
├── PP-OCRv6_small_det_openvino/
│   ├── PP-OCRv6_small_det_openvino_fp32.xml
│   └── PP-OCRv6_small_det_openvino_fp32.bin
├── PP-OCRv6_small_rec_openvino/
│   ├── PP-OCRv6_small_rec_openvino_fp32.xml
│   ├── PP-OCRv6_small_rec_openvino_fp32.bin
│   ├── ppocrv6_dict.txt
│   └── ppocrv6_rapidocr_dict.txt
├── RapidAI_TableRec_openvino/
│   ├── wired_table_v2/unet.xml + unet.bin
│   ├── lineless_table/lore_detect.xml + lore_detect.bin
│   ├── lineless_table/lore_process.xml + lore_process.bin
│   ├── table_cls/yolo_cls.xml + yolo_cls.bin
│   └── ocr_cls/ch_ppocr_mobile_v2.0_cls_infer.xml + .bin
└── font_openvino/
    ├── mobilenetv3.xml
    └── mobilenetv3.bin
```

RapidAI 表格桥接和融合代码已经包含在仓库内，不需要另外下载 `TableRec` 工程。

## 最小验收

仓库只提交一张验收样例：

```bash
python Code/test.py \
  --input dataset/exam_paper_02.png \
  --output test-result \
  --formats markdown \
  --no-debug-vis
```

命令退出码为 `0`、终端显示“全部样本处理成功”，并在 `test-result/run_*/` 生成 Markdown 和三阶段 JSON，即表示基础环境、模型和调用路径正常。

需要验证 DOCX 和 PDF 时执行：

```bash
python Code/test.py \
  --input dataset/exam_paper_02.png \
  --output test-result \
  --formats docx,markdown,pdf
```

PDF 输出依赖 `libreoffice` 或 `soffice` 命令。

## 集成入口

- 统一模型调用：[Code/model_integration/README.md](../Code/model_integration/README.md)
- 可复用运行入口：`Code/model_integration/runtime.py`
- 模型路径：`Code/model.py`
- OpenVINO 调用封装：`Code/docflow_src/docflow/inference/openvino_session.py`
- RapidAI 表格调用：`Code/docflow_src/docflow/adapters/rapidai_table_adapter.py`
- 模型初始化：`Code/model_integration/runtime.py` 中的 `make_engine()`
- 完整流程组装：`Code/test.py` 中的 `main()`

## 参考测试

以下数据只用于说明测试环境，不是所有电脑的固定结果：

- AMD Ryzen 9 9900X，12 核 24 线程，16 GB 内存
- Python 3.9.25，OpenVINO 2025.3
- OmniDocBench OCR 平均耗时：`0.855 s/页`，包含 DET、裁剪、REC 和解码
- 普通文档主进程峰值内存约 `2.84 GiB`
- 包含 5 个表格的复杂页面主进程峰值内存约 `3.70 GiB`

输入分辨率、文本行数、表格数量和 CPU 指令集都会影响实际结果。
