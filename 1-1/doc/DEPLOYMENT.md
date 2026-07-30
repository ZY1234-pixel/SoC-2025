# 部署说明

本文说明图片文本解析及矢量化转换模块的本地部署方法。所有命令均从仓库根目录执行。

## 运行环境

项目基于 Python 3.9 开发和测试，支持 Linux 与 Windows。生成 DOCX 和 Markdown 不依赖桌面软件；导出 PDF 需要安装 LibreOffice，并确保 `libreoffice`、`soffice` 或 `soffice.exe` 可从命令行调用。

仓库根目录应包含 `Code/`、`dataset/` 和 `doc/`。`test-result/` 无需提前创建，脚本会在首次运行时生成。

## 安装 Python 依赖

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

项目源码无需单独安装为 wheel。运行 `Code/test.py` 时，脚本会将 `Code/docflow_src/` 和仓库内的 PaddleOCR 运行时加入 Python 路径。

## 模型与运行时

默认配置依赖以下文件：

```text
Code/models/layout/pp-doclayout-v3/PP-DocLayoutV3.onnx
Code/models/det/ch/PP-OCRv6_small_det/PP-OCRv6_small_det.onnx
Code/models/rec/ch/PP-OCRv6_small_rec/PP-OCRv6_small_rec.onnx
Code/models/table/SLANet_plus/SLANet_plus.onnx
Code/models/font/mobilenetv3.ckpt
```

PaddleOCR 运行时应包含以下目录：

```text
Code/third_party/paddle_runtime/ppstructure
Code/third_party/paddle_runtime/ppocr
Code/third_party/paddle_runtime/tools
```

模型下载地址和目录示例见项目 [README](../README.md#模型准备)。

## 单样本验证

首次验证建议只生成 DOCX 和 Markdown，以便将 Python、模型问题与 LibreOffice 转换问题分开排查：

```bash
python Code/test.py --input dataset/exam_paper_02.png --output test-result --formats docx,markdown
```

命令成功后，终端会显示“全部样本处理成功”。结果保存在最新的 `test-result/run_YYYYMMDD_HHMMSS/`。

确认基础输出正常后，再验证 PDF 导出：

```bash
python Code/test.py --input dataset/exam_paper_02.png --output test-result --formats docx,markdown,pdf
```

## 输出目录

每次运行创建一个独立目录，各样本结果分别保存在子目录中：

```text
test-result/
└── run_YYYYMMDD_HHMMSS/
    ├── run_manifest.json
    ├── _runtime/
    └── <样本名>/
        ├── raw_result.json
        ├── <样本名>.recognition.json
        ├── <样本名>.json
        ├── <样本名>.render_plan.json
        ├── <样本名>.docx
        ├── <样本名>.md
        ├── <样本名>.pdf
        ├── <样本名>_assets/
        └── debug/
```

文件会随 `--formats` 和 `--no-debug-vis` 的设置增减。批量运行和版面检查方法写在 [TESTING.md](TESTING.md)；启动失败时先查 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。
