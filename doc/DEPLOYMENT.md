# 部署说明

这份说明面向第一次在本机运行图片文本解析及矢量化转换模块的开发人员。所有命令都从仓库根目录执行。

## 运行环境

项目按 Python 3.9 开发和测试，可在 Linux 或 Windows 上运行。生成 DOCX 和 Markdown 不依赖桌面软件；需要 PDF 时，再安装 LibreOffice，并让 `libreoffice`、`soffice` 或 `soffice.exe` 可以从命令行调用。

仓库中应当能看到 `Code/`、`dataset/` 和 `doc/`。`test-result/` 不需要提前创建，第一次运行时会自动生成。

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

仓库没有需要单独安装的项目 wheel。运行 `Code/test.py` 时，脚本会把 `Code/docflow_src/` 和打包的 PaddleOCR 运行时加入 Python 路径。

## 放置模型

默认配置需要以下文件：

```text
Code/models/layout/pp-doclayout-v3/PP-DocLayoutV3.onnx
Code/models/det/ch/PP-OCRv6_small_det/PP-OCRv6_small_det.onnx
Code/models/rec/ch/PP-OCRv6_small_rec/PP-OCRv6_small_rec.onnx
Code/models/table/SLANet_plus/SLANet_plus.onnx
Code/models/font/mobilenetv3.ckpt
```

如果缺少运行时目录，还要检查：

```text
Code/third_party/paddle_runtime/ppstructure
Code/third_party/paddle_runtime/ppocr
Code/third_party/paddle_runtime/tools
```

模型下载地址和目录示例见项目 [README](../README.md#准备模型)。

## 跑一个样本

先不要导出 PDF。这样可以把 Python 或模型问题与 LibreOffice 问题分开：

```bash
python Code/test.py --input dataset/exam_paper_02.png --output test-result --formats docx,markdown
```

命令结束时应看到“全部样本处理成功”。结果位于最新的 `test-result/run_YYYYMMDD_HHMMSS/`。

确认 DOCX 正常后，再试 PDF：

```bash
python Code/test.py --input dataset/exam_paper_02.png --output test-result --formats docx,markdown,pdf
```

## 输出目录

一次运行对应一个目录，每个样本再占一个子目录：

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
