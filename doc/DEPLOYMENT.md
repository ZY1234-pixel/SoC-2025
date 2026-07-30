# 部署指南

## 1. 适用范围

本交付包用于本地验证以下能力：

`图片 / PDF -> OCR + 版面分析 -> DocFlow 恢复 -> docx / markdown / pdf`


## 2. 环境要求

- Python 3.9 及以上，推荐 3.9 / 3.10
- Linux 或 Windows
- 若要输出 PDF，需预装 LibreOffice，并保证命令可直接调用
  - Linux: `libreoffice` 或 `soffice`
  - Windows: `soffice.exe`

## 3. 目录确认

进入交付包后，先确认以下目录存在：

- `Code/`
- `dataset/`
- `test-result/`
- `doc/`

## 4. 安装步骤

### 4.1 创建虚拟环境

```bash
python3.9 -m venv .venv
```

### 4.2 Linux 安装命令

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Code/requirement.txt
```

### 4.3 Windows PowerShell 安装命令

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r Code\requirement.txt
```

测试入口会直接加载 `Code/docflow_src/`，无需安装额外 wheel。

### 4.4 模型文件

确认以下文件存在：

```text
Code/models/layout/pp-doclayout-v3/PP-DocLayoutV3.onnx
Code/models/det/ch/PP-OCRv6_small_det/PP-OCRv6_small_det.onnx
Code/models/rec/ch/PP-OCRv6_small_rec/PP-OCRv6_small_rec.onnx
Code/models/table/SLANet_plus/SLANet_plus.onnx
Code/models/font/mobilenetv3.ckpt
```

## 5. 安装完成后的最小验证

```bash
python Code/test.py --input dataset/exam_paper_02.png --output test-result --formats docx,markdown
```

如果需要验证 PDF 导出：

```bash
python Code/test.py --input dataset/exam_paper_02.png --output test-result --formats docx,markdown,pdf
```

## 6. 输出位置说明

输出默认写入 `test-result/run_YYYYMMDD_HHMMSS/`，典型结构如下：

```text
test-result/
└── run_YYYYMMDD_HHMMSS/
    ├── run_manifest.json
    └── <样例名>/
        ├── <样例名>.recognition.json
        ├── <样例名>.json
        ├── <样例名>.render_plan.json
        ├── <样例名>.docx
        ├── <样例名>.md
        ├── <样例名>.pdf
        ├── <样例名>_assets/
        └── debug/
```

## 7. 推荐下一步

部署完成后，请按 `doc/TESTING.md` 执行统一测试流程并做效果验收。
