# 部署指南

## 1. 适用范围

本目录用于本地验证以下能力：

`图片 / PDF -> OCR + 版面分析 -> DocFlow 恢复 -> docx / markdown / pdf`

当前 GitHub 提交只包含代码与文档，不包含模型权重、测试数据和测试输出。

## 2. 环境要求

- Python 3.9 及以上，推荐 3.9 / 3.10
- Linux 或 Windows
- 若要输出 PDF，需预装 LibreOffice，并保证命令可直接调用
  - Linux: `libreoffice` 或 `soffice`
  - Windows: `soffice.exe`

## 3. 运行前准备

进入本目录后，先确认以下路径存在：

- `Code/`
- `dataset/`
- `test-result/`
- `doc/`

其中：

- `dataset/` 中请放入从百度网盘下载的测试输入
- `Code/models/` 中请放入从百度网盘下载的模型文件

再进入代码目录：

```bash
cd Code
```

## 4. 安装步骤

### 4.1 创建虚拟环境

```bash
python -m venv .venv
```

### 4.2 Linux 安装命令

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

### 4.3 Windows PowerShell 安装命令

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

说明：

- 当前仓库不包含 `Code/wheels/`，因此不再执行 wheel 安装
- `test.py` 会直接将 `Code/docflow_src/` 加入运行时路径

## 5. 安装完成后的最小验证

在 `Code/` 目录执行：

```bash
python test.py --input ../dataset/<样例文件> --output ../test-result --formats docx,markdown
```

如果需要验证 PDF 导出：

```bash
python test.py --input ../dataset/<样例文件> --output ../test-result --formats docx,markdown,pdf
```

## 6. 输出位置说明

输出默认写入 `../test-result/run_YYYYMMDD_HHMMSS/`，典型结构如下：

```text
test-result/
└── run_YYYYMMDD_HHMMSS/
    ├── run_manifest.json
    └── samples/
        └── <样例名>/
            ├── <样例名>.json
            ├── <样例名>.docx
            ├── <样例名>.md
            ├── <样例名>.pdf
            ├── <样例名>_assets/
            └── debug/
```

## 7. 推荐下一步

部署完成后，请按 `doc/TESTING.md` 执行统一测试流程并做效果验收。
