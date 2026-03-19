# 1-1 图片文本解析及矢量化转换

本目录用于提交该项目的代码与文档，不提交模型权重、数据集、测试输出和 wheel 包。

当前提交内容以推理与测试链路为主：

`图片 / PDF -> 版面分析 + OCR -> 标准 JSON -> DOCX / Markdown / PDF`

说明：

- `Code/test.py` 是主测试入口
- `Code/train.py` 目前是占位入口，不包含实际训练能力
- `Code/docflow_src/` 包含当前版本 DocFlow 核心源码
- `Code/third_party/paddle_runtime/` 包含运行所需的最小 Paddle 运行时代码

## 目录结构

```text
1-1/
├── Code/                   # 核心源码、测试脚本、运行时
├── dataset/                # 仅保留说明文件，真实测试数据请从百度网盘下载
├── test-result/            # 仅保留说明文件，真实测试输出请从百度网盘下载
├── doc/                    # 部署、测试、排障文档
├── README.md               # 本说明
└── release note.txt        # 版本说明
```

## 百度网盘资源

通过网盘分享的文件：SoC_1-1
链接: https://pan.baidu.com/s/1ywOGqQrG7lp1hEehasXJ1Q?pwd=y87x 提取码: y87x

下载后请按以下位置放置：

- 测试数据放到 `dataset/`
- 模型权重放到 `Code/models/`

## 快速开始

### Linux

```bash
cd Code
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
python test.py --input ../dataset --output ../test-result --formats docx,markdown
```

### Windows PowerShell

```powershell
cd Code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
python test.py --input ..\dataset --output ..\test-result --formats docx,markdown
```

如果需要输出 PDF，请把 `--formats` 改为 `docx,markdown,pdf`，并先安装 LibreOffice / soffice。