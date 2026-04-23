# 图片文本解析及矢量化转换交付包说明

本交付包用于本地验证图片文本解析及矢量化转换的完整恢复链路：

`图片 / PDF -> 版面分析 + OCR -> 标准 JSON -> DOCX / Markdown / PDF`

模型与数据集下载地址：
通过网盘分享的文件：SoC_1-1
链接: https://pan.baidu.com/s/12ouE5owq8Ii_KigQzOeirQ 提取码: 4phe

## 1. 目录结构

```text
DocFlow_FullFlow_Package/
├── dataset/                # 测试输入目录
├── Code/                   # 核心源码、运行测试脚本、模型和运行时
├── test-result/            # 测试输出目录
├── doc/                    # 详细文档
├── README.md               # 本说明
└── release note.txt        # 版本发布说明
```

`Code/` 目录中的关键内容如下：

```text
Code/
├── docflow_src/            # 当前版本核心源码
├── models/                 # 版面/检测/识别/表格模型
├── third_party/            # PaddleOCR 最小运行时
├── wheels/                 # 当前版本 wheel 包
├── test.py                 # 主测试入口
├── dataset.py              # 输入收集脚本
├── preprocess.py           # 图片/PDF 预处理脚本
├── model.py                # 包内路径配置
├── utils.py                # 运行工具函数
├── requirement.txt         # Python 依赖
└── runcmd.txt              # 常用命令清单
```

## 2. 推荐阅读顺序

1. 阅读 `doc/DEPLOYMENT.md`，完成环境安装。
2. 阅读 `doc/TESTING.md`，按统一流程执行测试。
3. 出现异常时查看 `doc/TROUBLESHOOTING.md`。

## 3. 快速开始

### 3.1 Linux

```bash
cd Code
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
python -m pip install wheels/docflow-0.3.0-py3-none-any.whl
python test.py --input ../dataset --output ../test-result --formats docx,markdown
```

### 3.2 Windows PowerShell

```powershell
cd Code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
python -m pip install wheels\docflow-0.3.0-py3-none-any.whl
python test.py --input ..\dataset --output ..\test-result --formats docx,markdown
```

如果需要输出 PDF，请把 `--formats` 改为 `docx,markdown,pdf`，并先安装 LibreOffice / soffice。

## 4. 常用输出

默认输出目录为 `test-result/`。每次运行都会生成一个新的 `run_时间戳/` 目录，结果按样例分层存放：

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

说明：
- 多页 PDF 只生成一个合并后的正式结果文件
- `debug/` 中会按页保存调试可视化图

## 5. 文档索引

- 部署说明：`doc/DEPLOYMENT.md`
- 测试流程：`doc/TESTING.md`
- 故障排查：`doc/TROUBLESHOOTING.md`
- 发布说明：`release note.txt`
