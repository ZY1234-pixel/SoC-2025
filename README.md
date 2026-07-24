# 图片文本解析与矢量化转换

图片型文档的版面分析与结构化还原模块。输入扫描图片 / PDF，输出保留原始排版结构的 DOCX、Markdown 与 PDF。

![pipeline](https://img.shields.io/badge/pipeline-OCR--%3ELayout--%3EReconstruction-blue)

## 效果展示

### 学术期刊论文（双栏 + 公式 + 表格）

| 原图 | 版面分析 | 还原输出 |
|:----:|:--------:|:--------:|
| <img src="test-result/.examples/academic-paper/01_original.png" width="320"> | <img src="test-result/.examples/academic-paper/02_layout.png" width="320"> | <img src="test-result/.examples/academic-paper/03_rendered.png" width="320"> |

### 报纸版面（多栏 + 混排）

| 原图 | 版面分析 | 还原输出 |
|:----:|:--------:|:--------:|
| <img src="test-result/.examples/newspaper/01_original.png" width="320"> | <img src="test-result/.examples/newspaper/02_layout.png" width="320"> | <img src="test-result/.examples/newspaper/03_rendered.png" width="320"> |

### 杂志图文（图文混排 + 双栏）

| 原图 | 版面分析 | 还原输出 |
|:----:|:--------:|:--------:|
| <img src="test-result/.examples/magazine/01_original.png" width="320"> | <img src="test-result/.examples/magazine/02_layout.png" width="320"> | <img src="test-result/.examples/magazine/03_rendered.png" width="320"> |

### 中文图书（段落 + 图片）

| 原图 | 版面分析 | 还原输出 |
|:----:|:--------:|:--------:|
| <img src="test-result/.examples/book/01_original.png" width="320"> | <img src="test-result/.examples/book/02_layout.png" width="320"> | <img src="test-result/.examples/book/03_rendered.png" width="320"> |

---

## 快速开始

### 环境准备

```bash
cd Code
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
pip install wheels/vecdoc-0.5.0-py3-none-any.whl
```

### 运行测试

```bash
python test.py -i ../dataset -o ../test-result -f docx,markdown
```

输出会按时间戳归档到 `test-result/run_YYYYMMDD_HHMMSS/` 目录下，每个样本包含 JSON 中间结果、DOCX / Markdown / PDF 输出文件以及 debug 可视化图。

**Windows (PowerShell)** 只需将路径分隔符改为 `\`，虚拟环境激活改为 `.\.venv\Scripts\Activate.ps1`，其余命令相同。

需要 PDF 输出的话，加上 `pdf` 格式并确保系统已安装 LibreOffice。

## 流水线

```
图片 / PDF
  → PP-DocLayoutV3 + PP-OCRv6（保留 Model Order 与原始识别证据）
  → Document Analysis（语义组合、来源追踪、文档级样式角色）
  → Reflow Layout Plan（Single / Sequential Columns / Grid + 单次 Page Fit）
  → 机械生成 DOCX / Markdown
  → LibreOffice 从最终 DOCX 导出 PDF
```

每个源页面严格对应一个输出页面。规划阶段只做一次静态缩放计算，生成后只验收、不重新生成；中间 Evidence、Analysis 和 Plan 均写入 JSON。

## 输出格式

| 格式 | 说明 |
|------|------|
| **DOCX** | 保留字号、对齐、多栏、表格、图片等排版信息，可直接用 Word 编辑 |
| **Markdown** | 适合二次加工，标题层级、表格、图片引用均保留 |
| **PDF** | 通过 LibreOffice 从 DOCX 转换 |

## 目录结构

```
.
├── dataset/              # 测试样本
├── Code/                 # 源码与运行时
│   ├── docflow_src/      # 核心源码
│   ├── models/           # 版面 / 检测 / 识别 / 表格模型
│   ├── third_party/      # PaddleOCR 最小运行时
│   ├── wheels/           # Wheel 包
│   ├── test.py           # 测试入口
│   └── requirement.txt   # Python 依赖
├── test-result/          # 运行输出
├── doc/                  # 详细文档
│   ├── DEPLOYMENT.md     # 部署说明
│   ├── TESTING.md        # 测试流程
│   └── TROUBLESHOOTING.md
└── README.md
```

## 模型与数据

版面检测、OCR 识别、表格结构等模型文件：

- 百度网盘：[SoC_1-1](https://pan.baidu.com/s/12ouE5owq8Ii_KigQzOeirQ) 提取码：`4phe`

解压后放入 `Code/models/` 目录。

## 常见问题

- **模型找不到**：确认 `Code/models/` 下模型目录结构正确，可参考 `doc/DEPLOYMENT.md`
- **PDF 输出失败**：检查 LibreOffice 是否安装且 `soffice` 在 PATH 中
- **版面还原偏差**：debug 目录下的 `*_layout_ocr.jpg` 和 `*_sorted_layout.jpg` 会展示版面检测和排序结果，有助于定位问题

更多细节见 `doc/` 目录。
