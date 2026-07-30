# DocFlow 图片文档结构化还原

DocFlow 将扫描图片或 PDF 转换为可编辑的 DOCX、Markdown 和 PDF。当前分支采用流式排版：保留阅读顺序、分栏、表格、图片和主要样式，并严格约束源页面与输出页面一一对应。

本项目优先保证内容可编辑和文档结构可维护，不以文本框绝对定位的方式逐像素复刻原图。

## 效果展示

以下结果均由当前流式排版流程生成，第三列是 DOCX 经 LibreOffice 渲染后的页面。

### 学术论文：双栏、公式与复杂表格

| 原图 | 版面分析 | DOCX 渲染 |
|:----:|:--------:|:---------:|
| <img src="doc/assets/readme/academic_paper_01/original.jpg" width="320"> | <img src="doc/assets/readme/academic_paper_01/layout.jpg" width="320"> | <img src="doc/assets/readme/academic_paper_01/rendered.jpg" width="320"> |

### 复杂表格：原生可编辑表格与单页约束

| 原图 | 版面分析 | DOCX 渲染 |
|:----:|:--------:|:---------:|
| <img src="doc/assets/readme/docstructbench_01/original.jpg" width="320"> | <img src="doc/assets/readme/docstructbench_01/layout.jpg" width="320"> | <img src="doc/assets/readme/docstructbench_01/rendered.jpg" width="320"> |

### 金融研报：密集双栏、图表与中文样式

| 原图 | 版面分析 | DOCX 渲染 |
|:----:|:--------:|:---------:|
| <img src="doc/assets/readme/eastmoney_02/original.jpg" width="320"> | <img src="doc/assets/readme/eastmoney_02/layout.jpg" width="320"> | <img src="doc/assets/readme/eastmoney_02/rendered.jpg" width="320"> |

### 报纸：四栏、跨栏图片与图注

| 原图 | 版面分析 | DOCX 渲染 |
|:----:|:--------:|:---------:|
| <img src="doc/assets/readme/newspaper_01/original.jpg" width="320"> | <img src="doc/assets/readme/newspaper_01/layout.jpg" width="320"> | <img src="doc/assets/readme/newspaper_01/rendered.jpg" width="320"> |

## 快速开始

### 环境要求

- Python 3.9，支持 Linux 和 Windows
- 完整的 `Code/models/` 模型目录
- 输出 PDF 时需要 LibreOffice，并确保 `libreoffice` 或 `soffice` 位于 `PATH`

### Linux

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Code/requirement.txt
```

### Windows PowerShell

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r Code\requirement.txt
```

代码直接从 `Code/docflow_src/` 加载，不需要额外安装项目 wheel。

### 单样本验证

```bash
python Code/test.py \
  --input dataset/exam_paper_02.png \
  --output test-result \
  --formats docx,markdown
```

### 全量测试

```bash
python Code/test.py \
  --input dataset \
  --output test-result \
  --formats docx,markdown,pdf
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--input`, `-i` | 输入图片、PDF 或目录，默认 `dataset/` |
| `--output`, `-o` | 结果根目录，默认 `test-result/` |
| `--formats`, `-f` | `docx,markdown,pdf` 的任意组合 |
| `--layout-model` | 选择版面分析模型，默认 `pp-doclayout-v3` |
| `--pdf-dpi` | PDF 转图片的 DPI，默认 `200` |
| `--no-debug-vis` | 不生成版面分析可视化图 |

## 处理流程

```text
图片 / PDF
  -> PP-DocLayoutV3 + PP-OCRv6
  -> Recognition Evidence
  -> Document Analysis
  -> Reflow Layout Plan
  -> DOCX / Markdown Renderer
  -> LibreOffice 导出 PDF（可选）
```

- **Recognition Evidence**：保存模型阅读顺序、OCR 行、原始区域和来源标识。
- **Document Analysis**：组合段落、标题、表格、图片、公式及图注，并推断文档级样式角色。
- **Reflow Layout Plan**：在 `single_flow`、`sequential_columns` 和 `grid_flow` 中选择页面结构，统一规划间距与单页缩放。
- **Renderer**：机械执行布局计划，不在渲染阶段重新推断版面。

规划阶段仅执行一次静态页容量计算，不使用“生成 DOCX、检测分页、反复缩放重生成”的闭环。

## 当前能力

- 段落、标题和图注保持可编辑，支持字号、行距、对齐、颜色及字重推断。
- 中文字体识别支持宋体、黑体、楷体和仿宋。
- 表格还原为原生可编辑 Word 表格，并保留跨行、跨列和主要边框样式。
- 支持单栏、多栏、规则网格、跨栏图片、图文混排、公式和页眉页脚。
- 每个源页面对应一个输出页面，页面规划会在必要时进行一次保守缩放。
- Evidence、Analysis 和 Render Plan 均保留为 JSON，便于定位识别、分析或渲染问题。

## 输出目录

每次执行会创建独立的时间戳目录：

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
            ├── page_0001.layout_ocr.jpg
            └── page_0001.reading_order_columns.jpg
```

未请求的输出格式不会生成；`raw_result.json` 和 `debug/` 可通过 `--no-debug-vis` 关闭。

## 模型文件

运行前应确保以下模型存在：

```text
Code/models/
├── layout/pp-doclayout-v3/PP-DocLayoutV3.onnx
├── det/ch/PP-OCRv6_small_det/PP-OCRv6_small_det.onnx
├── rec/ch/PP-OCRv6_small_rec/PP-OCRv6_small_rec.onnx
├── table/SLANet_plus/SLANet_plus.onnx
└── font/mobilenetv3.ckpt
```

模型下载：百度网盘 [SoC_1-1](https://pan.baidu.com/s/12ouE5owq8Ii_KigQzOeirQ)，提取码 `4phe`。解压后放入 `Code/models/`。

## 测试与验收

运行单元测试：

```bash
pytest -q
```

版面质量以渲染结果人工对照原图验收，重点检查：

- 阅读顺序、栏结构和跨栏范围
- 字号、行距、对齐和段落间距
- 图片比例、图注位置和表格完整性
- 输出页数是否与源页面一致

主流程还会自动检查 Evidence 来源完整性、原生表格数量和 PDF 页数。

## 能力边界

- 当前方向是可编辑的流式 DOCX，不是基于绝对定位文本框的像素级 Replica DOCX。
- DOCX 在 Microsoft Word 与 LibreOffice 中可能存在字体度量和分页差异。
- OCR 或版面检测错误会传递到后续结构分析，优先通过 `debug/` 和三阶段 JSON 定位问题。

## 目录结构

```text
.
├── dataset/              # 测试样本
├── Code/
│   ├── docflow_src/      # 分析、规划与渲染源码
│   ├── models/           # 版面、OCR、表格与字体模型
│   ├── third_party/      # PaddleOCR 运行时
│   ├── test.py           # 全流程入口
│   └── requirement.txt   # Python 依赖
├── tests/                # 单元测试
├── test-result/          # 本地运行结果
├── doc/                  # 部署、测试与设计文档
└── README.md
```

详细说明见 [部署指南](doc/DEPLOYMENT.md)、[测试手册](doc/TESTING.md) 和 [故障排查](doc/TROUBLESHOOTING.md)。
