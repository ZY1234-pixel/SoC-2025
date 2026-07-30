# 图片文本解析及矢量化转换

图片文本解析及矢量化转换模块读取扫描图片或 PDF，经过 OCR、版面分析和布局规划，输出 DOCX、Markdown，也可以调用 LibreOffice 导出 PDF。

这个分支使用 Word 原生的段落、分栏和表格来组织页面。生成的文字和表格可以继续编辑，版面则尽量贴近原图。它不是按 bbox 放置文本框的绝对定位方案，所以不同版本的 Word 或 LibreOffice 仍可能出现细小的字体度量差异。

## 还原效果

下面四组结果由当前代码生成。第三列是 DOCX 经 LibreOffice 转出的页面，不是直接拼接的图片。

### 学术论文：双栏、公式和复杂表格

| 原图 | 版面分析 | DOCX 渲染 |
|:----:|:--------:|:---------:|
| <img src="doc/assets/readme/academic_paper_01/original.jpg" width="320"> | <img src="doc/assets/readme/academic_paper_01/layout.jpg" width="320"> | <img src="doc/assets/readme/academic_paper_01/rendered.jpg" width="320"> |

### 连续表格：五张可编辑表格保持在同一页

| 原图 | 版面分析 | DOCX 渲染 |
|:----:|:--------:|:---------:|
| <img src="doc/assets/readme/docstructbench_01/original.jpg" width="320"> | <img src="doc/assets/readme/docstructbench_01/layout.jpg" width="320"> | <img src="doc/assets/readme/docstructbench_01/rendered.jpg" width="320"> |

### 金融研报：密集双栏、图表和中文样式

| 原图 | 版面分析 | DOCX 渲染 |
|:----:|:--------:|:---------:|
| <img src="doc/assets/readme/eastmoney_02/original.jpg" width="320"> | <img src="doc/assets/readme/eastmoney_02/layout.jpg" width="320"> | <img src="doc/assets/readme/eastmoney_02/rendered.jpg" width="320"> |

### 报纸：四栏、跨栏图片和两行图注

| 原图 | 版面分析 | DOCX 渲染 |
|:----:|:--------:|:---------:|
| <img src="doc/assets/readme/newspaper_01/original.jpg" width="320"> | <img src="doc/assets/readme/newspaper_01/layout.jpg" width="320"> | <img src="doc/assets/readme/newspaper_01/rendered.jpg" width="320"> |

## 开始使用

下面的命令都在仓库根目录执行。项目按 Python 3.9 开发和测试；只有导出 PDF 时才需要安装 LibreOffice。

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

`Code/test.py` 会直接加载 `Code/docflow_src/`，不用另装项目 wheel。

先跑一个样本，确认模型和依赖都能正常加载：

```bash
python Code/test.py --input dataset/exam_paper_02.png --output test-result --formats docx,markdown
```

处理整个 `dataset/`，并把 DOCX 转成 PDF：

```bash
python Code/test.py --input dataset --output test-result --formats docx,markdown,pdf
```

`test.py` 常用参数如下：

| 参数 | 用法 |
|------|------|
| `--input`, `-i` | 图片、PDF 或目录，默认读取 `dataset/` |
| `--output`, `-o` | 结果保存目录，默认为 `test-result/` |
| `--formats`, `-f` | 从 `docx,markdown,pdf` 中选择一种或多种格式 |
| `--layout-model` | 切换版面模型，默认使用 `pp-doclayout-v3` |
| `--pdf-dpi` | 输入 PDF 转图片时使用的 DPI，默认 `200` |
| `--no-debug-vis` | 关闭版面框和阅读顺序图 |

## 代码怎么处理一页文档

```text
图片 / PDF
  -> PP-DocLayoutV3 + PP-OCRv6
  -> Recognition Evidence
  -> Document Analysis
  -> Reflow Layout Plan
  -> DOCX / Markdown
  -> PDF（可选）
```

| 阶段 | 负责什么 | 保存结果 |
|------|----------|----------|
| Recognition Evidence | 接收模型输出，保留 OCR 行、区域、模型顺序和来源标识 | `<样本名>.recognition.json` |
| Document Analysis | 合并段落、标题、表格、图片、公式和图注，归纳文档内的样式 | `<样本名>.json` |
| Reflow Layout Plan | 决定分栏结构、元素间距、尺寸和页内缩放 | `<样本名>.render_plan.json` |
| Renderer | 按计划写入 DOCX 或 Markdown，不再猜测版面 | DOCX / Markdown |

布局计划有三种基本结构：普通单栏使用 `single_flow`，连续多栏使用 `sequential_columns`，图文混排或跨栏页面使用 `grid_flow`。

每张源页面对应一张输出页面。规划器在写 DOCX 之前估算一次页面容量，必要时统一缩放；它不会反复生成 DOCX 再检测分页。

## 目前能还原什么

正文、标题和图注会写成可编辑段落。系统会推断字号、行距、对齐、颜色和字重，中文字体分类包括宋体、黑体、楷体和仿宋。

检测到的表格会写成 Word 原生表格，跨行、跨列和主要边框仍可编辑。图片和公式作为媒体元素嵌入页面，布局规划支持单栏、多栏、跨栏图片、图文混排及页眉页脚。

三阶段 JSON 都带有来源关系。遇到错栏、漏块或样式异常时，可以判断问题出在模型识别、文档分析还是最终布局。

## 输出内容

每次运行都会新建一个时间戳目录。同名样本的结果放在自己的子目录里：

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

只会生成 `--formats` 中指定的文件。使用 `--no-debug-vis` 后，`raw_result.json` 和 `debug/` 不再输出。

## 准备模型

默认流程会读取这些文件：

```text
Code/models/
├── layout/pp-doclayout-v3/PP-DocLayoutV3.onnx
├── det/ch/PP-OCRv6_small_det/PP-OCRv6_small_det.onnx
├── rec/ch/PP-OCRv6_small_rec/PP-OCRv6_small_rec.onnx
├── table/SLANet_plus/SLANet_plus.onnx
└── font/mobilenetv3.ckpt
```

模型可以从百度网盘 [SoC_1-1](https://pan.baidu.com/s/12ouE5owq8Ii_KigQzOeirQ) 下载，提取码为 `4phe`。解压后保持上面的目录结构。

## 测试和版面检查

单元测试在仓库根目录运行：

```bash
pytest -q
```

自动检查会拦截来源丢失、表格退化和 PDF 超页。版面是否接近原图仍要看渲染结果，通常按下面的顺序检查：

1. 输出页数是否与输入一致。
2. 阅读顺序、栏数和跨栏范围是否正确。
3. 表格是否完整，图片比例和图注位置是否合理。
4. 字号、行距、对齐和段落间距是否接近原图。

更具体的命令和检查方法见 [部署说明](doc/DEPLOYMENT.md)、[测试说明](doc/TESTING.md) 和 [常见问题](doc/TROUBLESHOOTING.md)。

## 已知限制

- 这是可编辑的流式 DOCX 方案，不是按原始 bbox 放置文本框的 Replica DOCX。
- Microsoft Word 和 LibreOffice 使用的字体度量并不完全相同，同一份 DOCX 可能有少量换行或分页差异。
- OCR 和版面模型如果漏检或识别错误，后面的分析阶段不会凭空补回内容。此时应先看 `debug/` 和三阶段 JSON。

## 仓库结构

```text
.
├── dataset/              # 测试样本
├── Code/
│   ├── docflow_src/      # 分析、规划和渲染代码
│   ├── models/           # 版面、OCR、表格和字体模型
│   ├── third_party/      # PaddleOCR 运行时
│   ├── test.py           # 全流程入口
│   └── requirement.txt   # Python 依赖
├── tests/                # 单元测试
├── test-result/          # 本地运行结果
├── doc/                  # 部署、测试和排障文档
└── README.md
```
