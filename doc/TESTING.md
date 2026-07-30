# 测试说明

自动测试用于检查来源完整性、原生表格数量和输出页数。视觉还原质量需要将 DOCX 渲染为 PDF 或图片后，与原图进行人工对照。

以下命令都从仓库根目录执行。先按 [DEPLOYMENT.md](DEPLOYMENT.md) 配好 Python 和模型；测试 PDF 输出时还要安装 LibreOffice。

## 单元测试

代码改动后执行：

```bash
pytest -q
```

## 全流程测试

`Code/test.py` 接受一张图片、一个 PDF 或一个目录：

```bash
python Code/test.py --input <输入路径> --output <输出目录> --formats <格式列表>
```

| 参数 | 用法 |
|------|------|
| `--input`, `-i` | 输入文件或目录，默认读取 `dataset/` |
| `--output`, `-o` | 结果保存目录，默认为 `test-result/` |
| `--formats`, `-f` | `docx`、`markdown`、`pdf`，用逗号连接 |
| `--layout-model` | 版面模型，默认 `pp-doclayout-v3` |
| `--pdf-dpi` | 输入 PDF 转图片时使用的 DPI，默认 `200` |
| `--no-debug-vis` | 不保存版面框和阅读顺序图 |

首次验证环境时使用单个样本：

```bash
python Code/test.py --input dataset/exam_paper_02.png --output test-result --formats docx,markdown
```

改动涉及版面分析、布局规划或 DOCX 渲染时，处理完整数据集：

```bash
python Code/test.py --input dataset --output test-result --formats docx,markdown,pdf
```

PDF 输出会调用 LibreOffice。程序随后检查 PDF 页数；某个样本超过源页数时，本次运行会记为失败。

## 运行摘要与阶段数据

`run_manifest.json` 汇总样本数量、页数、失败列表、原生表格数量和布局类型。批量运行后应先检查该文件；`failures` 非空表示存在执行失败的样本。

每个样本有三份阶段数据：

- `<样本名>.recognition.json`：`pages[].items` 是 OCR 和版面模型留下的原始证据。
- `<样本名>.json`：`pages[].elements` 是合并后的段落、标题、表格和媒体块。
- `<样本名>.render_plan.json`：`pages[].sections` 记录分栏或网格结构，`pages[].fit_scale` 是页面缩放结果。

启用 debug 图后，`debug/page_0001.layout_ocr.jpg` 显示检测框，`debug/page_0001.reading_order_columns.jpg` 显示模型顺序。这两张图用于排查识别遗漏、栏位判断和跨栏范围错误。

## 视觉验收

将 DOCX 转换为 PDF 或图片，并与原图并排检查。建议按照以下顺序进行：

1. 检查页数，确认不存在空白页或内容溢页。
2. 检查栏数、阅读顺序、跨栏图片和页眉页脚。
3. 检查表格行、公式、图片和图注是否完整。
4. 对照字号、行距、对齐、段落间距和字体。

Markdown 主要检查文本顺序、标题层级、表格语法和图片路径。它不负责复刻页面版式。

## 问题记录

发现回归时，应保留原图、执行命令、`run_manifest.json`、对应样本目录和渲染截图。问题描述应包含页面位置和具体现象，例如“第二栏末行被裁掉”，以便稳定复现。
