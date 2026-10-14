# 测试说明

自动测试用于检查来源完整性、原生表格数量和输出页数。视觉还原质量需要将 DOCX 或 PDF 与原图进行人工对照。

以下命令都从仓库根目录执行。先按 [DEPLOYMENT.md](DEPLOYMENT.md) 配好 Python 和模型。

## 单元测试

代码改动后执行：

```bash
pytest -q tests
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
| `--formats`, `-f` | `docx`、`markdown`、`pdf`、`pdf_fidelity`，用逗号连接 |
| `--no-formula` | 关闭公式识别，跳过 PP-FormulaNet |
| `--pdf-dpi` | 输入 PDF 转图片时使用的 DPI，默认 `200` |
| `--save-stages` | 额外保存 recognition、analysis 和 render plan JSON |
| `--save-debug` | 额外保存版面调试图和原始模型结果 |

首次验证环境时使用单个样本：

```bash
python Code/test.py --input dataset/exam_paper_02.png --output test-result --formats docx,markdown
```

改动涉及版面分析、布局规划或 DOCX 渲染时，处理完整数据集：

```bash
python Code/test.py --input dataset --output test-result --formats docx,markdown,pdf
```

PDF 输出由 PyMuPDF 直接按检测框坐标生成，源页面与输出页面一一对应。

高保真 PDF：

```bash
python Code/test.py --input dataset --output test-result --formats pdf_fidelity
```

`pdf_fidelity` 保留原始页面图像作为视觉底图，并叠加不可见文本层，输出 `<样本名>.fidelity.pdf`。

## 运行摘要与阶段数据

`run_manifest.json` 汇总样本数量、页数、失败列表、原生表格数量和布局类型。批量运行后应先检查该文件；`failures` 非空表示存在执行失败的样本。

默认只输出最终文件和 `run_manifest.json`。需要排查问题时，使用 `--save-stages` 保存三份阶段数据：

- `stages/<样本名>.recognition.json`：OCR 和版面模型留下的原始证据。
- `stages/<样本名>.analysis.json`：合并后的段落、标题、表格和媒体块。
- `stages/<样本名>.render_plan.json`：分栏或网格结构及页面缩放结果。

使用 `--save-debug` 后，`debug/` 中会生成版面框、阅读顺序和原始模型结果，用于排查识别遗漏、栏位判断和跨栏范围错误。

DOCX 中的有效公式应显示为可编辑的 Word 公式对象；检查 `document.xml` 时对应 `m:oMath`。畸形 LaTeX 会保留为文本，公式区域不应再输出为图片。

## 视觉验收

将 DOCX 转换为 PDF 或图片，并与原图并排检查。建议按照以下顺序进行：

1. 检查页数，确认不存在空白页或内容溢页。
2. 检查栏数、阅读顺序、跨栏图片和页眉页脚。
3. 检查表格行、公式、图片和图注是否完整。
4. 对照字号、行距、对齐、段落间距和字体。

Markdown 主要检查文本顺序、标题层级、表格语法和图片路径。它不负责复刻页面版式。

## 问题记录

发现回归时，应保留原图、执行命令、`run_manifest.json`、对应样本目录和渲染截图。问题描述应包含页面位置和具体现象，例如“第二栏末行被裁掉”，以便稳定复现。
