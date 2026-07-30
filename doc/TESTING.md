# 测试说明

自动测试负责检查内容有没有丢失、表格有没有退化、输出有没有超页。至于版面是否像原图，仍要把 DOCX 转成图片后人工对照。

以下命令都从仓库根目录执行。先按 [DEPLOYMENT.md](DEPLOYMENT.md) 配好 Python 和模型；测试 PDF 输出时还要安装 LibreOffice。

## 单元测试

代码改动后先跑：

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

第一次验证环境时只跑一张图：

```bash
python Code/test.py --input dataset/exam_paper_02.png --output test-result --formats docx,markdown
```

改动涉及版面分析、布局规划或 DOCX 渲染时，处理完整数据集：

```bash
python Code/test.py --input dataset --output test-result --formats docx,markdown,pdf
```

PDF 输出会调用 LibreOffice。程序随后检查 PDF 页数；某个样本超过源页数时，本次运行会记为失败。

## 先看运行摘要

`run_manifest.json` 汇总了样本数量、页数、失败列表、原生表格数量和布局类型。批量运行后先看这里，`failures` 非空就不必逐个打开文件。

每个样本有三份阶段数据：

- `<样本名>.recognition.json`：`pages[].items` 是 OCR 和版面模型留下的原始证据。
- `<样本名>.json`：`pages[].elements` 是合并后的段落、标题、表格和媒体块。
- `<样本名>.render_plan.json`：`pages[].sections` 记录分栏或网格结构，`pages[].fit_scale` 是页面缩放结果。

如果启用了 debug 图，`debug/page_0001.layout_ocr.jpg` 显示检测框，`debug/page_0001.reading_order_columns.jpg` 显示模型顺序。它们适合排查漏检、错栏和跨栏范围错误。

## 人工检查 DOCX

建议把 DOCX 转成 PDF 或图片，再和原图并排查看。检查顺序如下：

1. 先数页数，确认没有空白页或内容溢到下一页。
2. 看大结构：栏数、阅读顺序、跨栏图片和页眉页脚。
3. 看完整性：表格行、公式、图片和图注有没有被裁掉。
4. 最后看样式：字号、行距、对齐、段落间距和字体是否接近原图。

Markdown 主要检查文本顺序、标题层级、表格语法和图片路径。它不负责复刻页面版式。

## 记录问题

发现回归时，至少保留原图、执行命令、`run_manifest.json`、对应样本目录和渲染截图。描述问题时指出页面位置和现象，例如“第二栏末行被裁掉”，比只写“排版不对”更容易复现。
