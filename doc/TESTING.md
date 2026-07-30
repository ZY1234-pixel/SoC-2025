# 测试操作手册


## 1. 测试前检查

开始前请确认：

- 已按 `DEPLOYMENT.md` 完成环境安装
- `dataset/` 目录中已放入待测图片或 PDF
- 若要测试 PDF 输出，系统命令行可直接调用 `soffice`

## 2. 主测试命令

```bash
python Code/test.py --input <输入路径> --output <输出路径> --formats <格式列表>
```

参数说明：

- `--input/-i`：输入文件或目录，默认 `dataset/`
- `--output/-o`：结果根目录，默认 `test-result/`
- `--formats/-f`：输出格式，支持 `docx,markdown,pdf`
- `--pdf-dpi`：PDF 转图片 DPI，默认 `200`
- `--no-debug-vis`：关闭可视化图导出
- `--layout-model`：选择版面分析模型，默认 `pp-doclayout-v3`

## 3. 推荐测试流程

### 3.1 单样本冒烟

用于确认环境、模型和主流程可跑通：

```bash
python Code/test.py --input dataset/exam_paper_02.png --output test-result --formats docx,markdown
```

通过标准：

- 命令正常结束
- 输出目录出现对应的 `json / docx / md`
- 控制台无未处理异常

### 3.2 目录级批量测试

用于验证主流程稳定性：

```bash
python Code/test.py --input dataset --output test-result --formats docx,markdown
```

通过标准：

- 输入目录中的样本都能被识别
- 每个样本目录下都能产出对应 `json / docx / md`
- 每个样本目录下的 `debug/` 中能看到版面分析与排序结果

### 3.3 带 PDF 的完整测试

在系统可用 `soffice` 的前提下执行：

```bash
python Code/test.py --input dataset --output test-result --formats docx,markdown,pdf
```

通过标准：

- 每个样例目录下会额外产出 `<样例名>.pdf`
- PDF 与 DOCX 主体内容保持一致

## 4. 验收项

### 4.1 JSON

- `pages[].blocks` 是否非空
- `image_path` 是否与输入一致
- 结构块类别是否基本合理

### 4.2 DOCX

- 阅读顺序是否正确
- 分栏是否符合原页面结构
- 标题、正文、表格、图片位置是否合理
- 是否存在明显错栏、串栏、多页溢出或异常断行

### 4.3 Markdown

- 表格是否能直接渲染
- 图片资源路径是否存在
- 文本顺序和层次是否正确

### 4.4 Debug 可视化

检查 `test-result/run_xxx/<样例名>/debug/` 下的：

- `page_0001.layout_ocr.jpg`
- `page_0001.reading_order_columns.jpg`

重点看检测框、排序顺序和最终版面块是否符合直觉。
