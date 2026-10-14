# 常见问题

以下问题按建议的排查顺序整理。命令默认从仓库根目录执行。

## `No module named ...`

先确认当前终端使用的是刚创建的虚拟环境：

```bash
python --version
python -m pip --version
```

然后重新安装依赖：

```bash
python -m pip install -r Code/requirement.txt
```

`Code/test.py` 会直接加载 `Code/docflow_src/`，不需要安装项目 wheel。如果报错来自其他入口，请确认它是否自行配置了源码路径。

## 报错“缺少必要运行资产”

脚本启动时会检查 PaddleOCR 运行时和默认模型。逐项确认这些路径：

```text
Code/third_party/paddle_runtime/ppstructure
Code/third_party/paddle_runtime/ppocr
Code/third_party/paddle_runtime/tools
Code/models_openvino/PP-DocLayoutV3_openvino/PP-DocLayoutV3.xml
Code/models_openvino/PP-OCRv6_small_det_openvino/PP-OCRv6_small_det_openvino_fp32.xml
Code/models_openvino/PP-OCRv6_small_rec_openvino/PP-OCRv6_small_rec_openvino_fp32.xml
Code/models_openvino/PP-OCRv6_small_rec_openvino/ppocrv6_dict.txt
Code/models_openvino/PP-OCRv6_small_rec_openvino/ppocrv6_rapidocr_dict.txt
Code/models_openvino/RapidAI_TableRec_openvino
Code/models_openvino/font_openvino/mobilenetv3.xml
```

路径存在但仍报错时，检查模型是否被放进了同名的双层目录，例如 `Code/models_openvino/models_openvino/...`。

## PDF 没有生成

PDF 由 PyMuPDF 直接生成，不依赖 LibreOffice。先单独运行：

```bash
python Code/test.py --input dataset/exam_paper_02.png --output test-result --formats pdf --no-debug-vis
```

若生成失败，检查终端中的坐标、字体或图片解码报错。DOCX 和 PDF 已相互独立，DOCX 失败不会阻止单独生成 PDF，PDF 失败也不会影响已有 DOCX。

## PDF 元素位置异常

PDF 直接使用版面检测坐标，不再经过 Word 重排。元素位置异常通常来自上游识别结果：

- 文本行框或 `TextRow` 坐标错误。
- 图片、公式或表格缺少 `image_base64`。
- 版面模型将区域分到了错误的语义元素。

此时应检查 `<样本名>.recognition.json`、`<样本名>.json` 和 debug 版面图，不要修改 PDF 渲染器去补偿上游坐标。

## 运行很慢

首次运行需要加载多个模型，耗时通常高于后续单页处理。输入为 PDF 时，可将 `--pdf-dpi` 从 `200` 调整为 `150` 进行初步验证；图片输入不受该参数影响。

debug 图也会占用一些时间和磁盘。只做吞吐测试时加上 `--no-debug-vis`。

## `No ccache found`

这是编译缓存提示，不影响推理结果。没有源码编译需求时可以忽略。需要安装时，使用系统包管理器即可，例如：

```bash
sudo apt-get install ccache
```

## PowerShell 不允许激活虚拟环境

可以为当前用户放开本地脚本执行权限：

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

重新打开 PowerShell 后再执行 `.\.venv\Scripts\Activate.ps1`。

## 提交可复现问题

问题材料应包括原始输入、执行命令、操作系统和 Python 版本、终端日志，以及失败样本的完整输出目录。版面问题还应附带标注位置的截图。
