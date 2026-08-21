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

执行以下命令确认系统可以找到 LibreOffice：

```bash
soffice --version
```

Linux 上命令也可能叫 `libreoffice`。如果系统找不到它，安装 LibreOffice 并重新打开终端。随后先排除 DOCX 生成问题：

```bash
python Code/test.py --input dataset/exam_paper_02.png --output test-result --formats docx,markdown
```

这一步成功后再加入 `pdf`。转换失败时保留终端报错；已经生成的 DOCX 仍在对应的运行目录中。

## Word 和 LibreOffice 的页数不同

两者的字体度量和分页行为并不完全一致。程序会检查 LibreOffice 导出的 PDF 页数，但无法预判其他环境中 Microsoft Word 的分页结果。

先确认使用的是同一份 DOCX，再比较发生换页前的字号、行距和表格高度。如果只在 Word 中多出空白页，记录 Word 版本，并同时保留 LibreOffice 的 PDF 作为对照。

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
