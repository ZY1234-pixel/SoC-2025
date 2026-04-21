# 故障排查

## 1. 报错 `No module named ...`

说明 Python 依赖未安装完整。请在 `Code/` 目录重新执行：

```bash
python -m pip install -r requirement.txt
python -m pip install wheels/docflow-0.3.0-py3-none-any.whl
```

## 2. 报错“缺少必要运行资产”

通常表示打包目录不完整或文件被移动。请确认以下路径存在：

- `Code/third_party/paddle_runtime/ppstructure`
- `Code/third_party/paddle_runtime/ppocr`
- `Code/third_party/paddle_runtime/tools`
- `Code/models/layout/doclayout_yolo_docstructbench_headfloat100_runtime`
- `Code/models/det/ch/PP-OCRv5_mobile_det_infer`
- `Code/models/rec/ch/PP-OCRv5_mobile_rec_infer`
- `Code/models/table/SLANet_plus_infer`

## 3. PDF 输出失败

请依次检查：

- 是否已安装 LibreOffice
- `soffice` 是否可在命令行直接运行
- 是否先用 `docx,markdown` 跑通过主链路

可先执行：

```bash
python test.py --input ../dataset --output ../test-result --formats docx,markdown
```

成功后请到 `test-result/run_xxx/samples/<样例名>/` 下查看结果。

## 4. Windows 无法激活虚拟环境

如果 PowerShell 禁止执行脚本，可管理员身份运行一次：

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 5. 速度较慢

- 首次运行会加载模型，速度通常偏慢
- 批量 PDF 可先尝试 `--pdf-dpi 150`
- 开启每页 `debug/` 可视化图会增加一定输出时间

## 6. 警告 `No ccache found`

如果你看到：

`UserWarning: No ccache found...`

这是警告，不是错误，通常不会导致流程失败。它的含义只是本机没有安装编译缓存工具 `ccache`。

可选处理方式：

- 直接忽略：只要结果正常产出即可继续测试
- 安装 `ccache`

```bash
conda install -c conda-forge ccache
```

或：

```bash
sudo apt-get update
sudo apt-get install -y ccache
```

安装后可执行：

```bash
ccache --version
```

## 7. 如何提供可复现问题

建议打包以下内容反馈：

- 原始输入文件
- 执行命令
- 终端完整日志
- 失败对应输出文件
- 操作系统和 Python 版本
