# PP-FormulaNet_plus-S FP16 交付说明

本次整理了 PP-FormulaNet_plus-S 公式识别模型的两个 FP16 包，分别用于 PaddleOCR/Paddle Inference 集成和 ONNX Runtime 跨框架集成。两个版本均保留 UniMERNet tokenizer，并附带 22 张公式测试图的逐图结果和可视化报告。

## 目录结构

```text
PP-FormulaNet_plus-S_pure_paddle_fp16/   # 纯 Paddle FP16 静态推理版本
PP-FormulaNet_plus-S_pure_onnx_fp16/     # 纯 ONNX FP16 版本
公式识别算法说明文档.docx
README.md
```

## Paddle FP16 包内容

路径：`PP-FormulaNet_plus-S_pure_paddle_fp16/`

```text
inference.json                    # Paddle 静态图结构
inference.pdiparams               # Paddle FP16 权重
inference.yml                     # PaddleOCR 预处理/后处理配置
tokenizer/                        # UniMERNet tokenizer
MANIFEST.sha256                   # 文件校验清单
test/                             # 22 张测试图、结果和 HTML 报告
README.md                         # 该版本详细说明
```

接入方式：如果目标工程已经使用 PaddleOCR 公式识别流程，可将该目录作为模型目录加载；如果是自建 Paddle Inference 流程，需要保持 `inference.yml` 中的预处理、后处理和 tokenizer 解码逻辑一致。

注意：该包中的 Paddle 静态图建议使用与导出环境兼容的 Paddle 3.0.x / PaddleOCR 3.x 运行时。若目标工程固定为 Paddle 3.2.x 或更高版本，建议在目标环境中重新导出并完成 predictor smoke test。

## ONNX FP16 包内容

路径：`PP-FormulaNet_plus-S_pure_onnx_fp16/`

```text
inference.onnx                    # ONNX FP16 模型
infer_formula_onnx.py             # ONNX Runtime 融合示例，含预处理和 tokenizer 解码
requirements_onnx.txt             # 推理脚本依赖
tokenizer/                        # UniMERNet tokenizer
MANIFEST.sha256                   # 文件校验清单
test/                             # 22 张测试图、结果和 HTML 报告
README.md                         # 该版本详细说明
```

模型接口：

- Runtime：ONNX Runtime
- Input：`x`，`float32`，shape `[N, 1, 384, 384]`
- Output：`fetch_name_0`，`int64` token ids
- 后处理：使用 `tokenizer/` 将 token ids 解码为 LaTeX

图像预处理需要与 PaddleOCR UniMERNet 保持一致：公式图裁边后缩放或填充到 `384 x 384`，灰度化，归一化，并整理为 `[N, 1, 384, 384]`。

ONNX 包已提供 `infer_formula_onnx.py`，对方可以直接使用其中的 `FormulaONNXPredictor` 类完成融合，不需要从零实现 ONNX Runtime 推理、预处理和 tokenizer 解码。单图验证命令如下：

```bash
cd PP-FormulaNet_plus-S_pure_onnx_fp16
pip install -r requirements_onnx.txt
python infer_formula_onnx.py --image test/images/arrow_1.png
```

## 测试结果

两个交付包均包含 `test/` 目录：

```text
test/images/       # 22 张输入公式图
test/results/      # 每张图的 LaTeX、JSON 结果，Paddle 版本另含渲染 PNG
test/results.csv   # 逐图结果表
test/summary.json  # 汇总指标
test/report.html   # 可视化报告，含 LaTeX 渲染
test/katex/        # 离线 KaTeX 渲染资源
```

打开各自的 `test/report.html` 可查看每张测试图的输入图片、识别出的 LaTeX、渲染公式和对比结果。该 HTML 使用相对路径和本地 KaTeX 资源，拷贝或打包时需保留整个 `test/` 目录。


## 校验

每个模型目录下均提供 `MANIFEST.sha256`。解压或拷贝后可在对应目录中执行：

```bash
sha256sum -c MANIFEST.sha256
```

校验通过后，再使用 `test/` 中的 22 张图片做一次工程侧 smoke test。
