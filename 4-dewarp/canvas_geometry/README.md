# 4-dewarp Geometry Module

这是 `4-dewarp` 项目级的独立附属几何模块。它从任意上游方法预测的页面三维规则网格估计自由宽高比，并据此生成动态输出画布。算法不假设 A4、Letter 或任何固定纸型，也不依赖文本行内容。

核心包 `dewarp_geometry/` 只依赖 NumPy，不绑定任何检测、分割或三维重建网络。模型适配、图像读写和具体 dewarp 实现均位于核心包之外。

## 方法

对每条网格行和网格列分别累加相邻三维顶点间的欧氏距离：

```text
L_row(i) = sum_j ||X(i,j+1) - X(i,j)||2
L_col(j) = sum_i ||X(i+1,j) - X(i,j)||2

width  = median_i L_row(i)
height = median_j L_col(j)
aspect = width / height
```

中位数聚合降低局部网格噪声影响。输出的 `confidence` 来自行列长度离散度、对边一致性和无效网格段比例，只是诊断分数，并非校准概率。

## 输入约定

- 支持 `HxWx3`、`3xHxW`、`1x3xHxW`。
- 网格行必须沿页面宽度方向，网格列必须沿页面高度方向。
- 三个坐标轴必须使用同一尺度；整体旋转、平移和统一缩放不影响结果。
- 3D 网格应覆盖整张纸。若只覆盖正文区域，估计的是该区域比例而不是页面比例。

## 单元测试

仅测试核心算法时只需 NumPy：

```bash
python -m unittest discover -s tests -v
```

测试包含矩形平面、刚体变换、统一缩放、圆柱弯曲面、CHW 输入、异常网格和动态画布。

## 通用网格命令行

对任意模型导出的 NumPy 三维网格运行：

```bash
python estimate_grid_aspect.py predicted_grid.npy --layout auto --long-edge 1400
```

该入口不加载任何神经网络，可用于 4-dewarp 各条模型管线共用。

## 可选 UVDoc 示例

UVDoc 官方仓库自带约 32 MB 的 `model/best_model.pkl`。为避免复制第三方代码和权重，本 PR 不提交模型文件：

```bash
bash scripts/setup_uvdoc.sh
pip install -r requirements-demo.txt
pip install -r third_party/UVDoc/requirements_demo.txt
python examples/uvdoc_dynamic_canvas.py input.jpg outputs \
  --uvdoc-root third_party/UVDoc \
  --long-edge 1400
```

也可以复用已有 UVDoc：

```bash
python examples/uvdoc_dynamic_canvas.py input.jpg outputs \
  --uvdoc-root /path/to/UVDoc \
  --checkpoint /path/to/best_model.pkl
```

每张成功图像输出 `*_dynamic.png`，目录中同时生成 `aspect_metadata.json`，包含宽高比、弧长、置信度、离散度、警告和画布尺寸。

低置信结果默认不生成拉伸图像。调试时可以添加 `--allow-low-confidence`，但不建议在生产环境启用。

## 接入 4-dewarp

在总项目已经取得 `grid3d` 的位置调用：

```python
from dewarp_geometry import canvas_from_aspect, estimate_aspect_from_grid

estimate = estimate_aspect_from_grid(grid3d, layout="chw")
if estimate.valid and estimate.confidence >= 0.15:
    output_size = canvas_from_aspect(estimate.aspect_ratio, long_edge=1400)
```

调用方可以是 4-dewarp 中的任意 3D 网格模型。只输出二维 mask、轮廓或角点的模型不能直接计算 3D 弧长；它们可以把轮廓作为覆盖范围约束，再由具有 3D 网格输出的管线调用本模块。

## 模型与许可证

可选示例的模型来源是 UVDoc 官方仓库 <https://github.com/tanguymagne/UVDoc>。UVDoc 代码采用其仓库中的 MIT License，使用时应保留原作者许可证和论文引用。核心几何模块不依赖或重新分发其权重。

```bibtex
@inproceedings{UVDoc,
  title={{UVDoc}: Neural Grid-based Document Unwarping},
  author={Floor Verhoeven and Tanguy Magne and Olga Sorkine-Hornung},
  booktitle={SIGGRAPH ASIA, Technical Papers},
  year={2023}
}
```

## 当前限制

- 单张图片的二维轮廓不能单独确定任意曲面的内在宽高比，仍需可靠的 3D 网格。
- 遮挡、页面未完整入镜、模型只预测到正文范围、网格折叠都会降低可信度。
- 估计器恢复无量纲 `W/H`，不恢复毫米等绝对尺寸。
