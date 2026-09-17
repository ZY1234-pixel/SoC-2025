# InternImage-L 实例分割推理

InternImage-L + MaskDINO 文档实例分割部署包，包含全图定位、原图 ROI 精修、
逐实例 mask 和叠加可视化。运行需要 PyTorch/CUDA 与兼容的 NVIDIA GPU。

## 运行

```bash
conda activate doc_instance_seg
cd 4-dewarp/internimage-l-instance
python infer.py
```

可在 IDE 中直接运行 `infer.py`。默认递归读取 `4-dewarp/TEST_dewarp/`，
加载 `checkpoints/best_hd95_1024.pth`，输出到本包 `outputs/`；默认路径不依赖工作目录。

```bash
python infer.py --help
python infer.py --max-images 1
python infer.py ../TEST_dewarp/instance --output-dir outputs_instance
python infer.py /path/to/image.jpg --output-mode production
python infer.py --output-mode debug
python infer.py --no-roi-refine
```

默认采用 1024 全图输入、1024 ROI 精修、0.50 置信度阈值，最多保留 3 个实例。
在 `infer.py` 顶部直接修改输出模式，IDE 点击运行即生效：

```python
OUTPUT_MODE = "debug"       # 叠加可视化 + ROI 边界对比图
# OUTPUT_MODE = "production"  # 只保存逐实例 0/255 单通道 PNG mask
```

命令行 `--output-mode` 优先于该开关。例如 `python infer.py --output-mode production`。
常用参数与 ROI 门限集中在 `instance_seg/settings.py`，命令行参数优先。
模型结构配置位于 `configs/internimage_l_maskdino.yaml`。
`--help` 和参数校验仅依赖 Python 标准库，不加载 PyTorch 或 CUDA。

## 输出

```text
outputs/
├── instances/  # <图名>_instance_001.png，原图尺寸，0/255 mask
├── overlays/   # <图名>.jpg，彩色 mask、轮廓、编号与置信度叠加
└── roi_debug/  # ROI 粗/细边界对比，仅 debug 模式
```

| 模式 | 独立 mask | 叠加图 | ROI 对比图 |
| --- | --- | --- | --- |
| `production` | 是 | 否 | 否 |
| `debug`（默认） | 否 | 是 | 是 |

保留输入子目录结构。第一阶段与 ROI 阶段共用一套权重。

## 目录

```text
internimage-l-instance/
├── infer.py                  # 启动入口
├── configs/                  # 模型结构配置
├── checkpoints/              # 部署权重
├── instance_seg/
│   ├── settings.py           # 路径与推理默认参数
│   ├── cli.py                # 命令行参数及校验
│   ├── pipeline.py           # 批量推理、结果保存、耗时统计
│   ├── runtime.py            # 严格加载权重、置信度排序、CUDA 计时
│   ├── config.py             # Detectron2 配置加载
│   ├── preprocessing.py      # 全图与 ROI 预处理
│   ├── refinement.py         # ROI 二次推理及候选验收
│   ├── geometry.py           # 几何、匹配和边界计算
│   ├── fusion.py             # 局部边界融合与回退
│   ├── visualization.py      # 叠加图和 ROI 对比图
│   ├── io.py                 # 递归读图和文件输出
│   └── models/               # InternImage、MaskDINO、CUDA 算子
├── tests/                    # 部署行为回归检查
├── docs/deployment.md        # 环境、权重来源与编译说明
├── licenses/                 # 上游代码许可证
└── requirements.txt
```

训练工程位于 `../../Instance/Train_InternImage-L-Instance/`。
部署模块使用独立包名 `instance_seg`，与训练侧的 `doc_instance_seg` 区分。

## 验证

在本目录、已安装推理依赖的环境中运行，无需 GPU 权重加载或额外测试框架：

```bash
python -m unittest discover -s tests -v
```

覆盖默认路径、轻量参数解析、非法参数、递归文件筛选、ROI 边界判断、
低置信度过滤以及两种输出模式。真实模型冒烟验证可运行 `python infer.py --max-images 1`。

## 原始 1024 训练结果

部署默认加载真正由 1024×1024 输入训练得到的 `best_hd95_1024.pth`。
来源、选模指标和 SHA-256 记录在 `checkpoints/model_1024.json`。
完整原始训练结果仍位于：

```text
Instance/Train_InternImage-L-Instance/doc_instance_seg/
├── outputs/train/        # 原始 1024 配置、日志、metrics、最终训练断点
└── checkpoints/          # best_hd95、best_mask_ap、best_boundary 等原始权重
```

该训练完成 296,910 次迭代，默认 best_hd95 在第 19,794 次迭代选出，
单实例验证 P95 HD95 为 31.773448。训练默认配置也已恢复为 1024×1024。
