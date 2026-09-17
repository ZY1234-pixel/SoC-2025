# DocDewarp（D2Dewarp 文档图像矫正）

**D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping**

本项目基于论文代码仓库 [D2Dewarp 官方代码](https://github.com/xiaomore/D2Dewarp) 的 PyTorch 实现，并在其基础上增加了**在线文档区域检测**与**在线文本 h/v 线分割**，构成完整的端到端文档图像矫正流水线。

---

## 1. 项目简介

实际推理流水线由三部分模型串联组成（均已在 `predict.py` 中集成）：

1. **文档区域实例分割（`internimage_l_instance/`）**：使用 InternImage-L + MaskDINO 在线预测文档的「实心区域掩码」，再由 `solid_region_to_edge` 处理为文档边缘掩码（边界一圈 255），替代原先从磁盘读取的固定边缘掩码。
2. **文本 h/v 线分割（`text_seg/`）**：使用 UNet（主干可选 `vgg` / `resnet50` / `starnet`）在线预测文本行（horizon_line）与竖线（vertical_line）的像素级特征图，作为形变场网络的输入，替代原先从磁盘读取的边缘图。
3. **形变场网络（`d2dewarp/networks/d2dewarp_model.py`）**：以 h 特征图、v 特征图、原图、文档边缘掩码为输入，预测形变场（bm），再通过 `grid_sample` 得到矫正结果。

推理阶段对每张图还会执行多级后处理：第一级 homography/网格展平（`--homo_rectify`）、形变网格铺满（`--fit_bm`）、越界采样点裁剪（`--prune_bm`）/ 对齐文档边界（`--snap_lbl`），并以 `--k` 轮迭代逐步精修最终矫正效果。

---

## 2. 项目结构

```text
DocDewarp/
│
├── predict.py                  # 推理入口（串联实例分割 + 文本分割 + 形变场网络）
├── train.py                    # D2Dewarp 形变场网络训练脚本
├── train.sh                    # 训练启动脚本（bash）
├── requirements.txt            # 依赖环境
├── 说明文档.docx               # 设计说明文档
│
├── d2dewarp/                   # 核心算法包
│   ├── networks/               # 网络结构
│   │   ├── d2dewarp_model.py   # D2Dewarp 形变场主网络（D2DewarpModel_my）
│   │   ├── unet_model.py        # UNet 编码器-解码器
│   │   ├── cross_attn.py        # 交叉注意力模块
│   │   ├── mask_encoding.py     # 掩码编码
│   │   ├── mask2flow_model.py   # 掩码到形变场
│   │   └── unet_parts.py        # UNet 基础组件
│   ├── loader/                 # 数据加载
│   │   ├── dataset_doc3d_grid_HV.py  # Doc3D 网格/h/v 数据集
│   │   └── data_prefetcher.py        # 数据预取
│   ├── model_utils/            # 训练工具
│   │   ├── losses.py            # 损失函数
│   │   ├── lr_scheduler.py      # 学习率调度（WarmupCosineLR）
│   │   ├── utils_model.py       # 模型工具
│   │   └── dewarp_utils.py
│   └── predict_utils/          # 推理后处理工具
│       ├── dewarp_core.py       # 第一级网格化展平（dewarp_document）
│       ├── edge_util.py         # 文档掩码 / 边缘 / 采样点裁剪等工具
│       ├── draw_grid.py         # 形变场网格可视化
│       └── main.py
│
├── text_seg/                   # 文本 h/v 线分割模型（UNet）
│   ├── unet.py                 # Unet 封装（detect_image / 前处理）
│   ├── predict.py              # 独立的单图 / 视频 / 文件夹预测脚本
│   ├── nets/                   # 主干网络：unet.py / vgg.py / resnet.py / starnet.py
│   ├── utils/                  # 前后处理工具
│   └── logs/                   # 训练权重（best_epoch_weights.pth）
│
└── internimage_l_instance/     # 文档区域实例分割（InternImage-L + MaskDINO）
    ├── infer.py                # 独立推理入口
    ├── configs/                # 模型结构配置（internimage_l_maskdino.yaml）
    ├── checkpoints/            # 部署权重（best_hd95_1024.pth）
    ├── instance_seg/           # 推理包：pipeline / runtime / preprocessing / refinement ...
    ├── tests/                  # 回归测试
    ├── docs/                   # 部署说明（环境 / 权重来源 / 编译）
    ├── licenses/               # 上游许可证
    └── README.md               # 该子模块独立说明文档
```

---

## 3. 环境配置

```bash
pip install -r requirements.txt
```

> 依赖包含 `torch==1.12.1+cu116`、`torchvision==0.13.1+cu116`、`opencv-python`、`numpy` 等（详见 `requirements.txt`）。
> 文档区域实例分割子模块基于 Detectron2 / CUDA 算子，相关编译与环境说明见 `internimage_l_instance/docs/deployment.md`。

---

## 4. 模型权重

推理需要三个权重文件，默认路径如下（可按需修改命令行参数）：

| 权重 | 默认路径 | 说明 |
|------|----------|------|
| 形变场网络 | `output/d2dewarp_add_edge_mask_448_uvfinal/200.pt` | D2Dewarp 主模型，由 `train.py` 训练得到 |
| 文本分割 | `text_seg/logs/best_epoch_weights.pth` | `text_seg` 的 UNet 权重 |
| 文档区域实例分割 | `internimage_l_instance/checkpoints/best_hd95_1024.pth` | InternImage-L + MaskDINO 权重 |

请将对应权重放到上述默认路径（或运行时通过参数指定）。

---

## 5. 推理（`predict.py`）

直接运行（使用代码中默认参数）或命令行指定参数：

```bash
# 默认参数运行
python predict.py

# 指定输入 / 输出 / 模型
python predict.py \
  --img_path test_common_test_dataset/img/perspective \
  --save_path test_common_test_dataset/dewarp \
  --model_path output/d2dewarp_add_edge_mask_448_uvfinal/200.pt
```

推理输出（位于 `--save_path` 下）：

```text
<save_path>/
├── perspective/             # 最终矫正结果（png）
├── bm_grid/perspective/     # 叠加形变场网格的可视化
├── debug_masked_input/      # 调试图：首轮输入、h/v 预测、第一级展平结果等
└── features/                # CoordAtt 融合输出的 h/v 特征图
```

### 5.1 参数说明

`predict.py` 支持以下命令行参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_size` | int | 448 | 模型输入图像尺寸 |
| `--model_path` | str | `output/d2dewarp_add_edge_mask_448_uvfinal/200.pt` | 形变场网络权重路径 |
| `--img_path` | str | `test_common_test_dataset/img/perspective` | 输入图像文件夹（多图）或单张图片路径 |
| `--save_path` | str | `test_common_test_dataset/dewarp` | 输出结果保存路径 |
| `--in_chans` | int | 3 | 输入通道数（RGB） |
| `--d_model` | int | 448 | 主干网络 / 注意力特征维度 |
| `--k` | int | 2 | 迭代预测轮数（逐轮以去黑边后的矫正图作为输入精修） |
| `--homo_rectify` | bool | True | 第一级矫正：拟合文档四边形 / 网格展平，把倾斜透视文档拉正 |
| `--homo_preserve_ar` | bool | False | 第一级矫正输出是否按文档长宽比取尺寸 |
| `--homo_canvas_margin` | int | 200 | 第一级矫正后四周留白像素，作为后续迭代余量 |
| `--homo_grid` | int | 10 | 网格化第一级矫正的网格密度（每边格数） |
| `--fit_bm` | bool | True | 把形变场采样区拉伸铺满整幅输入图 |
| `--fit_mode` | str | `homography` | 铺满方式：`homography` / `bbox` / `off` |
| `--fit_quantile` | float | 0.5 | 统计支撑区间两端忽略比例，抑制纸边离群点 |
| `--fit_clamp` | bool | True | 铺满后把采样坐标夹到图像范围内避免黑边 |
| `--prune_bm` | bool | False | 删除落到文档边缘外的采样点并重参数化，使文档边缘贴合图像边缘 |
| `--prune_smooth` | int | 9 | 重参数化时边界平滑窗口（奇数） |
| `--prune_inward` | int | 1 | 越界判定时文档掩码向内腐蚀像素数 |
| `--snap_lbl` | bool | False | 把采样点对齐到文档四边形 / 最近文档边缘（与 prune_bm 可二选一） |
| `--inward_offset` | float | 5.0 | 兜底重定向后往文档内部偏移像素数 |
| `--grid_size` | int | 20 | 叠加到原图的形变场网格密度（每边格数） |
| `--seg_model_path` | str | `text_seg/logs/best_epoch_weights.pth` | 文本分割模型权重路径 |
| `--seg_backbone` | str | `starnet` | 文本分割主干：`vgg` / `resnet50` / `starnet` |
| `--seg_input_size` | int | 640 | 文本分割输入尺寸 |
| `--seg_num_classes` | int | 3 | 文本分割类别数（background + horizon_line + vertical_line） |

---

## 6. 训练（`train.py`）

`train.py` 训练 D2Dewarp 形变场网络，使用 UVDoc 风格数据集（`Warp_DataSet_my`），损失为预测形变场与真值 `bm` 的 L1 Loss。

```bash
python train.py --data_path <数据集路径> --pre_trained_path <预训练权重> --batch_size 4
```

常用参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_size` | int | 448 | 输入尺寸 |
| `--d_model` | int | 448 | 特征维度 |
| `--in_chans` | int | 3 | 输入通道数 |
| `--epochs` | int | 200 | 训练轮数 |
| `--batch_size` | int | 4 | 批大小 |
| `--lr` | float | 1e-4 | 初始学习率 |
| `--min_lr` | float | 1e-6 | 最小学习率 |
| `--warmup_epochs` | int | 20 | 学习率预热轮数 |
| `--weight_decay` | float | 0.01 | 权重衰减 |
| `--data_path` | str | `UVDoc_final` | 数据集路径 |
| `--pre_trained_path` | str | `/home/tjq/PycharmProjects/D2Dewarp-main/model_weight.pt` | 预训练权重 |
| `--save_path` | str | `output` | 模型 / 日志保存目录 |
| `--exp_name` | str | `dewarp_HV_100K` | 实验名（自动追加 input size 与 seed） |
| `--seed` | int | 24610 | 随机种子 |

也可通过 `bash train.sh <OUTPUT_DIR>` 启动（脚本内可修改 `exp_name`、学习率、数据集路径等）。

---

## 7. 子模块独立使用

- **文档区域实例分割**：见 `internimage_l_instance/README.md`，独立入口为 `python internimage_l_instance/infer.py`。
- **文本 h/v 线分割**：见 `text_seg/predict.py`，支持单图 / 视频 / 文件夹 / 导出 ONNX 等模式。

---

## 8. 注意事项

- 推理脚本 `predict.py` 在导入形变场网络后还会调用 `vis_utils` 模块（`save_feature_map` / `save_spatial_map`）做特征图可视化，运行时需保证该模块可被 Python 导入（位于 `PYTHONPATH`）。
- 默认 `--homo_canvas_margin 200` 会在每轮输入四周补白边、并在输出时按该边距裁掉，以保证多轮迭代时尺寸一致且不引入多余缩放。
- 实例分割权重 `best_hd95_1024.pth` 与文本分割权重 `best_epoch_weights.pth` 需自行准备后放到默认路径，否则推理会因找不到权重而报错。
