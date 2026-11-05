# DocDewarp v2（文档畸变矫正 v2 版本）

本目录为 **文档畸变矫正 v2 版本**，在 UVDoc 基础网络（`UVDocIterative`，即 C-Net + F-Net 自
适应迭代细化）之上，新增了两类正则约束：

- **轴对齐损失（Axis-Aligned Loss, L_AL）**：强制预测 2D 网格在水平/垂直方向与真实网格对齐，
  抑制行/列错位（`utils.axis_alignment_loss`）。
- **交叉（微分坐标）损失（Cross Loss, L_cross）**：复用 DDCP 官方
  `ddcp.Source.loss.Losses.loss_fn4_v5_r_4`，在 2D 网格上施加微分坐标一致性约束。

训练脚本：`train_add_axis_cross_loss.py`
推理脚本：`my_pred_add_axis_loss.py`

---

## 1. 整体流程

推理阶段由两个相对独立的阶段串联（均集成在 `my_pred_add_axis_loss.py` 中）：

**阶段一：在线文档区域检测 + 单应性摆正**
1. 用 `internimage_l_instance`（InternImage-L + MaskDINO）在线预测文档区域「实心掩码」
   （文档内部像素为 255），替代磁盘读取的固定边缘掩码。
2. 背景去除：文档区域外置零（黑色背景）；可选 `mask_erode` 把掩码向内收缩以剔除边缘背景残留。
3. 四边形拟合 + 单应性摆正：由实心掩码取最大外接矩形得到 4 个角点，用
   `cv2.getPerspectiveTransform` 把倾斜/透视的文档拉正为正面视角（仅做一级透视/旋转矫正）。

**阶段二：UVDocIterative 自适应迭代细化**
- 加载改进模型 `UVDocIterative`（C-Net=base_net、F-Net=refine_net）。
- 推理流程与论文 Alg.1 / 专利自适应停止一致：
  - C-Net 对摆正后的原图预测初始后向形变场 `D_0`；
  - 之后每步 F-Net 预测「残差后向形变场」`G`，并用复合算子
    `compose(D_m, G) = grid_sample(D_m, G)` 将多次非线性映射链式累积到「输出 → 输入」的后向映射；
  - 用轴对齐线熵 `axis_aligned_line_entropy` 做自适应停止：当改善率低于 `tau=0.99` 时终止，
    保留线熵最小（对齐最好）的步作为最终结果。
- 最大迭代上限固定为 `M=5`；最终把自适应选中的最佳后向映射 `best_D` 应用到原始摆正图，
  做一次高保真反变形，输出与原图同尺寸。

---

## 2. 目录结构（关键文件）

```text
DocDewarpv2/
├── train_add_axis_cross_loss.py   # 训练脚本（新增 L_AL + L_cross）
├── my_pred_add_axis_loss.py       # 推理脚本（文档区域分割 + 自适应细化）
│
├── model.py                       # UVDocIterative（base_net / refine_net / adaptive_infer）
├── utils.py                       # IMG_SIZE、bilinear_unwarping、load_iterative_model、
│                                  #   axis_alignment_loss、axis_aligned_line_entropy
├── data_doc3D.py                  # Doc3D 数据集
├── data_UVDoc.py                  # UVDoc 数据集
├── data_mixDataset.py             # mixDataset（both 模式下混合两数据集）
├── data_utils.py                  # 数据工具（含 crop_tight 等）
├── data_custom_augmentations.py   # 外观/几何增强
├── line_utils.py                  # 线熵相关工具
│
├── ddcp/                          # 复用的 DDCP 交叉损失（Source/loss.py）
├── internimage_l_instance/        # 文档区域实例分割（InternImage-L + MaskDINO）
├── predict_utils/                 # 推理辅助工具
├── attention/                     # 注意力模块
├── eval/                          # 评估相关（含 C++/Matlab 指标）
│
├── demo.py / docUnet_pred.py / uvdocBenchmark_pred.py   # 其他预测入口
├── train.py / train_add_crossloss.py / train_remove3dloss.py / train_add_crossloss.py  # 其他训练入口
├── docUnet_eval.py / uvdocBenchmark_eval.py / uvdocBenchmark_metric.py                # 评估入口
│
├── requirements_train.txt         # 训练依赖
├── requirements_demo.txt
├── requirements_eval.txt
└── log/                           # 训练日志 / 模型权重输出目录
```

---

## 3. 环境配置

```bash
pip install -r requirements_train.txt   # 训练
pip install -r requirements_demo.txt    # 推理 / demo
pip install -r requirements_eval.txt    # 评估
```

> 推理依赖 `internimage_l_instance`（Detectron2 / CUDA 算子）做在线文档区域分割，
> 其编译与环境说明见 `internimage_l_instance/docs/deployment.md`。

---

## 4. 训练（`train_add_axis_cross_loss.py`）

### 4.1 模型与损失

- **模型**：`model.UVDocIterative(num_filter=32, kernel_size=5)`，由 `base_net`（C-Net，
  基础矫正网）与 `refine_net`（F-Net，自适应细化网）组成。
- **优化目标**（每个细化步 `m`，以及 C-Net 初始步），实际生效分量：

  \[
  \mathcal{L}
  = \alpha\,\mathcal{L}_{2D}
  + \gamma\,\mathcal{L}_{recon}
  + \delta\,\mathcal{L}_{AL}
  + w_{cross}\,\mathcal{L}_{cross}
  \]

  - \(\mathcal{L}_{2D}\)：`criterionL1(grids[m], grid2D_{GT})`（2D 网格 L1）。
  - \(\mathcal{L}_{recon}\)：`criterionL1(unwarped_{pred}, unwarped_{GT})`（重建 L1）。
    **注意**：`gamma_w` 在 `epoch < ep_gamma_start` 时为 0，即从 `ep_gamma_start` 起才启用重建损失。
  - \(\mathcal{L}_{AL}\)：`utils.axis_alignment_loss(grids[0], grid2D_{GT})`，**仅作用于 C-Net
    初始输出 `grids[0]`**（方案 A：避免在各细化步重复叠加同一约束、放大梯度）。
  - \(\mathcal{L}_{cross}\)：每步 2D 网格上的 DDCP 微分坐标损失。内部对齐 DDCP 官方 Eq.(6)：
    `l1_pixel + cross_alpha*local + 0.001*edge + 0.01*rectangle`，
    且先经 `_grid_to_pixel` 把 UVDoc 的 `[-1,1]` 网格还原到像素尺度再计算
    （否则 quadratic 项量级过弱）。`edge` / `rectangle` 该项恒为 0，系数按官方原样保留。
  - 3D 网格损失 \(\mathcal{L}_{3D}\)：代码已计算但默认 `beta_w = 0.0`，**当前不生效**。

- **日志分量**：逐 epoch 累加并输出 `Total / L2D / L3D / Lrecon / LAL / Lcross / ReconMSE`。

### 4.2 分阶段序贯训练（`--train_stage`）

- `joint`（默认）：C-Net 与 F-Net 端到端联合训练。
- `cnet`：仅训练 C-Net（冻结 F-Net），`refine_iters` 强制为 0，得到收敛的 C-Net 权重。
- `fnet`：冻结 C-Net，用收敛的 C-Net 权重（`--cnet_ckpt`，默认 `./model/best_model.pkl`）
  初始化 F-Net 并微调；自动套用论文默认超参（AdamW，`lr=1e-5`，`weight_decay=1e-4`，
  `n_epochs=0`、`n_epochs_decay=9`，共 10 个 epoch 线性衰减到 ~0）。

### 4.3 学习率与优化器

- 优化器：`Adam` 或 `AdamW`（由 `--optimizer` 决定），`betas=(0.9, 0.999)`。
- 调度：`LambdaLR`，前 `n_epochs` 个 epoch 保持恒定学习率，随后 `n_epochs_decay` 个 epoch
  线性衰减到 0。
- 支持 `--resume` 断点续训（恢复 `model_state` / `optimizer_state` / `epoch`）。
- 支持 `--base_ckpt` 用预训练 UVDocNet 权重同时初始化 C-Net 与 F-Net。

### 4.4 数据

- 数据集：`doc3D`（`data_doc3D`）、`UVDoc`（`data_UVDoc`），`both` 模式用 `mixDataset` 混合。
- 增强：外观增强（`appearance_augmentation`，默认 `["visual","noise","color"]`）；
  UVDoc 几何增强（`geometric_augmentationsUVDoc`，默认 `["rotate"]`）。
- 验证：默认用 doc3D 验证集（仅 uvdoc 训练时用 UVDoc），仅评估 C-Net 初始矫正（`max_iter=0`），
  以重建 MSE 选最优模型。

### 4.5 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path_doc3D` | `./data/doc3D/` | Doc3D 数据路径 |
| `--data_path_UVDoc` | `./data/UVDoc/` | UVDoc 数据路径 |
| `--data_to_use` | `uvdoc`（choices: `both`/`doc3d`/`uvdoc`） | 训练使用的数据集 |
| `--batch_size` | `8` | 批大小 |
| `--n_epochs` | `10` | 学习率恒定的 epoch 数 |
| `--n_epochs_decay` | `20` | 学习率线性衰减到 0 的 epoch 数 |
| `--lr` | `1e-5` | 初始学习率（AADD 用 1e-5 避免破坏预训练权重） |
| `--optimizer` | `adam`（choices: `adam`/`adamw`） | 优化器 |
| `--weight_decay` | `0.0` | 权重衰减（AdamW 微调 F-Net 时设 `1e-4`） |
| `--alpha_w` | `5.0` | 2D 网格 L1 损失权重 |
| `--beta_w` | `0.0` | 3D 网格 L1 损失权重（当前为 0，不生效） |
| `--delta_w` | `10` | 轴对齐损失 L_AL 权重 |
| `--cross_w` | `0.1` | 交叉（微分坐标）损失权重 |
| `--cross_alpha` | `0.1` | DDCP 交叉损失内部 local 项权重（对应 `lambda_loss_a=0.1`） |
| `--refine_iters` | `3` | 训练时展开的 F-Net 细化步数（0 = 仅 C-Net） |
| `--base_ckpt` | `None` | 用于初始化 C-Net/F-Net 的预训练 UVDocNet 路径 |
| `--train_stage` | `joint`（choices: `joint`/`cnet`/`fnet`） | 分阶段训练策略 |
| `--cnet_ckpt` | `./model/best_model.pkl` | `fnet` 阶段用于初始化 F-Net 的收敛 C-Net 权重 |
| `--gamma_w` | `1.0` | 重建损失权重 |
| `--ep_gamma_start` | `10` | 从该 epoch 起启用重建损失 |
| `--resume` | `None` | 断点续训的模型路径 |
| `--logdir` | `./log/default` | 日志 / 模型输出目录 |
| `-a/--appearance_augmentation` | `["visual","noise","color"]` | 外观增强（可选 shadow/blur/visual/noise/color） |
| `-gUVDoc/--geometric_augmentationsUVDoc` | `["rotate"]` | UVDoc 几何增强（可选 rotate/flip/perspective） |
| `--num_workers` | `8` | DataLoader 工作进程数 |

### 4.6 运行示例

```bash
# 默认（uvdoc 数据集，joint 训练）
python train_add_axis_cross_loss.py

# 使用 doc3D + UVDoc 混合数据，分阶段先训 C-Net 再微调 F-Net
python train_add_axis_cross_loss.py --data_to_use both --train_stage cnet
python train_add_axis_cross_loss.py --data_to_use both --train_stage fnet --cnet_ckpt ./model/best_model.pkl
```

**输出**：每个 epoch 在 `logdir/<实验名>/` 下保存
`ep_{epoch}_{val_mse:.5f}_{train_mse:.5f}_best_model.pkl`（验证 MSE 改善或末轮保存），
并在 `logdir/` 下写入 `<实验名>.txt` 训练/验证日志（含 Total/L2D/L3D/Lrecon/LAL/Lcross/ReconMSE）。

---

## 5. 推理（`my_pred_add_axis_loss.py`）

### 5.1 运行示例

```bash
python my_pred_add_axis_loss.py \
  --ckpt-path log/default/params8_lr=1e-05_optadam_wd0.0_nepochs10_nepochsdecay20_alpha5.0_beta0.0_gamma=1.0_gammastartep10_datauvdoc_refineIters3_delta10_crossW0.1_stagejoint/ep_31_0.00169_0.00342_best_model.pkl \
  --img_path data/test_common_test_dataset/img/rotate \
  --save_path data/test_common_test_dataset/dewarp/rotate \
  --debug data/test_common_test_dataset/dewarp/rotate/debug
```

### 5.2 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ckpt-path` | `log/default/params8_lr=1e-05_optadam_wd0.0_nepochs10_nepochsdecay20_alpha5.0_beta0.0_gamma=1.0_gammastartep10_datauvdoc_refineIters3_delta10_crossW0.1_stagejoint/ep_31_0.00169_0.00342_best_model.pkl` | UVDoc 改进模型权重（`.pkl`） |
| `--img_path` | `data/test_common_test_dataset/img/rotate` | 输入图像路径或文件夹 |
| `--save_path` | `data/test_common_test_dataset/dewarp/rotate` | 矫正结果保存目录 |
| `--debug` | `data/test_common_test_dataset/dewarp/rotate/debug` | 中间结果（实心掩码 / 去背景图 / 摆正图）保存目录 |
| `--margin` | `100` | **已不再生效**（改进模型基于原始摆正图做复合采样，迭代间不再回填黑边；保留仅为兼容命令行） |
| `--mask_erode` | `10` | 背景去除时文档掩码向内收缩的像素距离（0 = 不收缩） |

### 5.3 输出

位于 `--save_path` 下：

```text
<save_path>/
├── uwp_img/         # 最终矫正结果（自适应选中最佳步）
├── bm/              # 最终后向形变场（.mat，键名 "bm"）
├── uwp_img_iter{i}/ # 各迭代步中间矫正图（i=1..5）
└── bm_iter{i}/      # 各迭代步中间形变场（.mat）
<debug>/             # _solid_mask.png / _bg_removed.png / _frontal.png 等中间可视化
```

推理结束后打印每张图的 `n_opt`（实际自适应停止步 / 总步数）、轴对齐线熵与耗时，
并汇总平均耗时与 FPS。

---

## 6. 注意事项

- 训练默认 `--data_to_use uvdoc`，且 `--beta_w 0.0`：3D 网格损失已关闭，仅 2D/重建/轴对齐/交叉四项参与。
- 重建损失 `gamma_w` 在 `ep_gamma_start`（默认 10）之前为 0，之后才计入总损失。
- 推理的 `--margin` 参数已被训练/推理流程弃用（自适应复合采样不再需要回填黑边），保留仅为命令行兼容。
- 推理依赖 `internimage_l_instance` 的在线文档区域分割权重，需按该子模块说明准备后使用。
- 模型输入分辨率由 `utils.IMG_SIZE` 定义，推理时把摆正图 `F.interpolate` 到该分辨率，
  再对「全分辨率摆正图」做一次反变形，保证输出与原图同尺寸。
