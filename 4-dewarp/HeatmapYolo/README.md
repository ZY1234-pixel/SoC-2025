# HeatmapYolo v14：书本文档四角点检测（YOLOv8 主干 + 热力图输出头）

基于 YOLOv8s 主干改造的角点检测方案：输出头为「文档类别分类 + 4 通道关键点热力图」，
热力图在 P3（stride 8）特征上输出，解码采用 argmax + 抛物线亚像素精修。
当前为 v14 版本（更强的旋转/透视增强 + 推理侧几何后处理），适用于旋转、透视、
弯曲、折叠等强退化场景；历史版本（v12 等）可通过 git 历史查看。

## 更新日志

- **2026-08 v14**：训练增强加大（旋转 ±45°、透视 0.002、缩放 0.4、平移 0.05）；
  新增推理侧后处理 `corner_postprocess.py`（置信度门控、凸四边形排序防连线交叉、
  平行四边形补全防角点扎堆/飞出画面）；`predict.py` / `predict_batch.py` /
  `run_infer_samples.py` 全部接入后处理，并支持弱角点（疑似落在背景/边框）标记。
- **2026-07 v12（初始上库版本）**：修复热力图头通道塌缩（`warmup_bias_lr=0.0`）、
  修正 heatmap loss 分母为逐可见像素 MSE，开启旋转 + 透视增强。
  v12 代码可通过 git 历史查看，常规图精度略优于 v14，极端退化场景鲁棒性弱于 v14。

## 目录结构

```text
HeatmapYolo_v14/
├── train.py                  # 训练入口（分类 + 热力图联合训练，v14 增强配置）
├── heatmap_cls_model.py      # 模型定义（HeatmapClsHead / HeatmapClsModel）
├── heatmap_utils.py          # 高斯热力图生成 / 亚像素解码
├── loss_cls_heatmap.py       # 分类 + 热力图损失（逐可见像素 MSE）
├── dataset_cls_heatmap.py    # 数据集封装（强制 pose 任务解析关键点）
├── trainer_cls_heatmap.py    # 训练器（跳过 bbox 验证）
├── corner_postprocess.py     # 推理侧角点后处理（置信度门控 + 几何一致性修复）
├── predict.py                # 单张图片推理（已接入后处理）
├── predict_batch.py          # 文件夹批量推理（已接入后处理）
├── evaluate.py               # val 集指标评估
├── run_infer_samples.py      # 推理可视化（预测点 + GT + 弱角点标记）
├── make_sheets.py            # 将推理结果拼成联系表
├── export_torchscript.py     # 导出 TorchScript（部署用）
├── configs/
│   └── yolov8s.yaml          # 模型结构配置
├── doc_4corners_heatmap.yaml # 数据配置（路径按本机修改）
└── deploy/
    └── ncnn/                 # NCNN C++ 推理工程
        ├── CMakeLists.txt
        ├── build.bat
        └── main.cpp
```

权重（`*.pt` / `*.torchscript.pt` / `*.ncnn.param` / `*.ncnn.bin`）不随仓库提供，
请自行准备并放到 `weights/` 目录（v14 权重见项目组网盘）。

## 环境依赖

见 `requirements.txt`。核心版本：Python 3.11、PyTorch 2.x、ultralytics 8.4.110。

## 训练（v14 配置）

```bash
python train.py
```

- 输入 512×512，batch 32，100 epochs，SGD（lr0=0.005，warmup 3 epochs）
- 增强：旋转 ±45°、透视 0.002、缩放 0.4（缩放留边）、平移 0.05、水平翻转 0.5
  （角点被增强移出画面时自动标记不可见并从 loss 中屏蔽）
- 关键修复项：
  - `warmup_bias_lr=0.0`：默认 0.1 会让热力图头里的 BN/偏置参数在 warmup 阶段被打爆，
    导致通道塌缩（某个/全部角点通道输出全 0）。必须保留。
  - `heatmap_weight=40960`：loss 为逐可见像素 MSE，该值 = 10 × (imgsz/8)²，
    保持与原权重 10 相同的梯度权重。

## 推理

```bash
# 单张
python predict.py --image <图片路径> --weights weights/heatmap_v14_512_aug/last.pt
# 批量
python predict_batch.py --input_dir <图片目录> --weights weights/heatmap_v14_512_aug/last.pt
```

推理链路：letterbox 到 512 + RGB /255 -> 模型 -> argmax + 亚像素解码 ->
`corner_postprocess.py` 后处理（置信度门控、凸四边形排序、平行四边形补全、夹取回画面）。

## 推理侧后处理（v14 新增）

`corner_postprocess.py` 解决三类典型失败：

1. **飞出画面**：弱响应/贴边角点用平行四边形几何估计替换并夹取回画面；
2. **角点扎堆**：两个角点距离过近时保留置信度高者，低者用“最接近矩形的平行四边形补全”恢复；
3. **连线交叉（漏斗形）**：按质心极角得到凸四边形环形顺序，保证连线不交叉，
   再按 TL/TR/BR/BL 角色排序。

`run_infer_samples.py` 还会对“边缘能量明显偏低且贴近画面边框”的角点画黄圈标记（WEAK），
供下游决定是否拒绝。

## 评估

```bash
python evaluate.py --weights weights/heatmap_v14_512_aug/last.pt
python evaluate.py --compare <旧权重> <新权重>
```

v14 在 val 集（637 张，512 输入）的指标：

| 角点 | median | mean | ≤3px |
|------|--------|------|------|
| kpt0 左上 | 3.37px | 10.60px | 46.2% |
| kpt1 右上 | 3.10px | 11.35px | 49.1% |
| kpt2 右下 | 3.35px | 12.97px | 47.6% |
| kpt3 左下 | 4.22px | 16.19px | 44.1% |

说明：v14 的强旋转增强提升了极端场景鲁棒性（TEST_dewarp 旋转/透视类基本全部正常，
配合后处理无飞出/扎堆/漏斗），代价是常规图精度比 v12 略降（v12 中位约 2.3~2.8px）。
若常规图精度优先，可退回 v12。

## 部署（NCNN）

```bash
# 1. 导出 TorchScript
python export_torchscript.py --weights weights/heatmap_v14_512_aug/last.pt
# 2. 用 pnnx 转 ncnn（示例）
pnnx.exe weights/heatmap_v14_512_aug/heatmap_v14_512_aug.torchscript.pt inputshape="1,3,512,512"
# 3. 编译 C++ 工程（MSVC，需 ncnn + OpenCV，路径在 CMakeLists.txt 中配置）
cd deploy/ncnn && build.bat
```

C++ 推理模型输入 `in0`（1×3×512×512 RGB /255），输出 `out0`（6 类 logits）、
`out1`（4×64×64 热力图）。

## 已知问题 / 注意事项

- `best.pt` 可能不是最优权重：当前验证器返回恒定 fitness，建议统一使用 `last.pt`。
- 显式设置 `net.opt.num_threads = 0` 在部分 ncnn 版本会在加载模型时崩溃，请使用正整数。
- MSVC 在含中文路径下编译会报 LNK1201（PDB 写失败），请在纯英文路径下构建。
- 数据配置 `doc_4corners_heatmap.yaml` 中的路径为本机绝对路径，需按实际数据位置修改。
