# HeatmapYolo：书本文档四角点检测（YOLOv8 主干 + 热力图输出头）

基于 YOLOv8s 主干改造的角点检测方案：输出头改为「文档类别分类 + 4 通道关键点热力图」，
热力图在 P3（stride 8）特征上输出，解码时用 argmax + 抛物线亚像素精修，可用于后续的畸变矫正、
色彩纹理修复、增强等下游任务。

## 目录结构

```text
HeatmapYolo/
├── train.py                  # 训练入口（分类 + 热力图联合训练）
├── heatmap_cls_model.py      # 模型定义（HeatmapClsHead / HeatmapClsModel）
├── heatmap_utils.py          # 高斯热力图生成 / 亚像素解码
├── loss_cls_heatmap.py       # 分类 + 热力图损失（逐可见像素 MSE）
├── dataset_cls_heatmap.py    # 数据集封装（强制 pose 任务解析关键点）
├── trainer_cls_heatmap.py    # 训练器（跳过 bbox 验证，只训练/打印损失）
├── predict.py                # 单张图片推理
├── predict_batch.py          # 文件夹批量推理
├── evaluate.py               # val 集指标评估（逐角点误差 / 准确率 / 报告输出）
├── run_infer_samples.py      # 推理可视化（预测点 + GT 叠加）
├── make_sheets.py            # 将推理结果拼成联系表
├── export_torchscript.py     # 导出 TorchScript（部署用）
├── configs/
│   └── yolov8s.yaml          # 模型结构配置
├── doc_4corners_heatmap.yaml # 数据配置（路径按本机修改）
└── deploy/
    └── ncnn/                 # NCNN C++ 推理工程（含 bench 模式）
        ├── CMakeLists.txt
        ├── build.bat
        └── main.cpp
```

权重（`*.pt` / `*.torchscript.pt` / `*.ncnn.param` / `*.ncnn.bin`）不随仓库提供，
训练/推理前请自行准备并放到 `weights/` 目录。

## 环境依赖

见 `requirements.txt`。核心版本：Python 3.11、PyTorch 2.x、ultralytics 8.4.110。

## 训练

```bash
python train.py
```

默认配置（与最终采用的 heatmap_v12_512_aug 一致）：

- 输入 512×512，batch 32，100 epochs，SGD（lr0=0.005，warmup 3 epochs）
- 增强：旋转 10°、透视 0.001、缩放 0.1、平移 0.05、水平翻转 0.5
  （角点被增强移出画面时自动标记不可见并从 loss 中屏蔽）
- 关键修复项：
  - `warmup_bias_lr=0.0`：默认 0.1 会让热力图头里的 BN/偏置参数在 warmup 阶段被打爆，
    导致通道塌缩（表现为某个/全部角点输出全 0）。这是必须保留的配置。
  - `heatmap_weight=40960`：loss 已修正为逐可见像素 MSE，该值 = 10 × (imgsz/8)²，
    保持与原权重 10 相同的梯度权重。

## 推理

```bash
# 单张
python predict.py --image <图片路径> --weights weights/heatmap_v12_512_aug/last.pt
# 批量
python predict_batch.py --input_dir <图片目录> --weights weights/heatmap_v12_512_aug/last.pt
```

推理预处理与训练一致：letterbox 到 512 + RGB + /255。

## 评估

```bash
# 单模型
python evaluate.py --weights weights/heatmap_v12_512_aug/last.pt
# 新旧对比
python evaluate.py --compare <旧权重> <新权重>
```

输出指标：逐角点 mean/median/p90/p95 像素误差、≤1/2/3/5/10px 准确率、
上/下角分组、按类别分组、通道健康检查，以及 `val_metrics.csv/json`、逐图误差表、误差直方图。

参考基线（val 637 张，heatmap_v12_512_aug，512 输入）：

| 角点 | median | mean | ≤3px |
|------|--------|------|------|
| kpt0 左上 | 2.69px | 6.58px | 52.6% |
| kpt1 右上 | 2.54px | 5.73px | 54.3% |
| kpt2 右下 | 2.30px | 8.25px | 57.3% |
| kpt3 左下 | 2.80px | 9.78px | 51.8% |

## 部署（NCNN）

```bash
# 1. 导出 TorchScript
python export_torchscript.py --weights weights/heatmap_v12_512_aug/last.pt
# 2. 用 pnnx 转 ncnn（示例）
pnnx.exe weights/heatmap_v12_512_aug/heatmap_v12_512_aug.torchscript.pt inputshape="1,3,512,512"
# 3. 编译 C++ 工程（MSVC，需 ncnn + OpenCV，路径在 CMakeLists.txt 中配置）
cd deploy/ncnn && build.bat
```

C++ 推理模型输入 `in0`（1×3×512×512 RGB /255），输出 `out0`（6 类 logits）、
`out1`（4×64×64 热力图）。可执行文件支持：

```bash
ncnn_corner_infer.exe <图片或文件夹> [输出目录] [label目录]   # 推理
ncnn_corner_infer.exe --bench <图片> <迭代次数> <线程数>       # 性能测试
```

## 已知问题 / 注意事项

- `best.pt` 可能不是最优权重：当前验证器返回恒定 fitness，建议统一使用 `last.pt`。
- 显式设置 `net.opt.num_threads = 0` 在部分 ncnn 版本会在加载模型时崩溃，请使用正整数
  （本项目实测 4 线程最快，单张约 17~21ms）。
- MSVC 在含中文的路径下编译会报 LNK1201（PDB 写失败），请在纯英文路径下构建。
- 数据配置 `doc_4corners_heatmap.yaml` 中的路径为本机绝对路径，需按实际数据位置修改。
