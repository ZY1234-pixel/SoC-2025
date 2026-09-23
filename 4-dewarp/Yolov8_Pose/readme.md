# Yolov8-Pose 四角点检测

从拍摄或扫描的图片里检测文档（书本、纸张、证件卡等）的 4 个角点，供下游畸变矫正、
色彩纹理修复等使用。模型为 YOLOv8s-pose 变体，输入 640。

## 版本记录

| 版本 | 时间 | 新增/变化 |
|---|---|---|
| 初版 | 2026-05 | `inference_intranet.py` 推理封装（内网服务用）＋ 端侧 C++/NCNN 部署 |
| **0916** | **2026-09-16** | 新增 `predict.py`（推理）/ `corner_postprocess.py`（后处理）/ `evaluate.py`（评测）；输出口径统一为**"可见区域四边形"**；新增 `patch_env.py` 打环境补丁 |
| **0924** | **2026-09-24** | **int8 量化优化**（见第四节）：修正 3×3 stride=1 卷积的权重尺度、校准保持 absmax、按敏感度挑层做混合精度、QAT 微调；输入口径明确为"补方 letterbox + INTER_AREA" |

初版内容保留在本文档后半部分（"初版说明"两节），0916 的用法以本文档前半部分为准。

---

## 一、文件

### 1.1 推理与评测（0916，2026-09-16）

| 文件 | 说明 |
|---|---|
| `predict.py` | 推理：单图/目录，输出"可见区域四边形"、标注图与 `results.json` |
| `corner_postprocess.py` | 后处理核心：可见区域口径（裁剪 / 最大内接四边形）＋ 边拟合精修 |
| `evaluate.py` | 精度评测：对含 `images/`、`labels/` 的数据集统计点集误差 |
| `corner_order.py` | 角点规范排序（TL→TR→BR→BL），被上面两个脚本调用 |
| `patch_env.py` | 为 ultralytics 打 5 处补丁（**推理必须打第 4 处**，见 3.3） |
| `requirements.txt` | 依赖版本（ultralytics 8.4.48 / torch 2.7.1 等） |

### 1.2 初版部署（2026-05）

| 文件 | 说明 |
|---|---|
| `inference_intranet.py` | Python 推理脚本（封装了核心检测类，内网服务用） |
| `ncnn_pose/main.cpp` | 端侧 C++ 推理源码 |
| `ncnn_pose/CMakeLists.txt` | 端侧 C++ 编译配置文件 |

两种推理入口的关系：`inference_intranet.py` 是内网封装版；`predict.py` 是命令行版，带后处理开关，
便于评测与排查。两者共用同一份权重与同一套环境补丁，**输出口径以 `predict.py` 为准**。

## 二、快速开始（0916）

```bash
# 0) 依赖（建议 python 3.10 + CUDA 环境；其余见 requirements.txt）
pip install -r requirements.txt

# 1) 打环境补丁（在目标 venv 里跑；也可 --venv <venv根目录> 指定）
python patch_env.py --check
python patch_env.py --apply

# 2) 放入权重：best.pt 放到本目录 weights/ 下

# 3) 推理
python predict.py --source <图片或目录> --out outputs/infer

# 4) 有标注数据集的精度评测
python evaluate.py --ds <含 images/labels 的数据集目录>
```

---

## 三、0916 说明

### 3.1 输出口径：可见区域四边形

文档被拍摄时可能有一部分落在画面外（缺边缺角）。本版本**统一输出"可见区域"的四边形**：

- 只有**边**被裁掉 → 输出目标与画面矩形的交集，角点落在画面边框上；
- **角**被裁掉（交集是五边形以上）→ 取面积最大的内接四边形，结果里标记 `corner_cut=True`；
- 四个角都在画面内 → 原样输出。

### 3.2 类别输出

模型同时输出目标类别与置信度，与角点一一对应（每条检测含 `cls` / `cls_name` / `conf`）。
`predict.py` 的 `results.json` 顶层带 `names` 字段给出完整类别表：

| id | 类别 |
|---|---|
| 0 | double_page_book（双页书） |
| 1 | newspaper_poster（报纸/海报） |
| 2 | receipt（小票） |
| 3 | screen（屏幕） |
| 4 | single_page（单页文档） |
| 5 | unclassified（未分类） |
| 6 | id_card（证件卡） |

`evaluate.py` 会一并给出**类别准确率**与误判明细（如 `single_page->id_card×2`）。

### 3.3 后处理：边拟合精修

`corner_postprocess.py` 用图像梯度把预测四边形的四条边重新拟合再求交点，用于修正
"缺边时画面内的角点被一起拉偏"的问题。带两重保护：

1. **门控**：只有相邻两条边都拟合可靠（内点数达标）才替换该角点，否则保持模型原始预测；
2. **位移守卫**：角点相对原预测位移超过目标短边的 5% 时退回原值。

可用 `predict.py --no-snap` 关闭做对照。

### 3.4 环境补丁（重要）

ultralytics 需要 5 处补丁，`patch_env.py` 会自检并补齐（幂等，改前备份 `.bak_release`）：

| # | 位置 | 作用 | 影响面 |
|---|---|---|---|
| 1 | `data/utils.py` | 放宽关键点坐标校验（允许负值/越界到 ±999） | 数据读取 |
| 2 | `data/augment.py` | 增强后不把画外角点置为不可见 | 训练 |
| 3 | `utils/instance.py` | `Instances.clip` 新增 `clip_keypoints` 开关 | 训练 |
| 4 | `utils/ops.py` + `models/yolo/pose/predict.py` | `scale_coords(..., clip=False)` | **推理必须**，否则画面外的预测点被夹回画面 |
| 5 | `utils/loss.py` | `KeypointLoss` 顺序无关（4! 排列取最小） | 训练 |

版本基线：ultralytics 8.4.48、torch 2.7.1+cu118。`patch_env.py` 按精确文本替换，
换 ultralytics 版本后若文本不匹配，需按上表手动改。

### 3.5 评测说明

`evaluate.py` 的指标：

- **点集误差**：对 4 个角点的所有排列取最小的平均距离，不含"角点角色（TL/TR/BR/BL）是否对应"的惩罚，
  反映四边形本身准不准；若下游对角色敏感，需改用逐通道误差；
- 角点分两类统计：**画面内角点**（真实观测到的角）与**贴边角点**（裁切产生，落在画面边框上）。

数据集需为 YOLO-pose 标签格式（`images/` 与 `labels/` 同名配对）。

---

## 四、0924 int8 量化（2026-09-24）

端侧要跑 int8 才够快，压缩会带来角点偏差。本版**不改推理接口**，只针对 int8 精度做了一轮优化，
并把输入口径钉死。

### 4.1 做了哪些优化

| # | 优化 | 说明 |
|---|---|---|
| 1 | **修正 3×3 stride=1 卷积的权重尺度** | ncnn2table 对这类层（winograd 候选）写权重尺度时只用 31 级，其余层用 127 级；而 arm64 开 dotprod 时运行时会关掉 int8 winograd，等于白丢精度。按"量程不变、31 级换 127 级"重算这些层的权重尺度行，**其余行逐字节保留** |
| 2 | **校准方法保持 absmax** | 试过 ACIQ（权重、激活两种组合），未采用；交付表沿用 absmax 口径 |
| 3 | **混合精度：按敏感度挑层保 fp32** | 逐层做敏感度扫描，挑出最敏感的少数层保 fp32，替代"末尾 N 层"的一刀切；kpt 末层与 DFL 层（无偏置）恒保 fp32 |
| 4 | **QAT 微调** | 训练时复刻 ncnn 部署方案（权重逐通道对称、激活逐张量对称、钳位 ±127、STE 直通），并**周期性用校准集刷新激活尺度**；训练中敏感层全程保 fp32 |
| 5 | **统一输入口径** | 保持纵横比 + pad 114 补到正方形；缩放用 **INTER_AREA**；RGB、除以 255。校准表按同一口径生成 |

### 4.2 交付的 int8 模型

两版混合精度（**512 输入**；与权重一样**单独分发**，不入库）：

* **A 版**：6 层保 fp32 —— `conv_11/13/23/43/62/70` + kpt 头 `conv_65/68/71` + DFL `conv_72`
* **B 版**：A 版再加 `conv_61/69/42/44`

配套：`model.table`（校准表）与 `calib_list.txt`（校准图清单）。两版都附带同一份 fp32 基线，
便于做"int8 vs fp32"的漂移对照。

### 4.3 端侧接入注意

1. **预处理必须与交付口径一致**（补方 + INTER_AREA）。若端侧用的是 ncnn 自带的双线性缩放，
   需要按实际缩放重建校准表，否则表与实际推理不匹配。
2. **解码顺序**：导出图里 `out0 = [box(4), cls(7), kpt(12)] × N`，kpt 段与另一输出（`blob 276`）
   的 anchor 排列需按端侧解码实现对齐；跨张量按 index 直接配对会取到不同目标。
3. **`conv_72`（DFL 的 1×1 卷积，无偏置）必须留 fp32**：部分 ncnn 工具版本对无偏置卷积做量化会崩。

---

## 五、初版说明（2026-05，保留）

### 4.1 服务器端 (Python)

1. 确保 `best.torchscript` 与测试图片准备就绪。
2. 修改 `inference_intranet.py` 底部 `__main__` 中的路径参数。
3. 执行：`python inference_intranet.py`

> **tips**：Windows 环境下 PyTorch C++ 底层对中文路径支持存在 Bug。本脚本已内置自动映射系统纯英文临时目录的修复逻辑。

### 4.2 端侧 (C++ / NCNN)

**环境依赖：** CMake 3.12+, NCNN, OpenCV (需包含 Imgproc 模块)

1. 在代码目录下创建 `build` 文件夹并进入。
2. 执行 CMake 配置（请按实际情况替换路径）：
   `cmake .. -DNCNN_DIR="/path/to/ncnn" -DOpenCV_DIR="/path/to/opencv"`
3. 编译可执行程序：`cmake --build . --config Release`
4. 运行前，请确保 NCNN 模型参数文件以及依赖的 `.dll` 动态库与生成的可执行文件在同一目录下。
