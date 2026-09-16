# Yolov8-Pose 四角点检测

从拍摄或扫描的图片里检测文档（书本、纸张、证件卡等）的 4 个角点，供下游畸变矫正、
色彩纹理修复等使用。模型为 YOLOv8s-pose 变体，输入 640。

## 版本记录

| 版本 | 时间 | 新增/变化 |
|---|---|---|
| 初版 | 2026-05 | `inference_intranet.py` 推理封装（内网服务用）＋ 端侧 C++/NCNN 部署 |
| **0916** | **2026-09-16** | 新增 `predict.py`（推理）/ `corner_postprocess.py`（后处理）/ `evaluate.py`（评测）；输出口径统一为**"可见区域四边形"**；新增 `patch_env.py` 打环境补丁 |

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

## 四、初版说明（2026-05，保留）

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
