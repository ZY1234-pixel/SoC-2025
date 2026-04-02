# ESDNet-Lite

面向测试交付整理后的 ESDNet-Lite 工程。

当前版本只保留以下内容：

- PyTorch CPU 推理
- PyTorch GPU 推理
- NCNN 推理
- TensorRT FP16 推理
- UHDM 数据格式训练代码

## 目录结构

- `dataset/`: 数据目录说明
- `Code/`: 核心代码、配置和模型文件
- `test-result/`: 推理结果输出目录
- `doc/`: 补充说明文档
- `imagelist.txt`: 真实校准图片列表
- `image_list.txt`: `imagelist.txt` 的同内容别名
- `release_note.txt`: 本次交付整理说明

## Code 目录

- `model.py`: 模型入口
- `train.py`: 训练脚本
- `test.py`: 推理脚本
- `dataset.py`: UHDM 数据读取
- `utils.py`: 公共工具函数
- `preprocess.py`: 图像预处理与保存
- `esdnet_lite_uhdm.yaml`: 训练配置模板
- `requirement.txt`: 依赖说明
- `runcmd.txt`: 常用命令
- `weights/`: 当前随包提供的模型文件

## 环境

安装基础依赖：

```bash
pip install -r Code/requirement.txt
```

按实际环境额外安装：

- `torch`
- `torchvision`
- `ncnn`，仅 Python NCNN 推理需要
- `onnx`
- `TensorRT 8.6 GA Python bindings`，仅 TensorRT 推理需要

## 当前模型文件

位于 `Code/weights/`：

- `uhdm_lite_s_best.pt`
- `uhdm_lite_s_best.ncnn.param`
- `uhdm_lite_s_best.ncnn.bin`
- `uhdm_lite_s_best_hybrid_int8.ncnn.param`
- `uhdm_lite_s_best_hybrid_int8.ncnn.bin`
- `uhdm_lite_s_best_hybrid.table`

TensorRT engine 默认会按以下顺序自动查找：

- `Code/weights/*.plan`
- `../NAFNet/output_model/*.plan`

## 常用命令

PyTorch CPU 推理：

```bash
python Code/test.py \
  --backend torch \
  --preset lite-s \
  --model_path Code/weights/uhdm_lite_s_best.pt \
  --device cpu \
  --mode tile \
  --input_dir dataset/input \
  --output_dir test-result/torch_cpu \
  --tile 512 --tile_overlap 128
```

PyTorch GPU 推理：

```bash
python Code/test.py \
  --backend torch \
  --preset lite-s \
  --model_path Code/weights/uhdm_lite_s_best.pt \
  --device cuda:0 \
  --mode tile \
  --input_dir dataset/input \
  --output_dir test-result/torch_gpu \
  --tile 512 --tile_overlap 128
```

NCNN FP16 推理：

```bash
python Code/test.py \
  --backend ncnn \
  --preset lite-s \
  --ncnn_param Code/weights/uhdm_lite_s_best.ncnn.param \
  --ncnn_bin Code/weights/uhdm_lite_s_best.ncnn.bin \
  --input_dir dataset/input \
  --output_dir test-result/ncnn_fp16 \
  --tile 512 --tile_overlap 128
```

NCNN hybrid INT8 推理：

```bash
python Code/test.py \
  --backend ncnn \
  --preset lite-s \
  --ncnn_param Code/weights/uhdm_lite_s_best_hybrid_int8.ncnn.param \
  --ncnn_bin Code/weights/uhdm_lite_s_best_hybrid_int8.ncnn.bin \
  --input_dir dataset/input \
  --output_dir test-result/ncnn_int8 \
  --tile 512 --tile_overlap 128
```

TensorRT FP16 推理：

```bash
conda run -n torch25 python Code/test.py \
  --backend tensorrt \
  --preset lite-s \
  --trt_engine ../NAFNet/output_model/esdnet_lite_fp16.plan \
  --device cuda:0 \
  --input_dir dataset/input \
  --output_dir test-result/tensorrt_fp16 \
  --tile 256 --tile_overlap 64
```

训练：

```bash
python Code/train.py \
  --train_dir /path/to/UHDM/train \
  --val_dir /path/to/UHDM/test \
  --model_preset lite-s \
  --save_dir test-result/train_runs
```

## 说明

- 更详细的推理说明见 `doc/inference.md`
- 更多命令示例见 `Code/runcmd.txt`
- TensorRT 当前仅支持 `tile` 模式，且 `tile` 会自动对齐到 engine 固定输入尺寸
