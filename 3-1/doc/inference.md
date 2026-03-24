# Inference Guide

本工程当前只保留两条推理链路：

- PyTorch CPU / GPU 推理
- NCNN 推理

## PyTorch 推理

- 后端参数：`--backend torch`
- 设备参数：`--device cpu` 或 `--device cuda:0`
- 模式：`--mode tile` 或 `--mode whole`
- 当前随包默认模型：`Code/weights/uhdm_lite_s_best.pt`

## NCNN 推理

- 后端参数：`--backend ncnn`
- 仅支持 `tile` 模式
- 默认提供 FP16 和 hybrid INT8 两套模型
- 默认推理参数：`--tile 512 --tile_overlap 128`
