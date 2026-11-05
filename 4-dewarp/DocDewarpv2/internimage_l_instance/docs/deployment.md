# 环境、权重与部署

默认权重为 `checkpoints/best_hd95_1024.pth`，完整复制自原始 1024 训练的
`Instance/Train_InternImage-L-Instance/doc_instance_seg/checkpoints/best_hd95.pth`。
原始训练配置与结果位于该训练工程的 `outputs/train/`，保存的 `INPUT.IMAGE_SIZE` 为 1024。
选模指标为单实例验证集 P95 HD95=31.773448，checkpoint iteration=19794。
来源与文件校验值见 `../checkpoints/model_1024.json`。

保留 `criterion.empty_weight` 这一兼容缓冲区以直接加载既有 checkpoint，
部署模型不包含损失计算与训练分支。

已用环境基线为 Python 3.9.23、PyTorch 1.11.0、torchvision 0.12.0、CUDA 11.3、
Detectron2 0.6、MMCV 1.5.0、MMSegmentation 0.27.0、Pillow 9.5.0。
Detectron2 是安装到 Python 环境的依赖；复制本部署目录到其他机器时需另行安装。
安装 Python 依赖后，DCNv3 和 MultiScaleDeformableAttention 需按目标环境编译：

```bash
python -m pip install ./instance_seg/models/ops_dcnv3 --no-build-isolation
python -m pip install ./instance_seg/models/maskdino/modeling/pixel_decoder/ops --no-build-isolation
```

本机现有环境已安装两个扩展。若复用 CUDA 11.3 在 RTX 4090 上构建，原工程使用
GCC/G++ 9、`TORCH_CUDA_ARCH_LIST="8.6+PTX"`，nvcc 加
`-U__SIZEOF_INT128__`；其他平台应匹配各自的 PyTorch/CUDA 编译环境。

## 输出控制

`infer.py` 顶部的 `OUTPUT_MODE` 可设置为 `debug` 或 `production`。
`debug` 保存叠加可视化和 ROI 诊断；`production` 只生成原图分辨率的
单通道 uint8 PNG，像素取值为 0 和 255。命令行同名参数可覆盖 IDE 设置。

## 1024 恢复验证（2026-09-14）

6 项部署回归检查通过，覆盖 debug/production 及 IDE 默认值的命令行覆盖。
RTX 4090 使用原始 1024 权重处理 `perspective_08.jpg`，全图/ROI 输入均为 1024，
严格加载通过，ROI 接受 1/1。production 仅生成 4208×3120 单通道 uint8 PNG，
像素值为 0 和 255，未生成叠加图或 ROI 对比图。
