# D2Dewarp

**D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping**

本项目是基于原始论文代码仓库[D2Dewarp官方代码](https://github.com/xiaomore/D2Dewarp) PyTorch 推理实现。

---

## 1. 项目结构

```text
D2Dewarp/
│
├── predict.py                  # 推理脚本
├── requirements.txt            # 依赖环境
├── model_weight.pt             # 预训练模型（需要下载）
│
├── networks/
│   └── d2dewarp_model.py       # 网络结构
│
├── loader/
│   └── dataset_doc3d_grid_HV.py
│
├── images/                     # 输入图像目录
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
│
├── results/                      # 输出结果目录
│   ├── dewarp/                 # 校正结果
│   ├── pred_h/                # H-line预测
│   └── pred_v/                # V-line预测
│
└── README.md
```

## 2.环境配置
```commandline
pip install -r requirements.txt
```

## 3.模型下载
[下载地址](https://pan.baidu.com/share/init?surl=qnHjy3-ANlrrgjc7hac3Bg&pwd=2cut)
将模型放入根目录，测试图片放入images文件夹

## 4.推理脚本
### 4.1 直接右键运行`predict.py`
```shell
python predict.py
```
### 4.2 命令行运行
```shell
python predict.py \
--img_path images \
--save_path results \
--model_path model_weight.pt
```
### 4.3 参数说明
predict.py 支持以下命令行参数：

| 参数 | 类型 | 默认值               | 说明 |
|------|------|-------------------|------|
| `--input_size` | int | 448               | 模型输入图像尺寸 |
| `--model_path` | str | `model_weight.pt` | 预训练模型路径 |
| `--img_path` | str | `images/`         | 输入图像文件夹或单张图片路径 |
| `--save_path` | str | `results/`        | 输出结果保存路径 |
| `--hv_out_chans` | int | 1                 | H-line / V-line 输出通道数 |
| `--d_model` | int | 448               | 主干网络特征维度 |
| `--in_chans` | int | 4                 | 输入通道数（RGB + Edge） |

