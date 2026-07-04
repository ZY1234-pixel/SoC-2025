# 文本行分割

**基于UNet的文本行分割（水平/竖直线条）**

本项目是基于原始代码仓库[unet代码](https://github.com/bubbliiiing/unet-pytorch) PyTorch 改进推理实现。

---

## 1. 项目结构

```text
text-segmentation/
│
├── predict.py                  # 推理脚本
├── requirements.txt            # 依赖环境
├── train.py             # 训练脚本
│
├── nets/
│   └── unet.py       # 网络结构
│
├── logs/
│   └── xxx.pt                  # 推理模型
│
├── img/                     # 输入图像目录
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
│
├── img_out/                      # 输出结果目录
│   ├── xxx.jpg                 
│   ├── xxx.jpg                
│   └── xxx.jpg                
│
└── README.md
```

## 2.环境配置
```commandline
pip install -r requirements.txt
```

## 3.模型下载
[下载地址](链接: https://pan.baidu.com/s/1qvKlLAGhf35z6OqJcDJEjw?pwd=wfmi)
将模型放入logs目录，测试图片放入img文件夹

## 4.推理脚本
### 4.1 直接右键运行`predict.py`
```shell
python predict.py
```
更多参数见predict.py，模型参数目录见unet.py
