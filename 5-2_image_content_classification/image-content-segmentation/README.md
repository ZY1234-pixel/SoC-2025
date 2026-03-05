## 图像内容分割算法

### 背景

包含多类别元素的复合图像在文档数字化等领域的应用日益广泛，当前传统图像分割技术在面对复合图像时，常存在类别识别不全面、精准度不足等问题，尤其针对小文字区域等高精度要求目标，易出现误判为背景或噪音的情况，严重影响后续信息提取的准确性。本算法用于将输入的RGB 格式多类别扫描/拍摄图像，分割成背景、人物、文字、网纹/图像内容、线条/边缘等图像对象，并输出多通道类别掩码图像、COCO格式的JSON标注文件。

### 算法流程
1. 加载模型参数
2. 模型推理：算法对每张图像进行串行模型推理：使用UNet，主干网络为StarNet
3. 后处理
4. 保存结果


### 开始

#### 1.环境配置

```powershell
git clone git@github.com:ZY1234-pixel/SoC-2025.git
conda create -n "环境名" python=3.8
conda activate "环境名"
pip install -r requirements.txt
```

#### 2.下载权重文件
[百度网盘](https://pan.baidu.com/s/1dWwtw2c9my-WUoBHiSKwNg?pwd=i8zj)(提取码：i8zj)  

UNet文件夹下的权重文件放入`file/log`文件夹下  

测试图片放入`file/img`文件夹下  

输出结果会保存到`file/img_out`文件夹下
#### 3.运行代码
直接运行`predict.py`
