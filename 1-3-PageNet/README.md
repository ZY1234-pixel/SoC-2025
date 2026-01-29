## 中文手写体识别算法

### 背景

中文手写体识别算法是**模式识别**与**自然语言处理**交叉领域的核心研究方向，聚焦于将人类手写的中文汉字、词语、句子等视觉信息转化为机器可识别的数字文本，其发展源于**人机交互的核心需求**，并伴随计算机技术、人工智能、深度学习的迭代不断演进，同时因中文自身的字形复杂性，成为手写体识别领域的典型难题和研究重点。

### 开始

#### 1.环境配置

```powershell
conda create -n "环境名" python=3.8
conda activate "环境名"
pip install -r requirements.txt
```

#### 2.下载权重文件
通过网盘分享的文件：checkpoints链接: https://pan.baidu.com/s/1GBOIk7ARfDPuoO2u7CCw8A?pwd=9taz 提取码: 9taz 

#### 3.运行代码

```sh
python infer.py
```