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
通过网盘分享的文件：checkpoints
链接: https://pan.baidu.com/s/13VYuso9cu-wkXwOOMTP-eA?pwd=2c8z 提取码: 2c8z 
--来自百度网盘超级会员v5的分享

#### 3. 下载数据集
通过网盘分享的文件：datasets
链接: https://pan.baidu.com/s/1m-BK-x_Z2TjBKLL8ryjTVA?pwd=wkcm 提取码: wkcm 
--来自百度网盘超级会员v5的分享

数据集下载后按照如下文件结构:
```
datasets
├─IC13Comp
├─MTHv2_test
└─raw
   └─SCUT-HCCDoc
      │  hccdoc_test.json
      │  hccdoc_train.json
      └─image
```
运行下述代码，将SCUT-HCCDoc dataset转换为lmdb格式。
```
python tools/convert_hccdoc_to_lmdb.py \
  --image_root datasets/raw/SCUT-HCCDoc/image/ \
  --annotation_file datasets/raw/SCUT-HCCDoc/hccdoc_test.json \
  --dict_path dicts/scut-hccdoc.txt \
  --lmdb_root datasets/SCUT-HCCDoc_test
```

#### 4.运行代码

```sh
python infer.py
```

### 相关资料
#### 模型结构及原理介绍
通过网盘分享的文件：Docs
链接: https://pan.baidu.com/s/1wbpdOlZusonyNTmGkncnjw?pwd=f2w1 提取码: f2w1 
--来自百度网盘超级会员v5的分享
