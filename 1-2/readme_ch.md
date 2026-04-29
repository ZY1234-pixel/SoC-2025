# 希伯来语文本识别

## 训练流程

### 安装paddleOCR
安装方式请参考PaddleOCR安装:https://www.paddleocr.ai/main/version3.x/installation.html

### 数据集构建
数据集为通过Text-renderer生成的30万自建数据集。关于数据格式介绍，可以参考Paddle文本识别任务模块数据集示例：https://www.paddleocr.ai/latest/version3.x/module_usage/text_recognition.html#41

下载示例数据集：
```shell
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar
tar -xf ocr_rec_dataset_examples.tar
```
PS：自建数据集在百度网盘提供。

### 模型训练
在确保对数据集进行校验之后，采用单卡方式进行训练
```shell
python tools/train.py -c configs/rec/hebrew_rec.yml \
   -o Global.pretrained_model=./output/hebrew150_30000_rec_v2/best_accuracy
```

### 模型评估
对模型的训练结果进行评估
```shell
python tools/eval.py -c configs/rec/hebrew_rec.yml \
   -o Global.pretrained_model=./output/hebrew150_30000_rec_v2/best_accuracy
```

### 模型测试
单图测试
```shell
python test.py
```