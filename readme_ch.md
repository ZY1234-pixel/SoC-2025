# 中文手写体文本识别

## 训练流程

### 安装paddleX
安装方式请参考PaddleX安装:https://paddlepaddle.github.io/PaddleX/main/installation/installation.html

### 数据集构建
数据集可以选择自制数据或者参考paddle公开的中文手写体数据集。关于数据格式介绍，可以参考Paddle文本检测/文本识别任务模块数据标注教程：https://paddlepaddle.github.io/PaddleX/main/data_annotations/ocr_modules/text_detection_recognition.html
数据集获取命令：
```shell
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/handwrite_chinese_text_rec.tar -P ./dataset
tar -xf ./dataset/handwrite_chinese_text_rec.tar -C ./dataset/
```
PS：安装完毕后对压缩包进行解压。

数据集校验：
```shell
python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv5_mobile_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.output=./output/check_data \
    -o Global.dataset_dir=./dataset/handwrite_chinese_text_rec
```

### 模型训练
在确保对数据集进行校验之后，训练命令如下：
```shell
python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv5_mobile_rec.yaml \
    -o Global.mode=train \
    -o Global.output=./output/train_0409_v5 \
    -o Global.dataset_dir=./dataset/handwrite_chinese_text_rec
```

### 模型评估
对模型的训练结果进行评估
```shell
python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv5_mobile_rec.yaml \
    -o Global.mode=evaluate \
    -o Global.output=./output/train_0409_v5 \
    -o Global.dataset_dir=./dataset//RealCE-1K \
    -o Evaluate.weight_path=./output/train_0409_v5/best_accuracy/best_model/model.pdparams
```

### 模型测试
单图测试
```shell
python infer.py
```