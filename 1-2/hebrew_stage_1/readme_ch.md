# 希伯来语文本识别

## 训练流程

### 安装paddleOCR
安装方式请参考PaddleOCR安装:https://www.paddleocr.ai/main/version3.x/installation.html

### 数据集构建
数据集为通过Text-renderer生成的10万自建数据集。关于数据格式介绍，可以参考Paddle文本识别任务模块数据集示例：https://www.paddleocr.ai/latest/version3.x/module_usage/text_recognition.html#41

下载示例数据集：
```shell
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar
tar -xf ocr_rec_dataset_examples.tar
```
PS：自建数据集在百度网盘提供。

### 模型训练
在确保对数据集进行校验之后，采用单卡方式进行训练
```shell
python tools/train.py -c configs/rec/hebrew_rec_optimized.yml
```

### 模型评估
对模型的训练结果进行评估
```shell
python tools/eval.py \
  -c configs/rec/hebrew_rec_optimized.yml \
  -o Global.checkpoints=./output/hebrew_stage1_rec/best_accuracy
```

### 模型测试
单行文本图像测试
```shell
python tools/infer_rec.py \
  -c configs/rec/hebrew_rec_optimized.yml \
  -o Global.checkpoints=./output/hebrew_stage1_rec/best_accuracy \
     Global.infer_img=./hebrew_data/test/images/000000000.jpg
```


单张文档图像测试

脚本模式
```shell
python test.py
```

或者采用命令行形式，检测模型使用PP-OCRv5默认模型
```shell
#####下载检测模型
wget -nc https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar
tar -xf PP-OCRv5_server_det_infer.tar

#####单张图像识别
python tools/infer/predict_system.py \
  --image_dir="./hebrew_data/test/hebrew.jpg" \
  --det_model_dir="./PP-OCRv5_server_det_infer" \
  --rec_model_dir="./hebrew_infer/hebrew_stage1_rec_infer" \
  --rec_char_dict_path="./hebrew_data/hebrew_mixed/dict.txt" \
  --use_angle_cls=false \
  --drop_score=0.0 \
  --draw_img_save_dir="./output/hebrew_system_test"
```
