# PP-OCRv6 RTL 多语言识别

## 训练目标
本次训练基于 PaddleOCR `release/3.7` 源码和 `PP-OCRv6_small_rec` 识别模型，对 2 种从右到左书写的语言（阿拉伯语/希伯来语）进行混合微调，并适当加入英文、数字和常见标点。当前版本训练的是单行识别模型，每张训练图片只包含一行文字。

2 种语言如下：
- 阿拉伯语：`ar`
- 希伯来语：`he`

本版重点解决 RTL 语言与英文、数字混排时的训练标签顺序和推理后处理问题。训练阶段标签采用适合 PaddleOCR 从左到右学习图像特征的视觉顺序；测试阶段再通过后处理将识别结果转换为与 `label.txt` 一致的逻辑文本顺序。



## 环境安装
### 创建虚拟环境
为了不影响原有 PaddleOCR 环境，重新创建 `paddleocr37` 虚拟环境：
```bash
conda create -n paddleocr37 python=3.9 -y
conda activate paddleocr37
python -m pip install -U pip setuptools wheel
```

### 安装 PaddlePaddle 3.1.1
如果环境中已经装过其他 Paddle 版本，先卸载：
```bash
python -m pip uninstall -y paddlepaddle paddlepaddle-gpu
```

GPU 环境安装 PaddlePaddle 3.1.1：
```bash
python -m pip install paddlepaddle-gpu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

### 准备 PaddleOCR release/3.7 源码
创建 release/3.7 源码目录：
```bash
git -C PaddleOCR fetch origin
git -C PaddleOCR worktree add -b release37-ft PaddleOCR-release37 origin/release/3.7
```

安装 PaddleOCR 训练依赖：
```bash
cd PaddleOCR-release37
python -m pip install -r requirements.txt
python -m pip install python-bidi
```

安装完成后检查版本：
```bash
python -c "import paddle; print(paddle.__version__); print(paddle.device.is_compiled_with_cuda()); print(paddle.device.get_device())"
```



## 数据集构建
数据集通过 `text-renderer` 生成，语料主要来自 Wikimedia `pages-articles` dump。训练内容包含阿拉伯语、希伯来语、常见英文块、数字和常见标点，单行文本长度不超过 40 字符。

本版混合数据集目录为：
```text
data/mixed2_data/
├── train.txt
├── val.txt
├── dict.txt
├── metadata.json
└── images/
```

数据集命名格式：
```text
images/train_word_1.jpg
images/train_word_2.jpg
...
images/val_word_1.jpg
images/val_word_2.jpg
...
```

数据集规模：
```text
train.txt: 100000
val.txt:    10000
dict.txt:     172
```

语言比例：
```text
训练集：阿拉伯语 50000，希伯来语 50000
验证集：阿拉伯语  5000，希伯来语  5000
```



## 模型训练
进入 release/3.7 源码目录并激活环境：
```bash
cd /home/ww/ww/whu/project/PaddleOCR-release37
conda activate paddleocr37
```

单卡训练：
```bash
python tools/train.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_mixed2_hebrewstyle_ar_finetune.yml
```

如需指定 GPU：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_mixed2_hebrewstyle_ar_finetune.yml
```

本次训练输出目录：
```text
output/mixed2_small_rec_hebrewstyle_ar/
```

当前最优模型结果：
```text
best_epoch: 38
acc: 0.9846999990153
norm_edit_dis: 0.9991187305900626
fps: 1261.0641413297285
```



## 模型评估
使用最优权重评估验证集：
```bash
python tools/eval.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_mixed2_hebrewstyle_ar_finetune.yml \
  -o Global.checkpoints=./output/mixed2_small_rec_hebrewstyle_ar/best_accuracy
```

也可以使用 `best_model/model`：
```bash
python tools/eval.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_mixed2_hebrewstyle_ar_finetune.yml \
  -o Global.checkpoints=./output/mixed2_small_rec_hebrewstyle_ar/best_model/model
```



## 导出推理模型
将训练得到的最优模型导出成 PaddleOCR 推理模型：
```bash
python tools/export_model.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_mixed2_hebrewstyle_ar_finetune.yml \
  -o Global.pretrained_model=./output/mixed2_small_rec_hebrewstyle_ar/best_accuracy \
     Global.save_inference_dir=./output/mixed2_small_rec_hebrewstyle_ar/inference_model
```

导出后的推理模型目录：
```text
output/mixed2_small_rec_hebrewstyle_ar/inference_model/
├── inference.json
├── inference.pdiparams
└── inference.yml
```



## 推理模型测试
RTL 混合模型建议使用后处理脚本测试：
```text
tools/infer/predict_rec_rtl.py
```

该脚本会将模型输出的视觉顺序转换为与 `label.txt` 一致的逻辑文本顺序，并输出识别结果、置信度和召回率。

测试 `test-images/single2_en` 文件夹中的单行图片：
```bash
python tools/infer/predict_rec_rtl.py \
  --use_gpu True \
  --rec_model_dir ./output/mixed2_small_rec_hebrewstyle_ar/inference_model \
  --rec_char_dict_path ./data/mixed2_hebrewstyle_ar/dict.txt \
  --image_dir ./test-images/single2_en \
  --label_path ./test-images/single2_en/label.txt \
  --save_res_path ./output/mixed2_small_rec_hebrewstyle_ar/predicts_single2_en_8imgs.txt \
  --terminal_order logical
```

测试单张图片：
```bash
python tools/infer/predict_rec_rtl.py \
  --use_gpu True \
  --rec_model_dir ./output/mixed2_small_rec_hebrewstyle_ar/inference_model \
  --rec_char_dict_path ./data/mixed2_hebrewstyle_ar/dict.txt \
  --image_dir ./test-images/single2_en/ar_en_1.png \
  --terminal_order logical
```

结果文件字段说明：
```text
image: 图片路径
predict: 后处理后的逻辑顺序预测文本
label: label.txt 中的真实标签
recall: 字符级召回率
confidence: PaddleOCR 输出的识别置信度
```



## RTL 后处理说明
训练时，阿拉伯语和希伯来语标签采用视觉顺序，以匹配 PaddleOCR 从左到右读取图像特征的学习方式；推理时需要将模型输出转换回逻辑顺序，才能和 `label.txt`、真实文本语义保持一致。

当前后处理只调整顺序和通用空格/标点贴合关系，不修改模型识别出的字符本身。

终端显示 RTL 文本时，英文路径、括号、引号和置信度数字可能影响视觉排版，因此最终结果应以保存的 txt 文件中 `predict` 字段为准。



## 整图测试说明
当前训练的是识别模型，只负责对已经裁切好的单行文字图片进行识别。如果要测试完整文档图片，需要额外搭配检测模型，将整图先检测裁切成文本行，再调用本次导出的识别模型和 RTL 后处理逻辑。

示例命令：
```bash
python tools/infer/predict_system.py \
  --image_dir /path/to/document.jpg \
  --det_model_dir /path/to/det_model \
  --rec_model_dir ./output/mixed2_small_rec_hebrewstyle_ar/inference_model \
  --use_angle_cls false \
  --drop_score 0.0 \
  --draw_img_save_dir ./output/mixed2_small_system_test
```



## 注意事项
- 本版数据集只包含阿拉伯语和希伯来语，外加适量英文、数字和常见标点。
- 当前训练图片为单行图；整图文档测试需要先检测裁切文本行。
- `predict` 字段为逻辑文本顺序，适合保存、评估和与 `label.txt` 对比。
- `visual` 或模型原始输出只适合排查训练标签顺序，不建议作为最终识别文本。
- `use_space_char` 当前为 `false`，如果后续希望模型显式学习空格，需要同步调整字典、配置和训练数据。
