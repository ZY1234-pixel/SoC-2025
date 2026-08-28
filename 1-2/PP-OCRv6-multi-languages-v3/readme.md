# PP-OCRv6 多语言识别

## 训练目标
本次训练基于 PaddleOCR `release/3.7` 源码和 `PP-OCRv6_small_rec` 识别模型，对 7 种语言进行混合微调，并适当加入英文、数字、常见标点和真实切片。当前版本训练的是单行识别模型，每张训练图片只包含一行文字，最长文本不超过 40 字符，同时开启空格建模以缓解空格识别错误。

7 种语言如下：
- 韩语：`ko`
- 泰语：`th`
- 俄语：`ru`
- 保加利亚语：`bg`
- 乌克兰语：`uk`
- 哈萨克语：`kk`
- 希腊语：`el`


## 环境安装
### 创建虚拟环境
为了不影响原有 PaddleOCR 环境，重新创建 `paddleocr37` 虚拟环境：
```bash
conda create -n paddleocr37 python=3.9 -y
conda activate paddleocr37
python -m pip install -U pip setuptools wheel
```

### 重新安装 PaddlePaddle 3.1.1
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
```

安装完成后检查版本：
```bash
python -c "import paddle; print(paddle.__version__); print(paddle.device.is_compiled_with_cuda()); print(paddle.device.get_device())"
```


## 数据集构建
本版训练数据集为 `ocr_7lang_richbg_someocr_mix_300k`，总量 33 万张，其中训练集 30 万张，验证集 3 万张。

数据来源与构成：
- 以 `mixed7_data_punct_enmix_richbg` 为基础合成数据
- 混入 `some_OCR` 中筛选后的真实切片
- 训练集切片与合成数据比例约为 1:10
- 验证集切片与合成数据比例约为 1:5
- 尽量保持 7 种语言数量接近

文本规则：
- 每张图像只保留单行文本
- 最长文本（含空格）不超过 40 字符
- 保留必要的空格、数字和常见标点
- 真实切片只保留与标签内容一致、且为单行的样本

目录结构：
```text
ocr_7lang_richbg_someocr_mix_300k/
├── train.txt
├── val.txt
├── dict_no_space.txt
└── images/
```

命名格式：
```text
images/train_word_1.jpg
images/train_word_2.jpg
...
images/val_word_1.jpg
images/val_word_2.jpg
...
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
  -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_ocr7lang_someocrmix_300k_finetune.yml
```

如需指定 GPU：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_ocr7lang_someocrmix_300k_finetune.yml
```

本次训练输出目录：
```text
output/ocr7lang_small_rec_someocrmix_300k/
```

训练策略上使用 Cosine 学习率调度并带 warmup，开启 EMA 和早停机制；其中 `trigger_acc: 0.9`，`patience: 5`，`min_delta: 0.001`。
`use_space_char` 设为 `true`，以改善空格识别问题。


## 模型评估
使用最优权重评估验证集：
```bash
python tools/eval.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_ocr7lang_someocrmix_300k_finetune.yml \
  -o Global.checkpoints=./output/ocr7lang_small_rec_someocrmix_300k/best_model/model
```


## 导出推理模型
将训练得到的 `best_model/model` 导出成 PaddleOCR 推理模型：
```bash
python tools/export_model.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_ocr7lang_someocrmix_300k_finetune.yml \
  -o Global.checkpoints=./output/ocr7lang_small_rec_someocrmix_300k/best_model/model \
     Global.save_inference_dir=./output/ocr7lang_small_rec_someocrmix_300k/inference
```

导出后的推理模型目录：
```text
output/ocr7lang_small_rec_someocrmix_300k/inference/
├── inference.json
├── inference.pdiparams
├── inference.yml
└── ppocr_keys.txt
```


## 推理模型测试
使用导出的推理模型测试单张单行图片，并在终端输出识别结果和置信度：
```bash
python tools/infer/predict_rec.py \
  --use_gpu True \
  --rec_model_dir ./output/ocr7lang_small_rec_someocrmix_300k/inference \
  --image_dir /path/to/your/single_line.jpg \
  --rec_algorithm SVTR_LCNet \
  --rec_image_shape 3,48,320 \
  --rec_batch_num 1 \
  --max_text_length 40 \
  --use_space_char True \
  --rec_char_dict_path ./data/ocr_7lang_richbg_someocr_mix_300k/dict_no_space.txt
```

当前模型已完成 `somecase` 回归测试，带标签部分的结果如下：
- `kor`：`bad -> good` 为 1349/1360
- `thai`：`bad -> good` 为 44/51
- `kaz`：`bad -> good` 为 225/237
- 有标签样本总体 `edit<=3` 覆盖率约 99.14%


## 整图测试说明
当前训练的是识别模型，只负责对已经裁切好的单行文字图片进行识别。如果要测试完整文档图片，需要额外搭配检测模型，将整图先检测裁切成文本行，再调用本次导出的识别模型。

示例命令：
```bash
python tools/infer/predict_system.py \
  --image_dir /path/to/document.jpg \
  --det_model_dir /path/to/det_model \
  --rec_model_dir ./output/ocr7lang_small_rec_someocrmix_300k/inference \
  --use_angle_cls false \
  --drop_score 0.0 \
  --draw_img_save_dir ./output/ocr7lang_small_system_test
```


## 注意事项
- 本版没有加入希伯来语和阿拉伯语，暂不处理从右到左书写问题。
- 当前训练图片为单行图，重点是 7 种语言的混合识别和英文混排。
- 标点、数字和空格都已纳入训练目标，但仍建议继续用真实数据做回归测试。
- 如果后续要进一步提升空格和标点稳定性，可以继续增加真实切片并做小学习率微调。
