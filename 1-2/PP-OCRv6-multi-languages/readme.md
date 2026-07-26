# PP-OCRv6 多语言识别
## 训练目标
本次训练基于 PaddleOCR `release/3.7` 源码和 `PP-OCRv6_medium_rec` 识别模型，对 7 种语言进行混合微调。当前版本只做单行、单语种识别训练，不生成同一行内多语言混合图片。

7 种语言如下：
- 韩语：`hy`
- 泰语：`ty`
- 俄语：`ey`
- 保加利亚语：`bjly`
- 乌克兰语：`wkl`
- 哈萨克语：`hsk`
- 希腊语：`xl`



## 环境安装
### 创建虚拟环境
为了不影响原有 PaddleOCR 环境，重新创建 `paddleocr37` 虚拟环境：
```bash
conda create -n paddleocr37 python=3.9 -y
conda activate paddleocr37
python -m pip install -U pip setuptools wheel
```

### 重新安装 PaddlePaddle 3.1.1（release/3.7更推荐这个版本的框架）
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
数据集通过 `text-renderer` 生成，语料主要来自 Wikimedia `pages-articles` dump，共33万张，包括训练集30万张，验证集3万张。
具体内容由百度网盘提供。



## 模型训练
进入 release/3.7 源码目录并激活环境：
```bash
cd /home/ww/ww/whu/project/PaddleOCR-release37
conda activate paddleocr37
```

单卡训练：
```bash
python tools/train.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_medium_rec_mixed7_finetune.yml
```

如需指定 GPU：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_medium_rec_mixed7_finetune.yml
```



## 模型评估
使用最优权重评估验证集：
```bash
python tools/eval.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_medium_rec_mixed7_finetune.yml \
  -o Global.checkpoints=./output/mixed7_medium_rec_punct/best_accuracy
```

也可以使用 `best_model/model`：
```bash
python tools/eval.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_medium_rec_mixed7_finetune.yml \
  -o Global.checkpoints=./output/mixed7_medium_rec_punct/best_model/model
```



## 导出推理模型
将训练得到的 `best_model/model` 导出成 PaddleOCR 推理模型：
```bash
python tools/export_model.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_medium_rec_mixed7_finetune.yml \
  -o Global.checkpoints=./output/mixed7_medium_rec_punct/best_model/model \
     Global.save_inference_dir=./output/mixed7_medium_rec_punct/inference_model
```



## 推理模型测试
使用导出的推理模型测试单张单行图片，并在终端输出识别结果和置信度：
```bash
python tools/infer/predict_rec.py \
  --use_gpu True \
  --rec_model_dir ./output/mixed7_medium_rec_punct/inference_model \
  --image_dir /path/to/your/single_line.jpg \
```



## 整图测试说明
当前训练的是识别模型，只负责对已经裁切好的单行文字图片进行识别。如果要测试完整文档图片，需要额外搭配检测模型，将整图先检测裁切成文本行，再调用本次导出的识别模型。

示例命令：
```bash
python tools/infer/predict_system.py \
  --image_dir /path/to/document.jpg \
  --det_model_dir /path/to/det_model \
  --rec_model_dir ./output/mixed7_medium_rec_punct/inference_model \
  --use_angle_cls false \
  --drop_score 0.0 \
  --draw_img_save_dir ./output/mixed7_system_test
```



## 注意事项
- 本版数据集没有加入希伯来语和阿拉伯语，暂不处理从右到左书写问题。
- 当前数据集中同一张图只包含一种语言，尚未加入单行内多语言混合样本。
- 标点和数字是适量加入的，并非每张图都包含；目标是提升模型对 `, . ? : ; !` 和数字的泛化能力。
- `use_space_char` 当前为 `false`，如果后续希望模型显式学习空格，需要同步调整字典、配置和训练数据。