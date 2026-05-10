# 书本关键点检测


---

## 1. 环境要求

本模块环境要求与原项目的deeplabv3+环境基本一致，复用相关环境即可。

---

## 2. 数据集准备

注：本模块数据集为先前1427张数据集与后来补充的分类数据集(2796张)合并。除复制到同一个文件夹外未作其他处理，角点坐标json文件也已上传。

本人训练的设置如下:


### 2.1 数据格式
数据集需包含三部分：
1. **原始图片**：`.jpg` 格式，存放于 `newData/JPEGImages/`
2. **分割掩码**：`.png` 格式，同一文件名，存放于 `newData/SegmentationClass/`  
   - 像素值：0（背景）、1（书本区域），注意**不是** 0/255
3. **角点标注**：单个 JSON 文件 `newData/book_corners.json`，内容格式：
   ```json
   {
     "图片名.png": [[左上x,左上y], [右上x,右上y], [右下x,右下y], [左下x,左下y]],
     ...
   }
   ```
   四个角点的顺序必须为：左上、右上、右下、左下（顺时针）。

### 2.2 目录结构
```
newData/
├── JPEGImages/          # 所有训练图片
├── SegmentationClass/   # 所有分割掩码
├── book_corners.json    # 角点标注
└── ImageSets/
    └── Segmentation/
        ├── train.txt
        └── val.txt
```

### 2.3 生成训练/验证划分
运行脚本 `voc_annotation.py`，它会扫描 `SegmentationClass` 下的掩码文件，随机划分训练集和验证集（比例可调，默认 90% 训练）。
```bash
python voc_annotation.py
```

确保 `voc_annotation.py` 中配置了正确的数据集路径（默认为 `newData`）：
```python
VOCdevkit_path = 'newData'
```

---

## 3. 模型训练

### 3.1 配置训练参数
打开 `train.py`，确认或修改以下关键配置：
```python
Cuda            = True          # 使用 GPU
input_shape     = [256, 256]    # 输入尺寸，256x256
backbone        = "mobilenetv3" # 骨干网络
num_classes     = 2             # 背景 + 书本
num_keypoints   = 4             # 四个角点
model_path      = "model_data/450_act3_mobilenetv3_large.pth"  # 预训练权重
VOCdevkit_path  = 'newData'     # 数据集路径
keypoint_json_path = os.path.join(VOCdevkit_path, "book_corners.json")
```

其他超参数（学习率、batch size、epoch数等）已预设,与先前一致。  

### 3.2 开始训练
```bash
python train.py
```
训练过程会打印每个 epoch 的 loss、F-score，并自动保存最优权重到 `logs/时间戳/best_epoch_weights.pth`。  

训练设置与先前一致。

日志和权重保存在 `logs/` 目录下（按时间戳命名）。

---

## 4. Python 端
本模块训练还是保持了分割头，在训练完成后模型转换，仅保留了关键点检测头。
### 4.1 纯关键点模型推理
训练后将完整模型提取为仅保留关键点头的模型（用于部署）：
```bash
python export_kpt_model.py   # 生成 kpt_model_256x256.pth
```
然后用 `test_kpt_only.py` 测试：
```bash
python test_kpt_only.py      # 需手动修改脚本内的 model_path 和图片路径
```
该脚本会打印四个角点的原图坐标，可与标注对比。

---

## 5. 模型导出为 ncnn 格式

### 5.1 导出 TorchScript
```bash
python export_pnnx.py
```
该脚本会生成 `kpt_model_256x256.pt`（TorchScript 格式）。

### 5.2 使用 pnnx 转换为 ncnn 模型
确保已下载 pnnx 可执行文件并加入 PATH，然后执行：
```bash
pnnx kpt_model_256x256.pt inputshape=[1,3,256,256]
```
转换后会生成：
- `kpt_model_256x256.ncnn.param`
- `kpt_model_256x256.ncnn.bin`

这两个文件用于 C++ 推理。

---

## 6. C++ 部署

### 6.1 结构
参见 `ncnn_kpt/` 目录：
```
ncnn_kpt/
├── CMakeLists.txt
├── kpt_infer.cpp
├── kpt_model_256x256.ncnn.param
├── kpt_model_256x256.ncnn.bin
└── test.jpg              # 测试图片
```

### 6.2 配置依赖路径
编辑 `CMakeLists.txt`，修改 `ncnn` 和 `OpenCV` 的搜索路径：
```cmake
set(NCNN_DIR "D:/ncnn/x64/lib/cmake/ncnn")
set(OpenCV_DIR "D:/opencv/build/x64/vc16/lib")
find_package(ncnn REQUIRED CONFIG PATHS ${NCNN_DIR})
find_package(OpenCV REQUIRED CONFIG PATHS ${OpenCV_DIR})
```

### 6.3 编译
打开 Visual Studio 的 Developer Command Prompt，进入 `ncnn_kpt/` 目录：
```bash
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```
生成的可执行文件在 `build/Release/kpt_infer.exe`。

### 6.4 运行
将 `kpt_model_256x256.ncnn.param` 和 `kpt_model_256x256.ncnn.bin` 以及相关.dll文件复制到 exe 所在目录，然后：
```bash
kpt_infer.exe test.jpg
```
程序会打印四个角点的坐标，并生成 `kpt_result.jpg` 可视化结果。

---

## 7. 常见问题

### 7.1 路径问题
- 所有代码中的路径建议使用绝对路径，避免中文字符。
- 数据集目录结构必须严格遵循上述约定，否则 `dataloader.py` 会报找不到文件？
- cpp推理中未设置utf-8编码，输出中文有乱码，角标没问题。

### 7.2 掩码像素值错误
如果掩码是 0/255 而不是 0/1，训练会失败。

### 7.3 C++ 推理结果随机或不稳定
- 在 `kpt_infer.cpp` 中设置 `net.opt.num_threads = 1;`（已设置，推测与先前的线程问题相似）

---

## 8. 项目文件结构
```
项目根目录/
├── nets/                     # 网络模型定义
│   ├── deeplabv3_plus.py
│   ├── deeplabv3_training.py
│   ├── mobilenetv3.py
│   └── ...
├── utils/                    # 工具函数与数据处理
│   ├── dataloader.py
│   ├── utils_fit.py
│   ├── callbacks.py
│   ├── utils_metrics.py
│   ├── utils.py
│   └── keypoint_utils.py
├── train.py                  # 训练入口
├── voc_annotation.py         # 生成数据集划分
├── deeplab.py                # 完整模型推理封装
├── predict.py                # 批量预测
├── test_kpt_only.py          # 纯关键点模型测试
├── export_kpt_model.py       # 提取关键点权重
├── export_pnnx.py            # 导出 TorchScript
├── ncnn_kpt/                 # C++ 部署工程
├── newData/                  # 数据集（图片+掩码+标注）（需自行设置?)
├── model_data/               # 预训练权重（与先前一致）
└── logs/                     # 训练日志和权重
```

---
