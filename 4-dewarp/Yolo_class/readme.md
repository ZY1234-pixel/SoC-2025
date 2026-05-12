## 复现指南：YOLOv8 书本图像六分类模型

### 1. 环境要求
与先前项目基本一致，需要额外配置yolo的 ultralytics (`pip install ultralytics`)


### 2. 数据集准备
原始数据按类别分别存放在不同文件夹下，例如：
```
newData/
├── 报纸或海报/
├── 单侧书本页面/
├── 非刚体票据/
├── 双页面左右结构展开书本/
├── 显示器或投影屏/
└── unclassified/
```

#### 2.1 生成训练/验证集划分
运行 `split_cls_dataset.py`，脚本会读取 `newData`，按 85% 训练、15% 验证的比例随机划分，并转换为英文类别名，生成 `cls_dataset` 目录结构：
```
cls_dataset/
└── train/
│   ├── double_page_book/
│   ├── newspaper_poster/
│   ├── receipt/
│   ├── screen/
│   ├── single_page/
│   └── unclassified/
└── val/
    └── (同上)
```
确保 `split_cls_dataset.py` 中的 `src_dir` 指向原始数据文件夹，`dst_dir` 设为 `cls_dataset`。运行：
```bash
python split_cls_dataset.py
```

### 3. 模型训练
使用训练脚本 `train_cls.py`，会从预训练的 `yolov8n-cls.pt` 开始微调。
```bash
python train_cls.py
```
主要训练参数：
- 输入尺寸：256×256
- 训练轮数：100
- 批次大小：32
- 训练结果默认保存在 `runs/classify/book_cls_6class/`，最优权重为 `weights/best.pt`

### 4. 导出 TorchScript 模型
运行导出脚本：
```bash
python export_pnnx.py
```
该脚本会加载 `best.pt`，输出 `best.torchscript` 至同级目录。

> **注意**：若路径不一致，请修改脚本中的模型路径或直接使用命令行：
> ```bash
> yolo export model="runs/classify/YoloV8_ClassFor6/book_cls_6class/weights/best.pt" format=torchscript imgsz=256
> ```

### 5. 转换为 ncnn 模型
使用 pnnx 工具将 TorchScript 转为 ncnn 格式：
```bash
pnnx runs\classify\YoloV8_ClassFor6\book_cls_6class\weights\best.torchscript inputshape=[1,3,256,256]
```
成功后会在同级目录生成 `best.ncnn.param` 和 `best.ncnn.bin`。将这两个文件拷贝到 `ncnn_cls/` 目录下。

### 6. C++ 部署
#### 6.1 准备环境
- 安装 ncnn（需预先编译或下载预编译包，并配置 CMake 搜索路径）
- 安装 OpenCV（需与 Visual Studio 版本匹配）
- 修改 `ncnn_cls/CMakeLists.txt` 中的 `NCNN_DIR` 和 `OpenCV_DIR` 为实际路径。

#### 6.2 编译
打开 Visual Studio 的 Developer Command Prompt，执行：
```cmd
cd ncnn_cls
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```
生成的可执行文件 `cls_infer.exe` 位于 `build/Release/`。

#### 6.3 运行推理
将测试图片（如 `test.jpg`）和 `best.ncnn.param`、`best.ncnn.bin` 放入 `Release` 目录，运行：
```cmd
cls_infer.exe test.jpg
```
终端会输出预测类别、置信度及各分类概率，并生成 `cls_result.jpg` 可视化结果。

### 7. 常见问题
- cpp推理中未设置utf-8编码，输出中文有乱码，类别没问题。

### 8. 项目文件结构
```
├── YoloV8_ClassFor6/
│   ├── split_cls_dataset.py
│   ├── train_cls.py
│   ├── export_pnnx.py
│   └── cls_dataset/              (划分好的数据集，可重新生成)
├── runs/classify/YoloV8_ClassFor6/book_cls_6class/weights/
│   ├── best.pt
│   └── best.torchscript
├── ncnn_cls/
│   ├── CMakeLists.txt
│   ├── cls_infer.cpp
│   ├── best.ncnn.param
│   └── best.ncnn.bin
└── README.md                     (本文件)
```