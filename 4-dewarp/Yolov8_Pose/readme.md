# Yolov8-Pose四角点检测


---

## 文件

### 核心代码 
* `inference_intranet.py`：Python 推理脚本（封装了核心检测类）。
* `main.cpp`：端侧 C++ 推理源码。
* `CMakeLists.txt`：端侧 C++ 编译配置文件。

### 模型权重 (另附)
* `best.torchscript`
* `model.ncnn.param` 与 `model.ncnn.bin`：端侧设备推理所需的 NCNN 静态图和权重。

---



##  一：服务器端 (Python)


1. 确保 `best.torchscript` 与测试图片准备就绪。
2. 修改 `inference_intranet.py` 底部 `__main__` 中的路径参数。
3. 执行：`python inference_intranet.py`

> **tips**：Windows 环境下 PyTorch C++ 底层对中文路径支持存在 Bug。本脚本已内置自动映射系统纯英文临时目录的修复逻辑。

---

## 二：端侧 (C++ / NCNN)

**环境依赖：** CMake 3.12+, NCNN, OpenCV (需包含 Imgproc 模块)

1. 在代码目录下创建 `build` 文件夹并进入。
2. 执行 CMake 配置（请按实际情况替换路径）：
   `cmake .. -DNCNN_DIR="/path/to/ncnn" -DOpenCV_DIR="/path/to/opencv"`
3. 编译可执行程序：`cmake --build . --config Release`
4. 运行前，请确保 NCNN 模型参数文件以及依赖的 `.dll` 动态库与生成的可执行文件在同一目录下。


---

