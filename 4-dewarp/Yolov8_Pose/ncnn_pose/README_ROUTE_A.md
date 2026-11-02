# 纯张量 Top-K 路线 A 角点解码

`main_route_a.cpp` 是面向单文档四角点任务的 NCNN + OpenCV 独立推理示例。
不读取原图像素做边缘拟合，只根据 `out0` 完成候选选择和角点融合。

## 模型契约

- 输入：`1x3x512x512`，RGB，float32，除以 255。
- 预处理：保持纵横比，`INTER_AREA` 缩放，letterbox 使用 114 填充。
- 输出：`out0 = [box(4), class(nc), keypoint(12)] x anchors`。
- 当前三分类模型：`nc=3`，`out0` 尺寸为 `19x5376`。
- 类别：`0=book`、`1=non_book`、`2=card`。

## 解码流程

实现只使用 `out0`：

1. 对每个 anchor 取三个类别分数的最大值；
2. 选择分数最高的 20 个 anchor；
3. 对每两个四边形计算 24 种角点排列下的最小平均距离；
4. 选择到其余候选距离之和最小的 medoid；
5. 保留与 medoid 距离不超过
   `3% * 512 * sqrt(2)` 像素的候选；
6. 按类别分加权平均角点；
7. 反算 letterbox 坐标；
8. 裁剪到可见画面；若存在缺角，取面积最大的内接四边形；
9. 按 `TL/TR/BR/BL` 顺序输出四个角点。

类别数由输出宽度自动推导，不写死 7 类偏移：

```cpp
const int class_count = out0.h - 16;
const int keypoint_offset = 4 + class_count;
```

## 编译

```bash
cmake -S Yolov8_Pose/ncnn_pose -B build
cmake --build build --config Release --target doc_detector_route_a
```

## 运行

```bash
doc_detector_route_a \
  model_fp16.ncnn.param \
  model_fp16.ncnn.bin \
  input.jpg \
  output.jpg
```

## 实测结果

以下数据均为 `conf>=0.40`，单位为原图像素：

| 自测数据集 | 单框 | 纯张量路线 A |
|---|---:|---:|
| real40 | 40.51 / 40.98 / 59.59 | 26.04 / 27.63 / 42.53 |
| doccorn300 | 7.77 / 9.67 / 16.74 | 5.31 / 6.61 / 11.52 |
| real_cut_r1b | 16.33 / 19.28 / 30.26 | 12.76 / 16.07 / 22.77 |
| real_hard12 | 45.36 / 65.38 / 117.55 | 36.74 / 45.58 / 75.48 |
| real_r2 | 39.41 / 51.24 / 109.36 | 36.13 / 37.84 / 59.88 |
| A3 实拍试卷留出集 | 61.82 / 65.92 / 93.70 | 42.14 / 43.54 / 61.04 |

表内数值依次为中位数 / 均值 / p90。

C++ 解码结果与 Python 参考实现相比，在普通图和缺角图上的四角坐标
误差均小于 `0.01px`。

## 适用范围

该示例面向“一张图只有一个主要文档”的业务场景，因此有意不使用 NMS。
如果 SDK 需要同时返回多个文档，应先完成多目标筛选，再对每个目标分别执行
路线 A 的角点融合。
