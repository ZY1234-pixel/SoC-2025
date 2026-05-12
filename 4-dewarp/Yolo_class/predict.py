from ultralytics import YOLO

model = YOLO(r"D:\奔图\deeplabv3p_zzh\runs\classify\YoloV8_ClassFor6\book_cls_6class\weights\best.pt")
results = model("VOCdevkit/VOC2007/JPEGImages/1.jpg")
print(results[0].probs.top1)   # 输出类别 id
print(results[0].names)        # 类别名映射