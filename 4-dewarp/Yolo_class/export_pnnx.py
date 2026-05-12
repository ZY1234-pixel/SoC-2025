from ultralytics import YOLO
model = YOLO(r"runs/classify/YoloV8_ClassFor6/book_cls_6class/weights/best.pt")
model.export(format="torchscript", imgsz=256)