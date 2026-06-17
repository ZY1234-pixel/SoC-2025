# Instance Segmentation Inference

This folder contains the deployment-side inference files for the YOLO instance
segmentation model.

- `predict.py`: batch inference script.
- `img/`: input images.
- `weights/best.pt`: default trained weights copied from `Instance_seg_train`.
- `img_out/`: generated prediction outputs.

Training code, datasets, run logs, and the local Ultralytics source are kept in:

```text
../../Instance_seg_train
```

Run inference:

```bash
python predict.py
```

Useful options:

```bash
python predict.py --source img/several
python predict.py --device cpu
python predict.py --conf 0.05 --iou 0.95 --max-det 100
```
