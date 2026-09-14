# InternImage-L semantic segmentation

Independent inference package for InternImage-L document foreground masks.

## Run

Click/run:

```bash
python infer.py
```

Default behavior:

- input: `../TEST_dewarp/`
- checkpoint: `checkpoints/best_final_before.pth`
- output: `outputs/`
- threshold: `0.60`
- preprocess: 1024x1024 letterbox, RGB float in `[0, 1]`

The script scans `../TEST_dewarp/` recursively, so images inside `../TEST_dewarp/perspective/` and
`../TEST_dewarp/instance/` are included automatically.

## Outputs

```text
outputs/
├── masks/       # original-size 0/255 masks for deployment/visual use
├── masks_1024/  # 1024x1024 letterbox-space masks
└── overlays/    # green mask overlays on original images
```

Subdirectories under `../TEST_dewarp/` are preserved in `outputs/`.

## Optional

```bash
python infer.py --input ../TEST_dewarp/perspective --output-dir outputs_perspective
python infer.py --checkpoint checkpoints/best_hd95_epoch_45.pth
python infer.py --device cpu --max-samples 3
python infer.py --no-overlays
```
