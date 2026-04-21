# 模型目录说明

Git 仓库中不提交模型权重与推理参数文件。

请将百度网盘中的模型文件下载后，按下面结构放回本目录：

```text
Code/models/
├── layout/
│   └── picodet_lcnet_x1_0_fgd_layout_cdla_infer/
├── det/
│   └── ch/
│       └── PP-OCRv5_mobile_det_infer/
├── rec/
│   └── ch/
│       └── PP-OCRv5_mobile_rec_infer/
└── table/
    └── SLANet_plus_infer/
```

请在提交前把百度网盘信息补充到顶层 `README.md`。
