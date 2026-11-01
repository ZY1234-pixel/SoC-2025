#!/usr/bin/env python3
"""ESDNet-Lite去屏纹：唯一需要运行的命令行入口。

用法：python infer.py --input input.jpg --output output.png --threads 8

职责划分：
- 本文件：运行环境、命令行、图片读取和保存。
- model.py：NCNN模型加载与预测。
- tiling.py：补边、分块和余弦融合。
- fusion.py：接缝校正及保护式回退。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
import time

# 保持交接目录干净，导入本地模块时不生成__pycache__文件。
sys.dont_write_bytecode = True

# Linux分配器选项需要在启动时生效；缺少时重启本文件一次，替代外部启动脚本。
if __name__ == "__main__":
    if sys.platform.startswith("linux"):
        settings = {
            "MALLOC_ARENA_MAX": "2",
            "MALLOC_MMAP_THRESHOLD_": "131072",
            "MALLOC_TRIM_THRESHOLD_": "262144",
        }
        missing = {
            key: value for key, value in settings.items() if key not in os.environ
        }
        if missing:
            os.execve(
                sys.executable,
                [sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
                {**os.environ, **missing, "OPENBLAS_NUM_THREADS": "1"},
            )
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


import numpy as np
from PIL import Image

from model import ESDNetLiteNCNN
from fusion import restore_consistent


def main():
    """解析输入输出路径；保存PNG，不在交接目录生成额外报告或缓存。"""
    parser = argparse.ArgumentParser(description="ESDNet-Lite：CPU INT8去屏纹推理")
    parser.add_argument(
        "--input", required=True, help="输入图片路径，支持JPEG、PNG等格式"
    )
    parser.add_argument("--output", required=True, help="输出PNG路径，尺寸与输入相同")
    parser.add_argument("--threads", type=int, default=8, help="NCNN线程数，默认8")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads必须为正整数")
    source, dest = Path(args.input).resolve(), Path(args.output).resolve()
    root = Path(__file__).resolve().parent
    if not source.is_file():
        parser.error(f"找不到输入图片：{source}")
    if dest == source or dest.suffix.lower() != ".png":
        parser.error("输出必须是另一个以.png结尾的文件，不能覆盖输入")
    if root / "weights" in dest.parents:
        parser.error("请将结果保存在weights目录之外，避免覆盖模型")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    started = time.perf_counter()
    weights = root / "weights/esdnet_lite_int8"
    model = ESDNetLiteNCNN(
        weights.with_suffix(".param"), weights.with_suffix(".bin"), threads=args.threads
    )
    try:
        # Pillow统一解码为RGB，避免把其他库常见的BGR排列送入模型。
        with Image.open(source) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.uint8)

        def progress(done, total):
            """适度显示进度，避免每个小步骤都打印，干扰终端阅读。"""
            if done % 16 == 0 or done == total:
                print(f"分块进度：{done}/{total}", flush=True)

        # 融合流程内部负责归一化、补边及最终量化，不需要再次叠加原图。
        result, _ = restore_consistent(
            model, rgb, progress=progress, assembly="guarded"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result).save(dest, format="PNG", compress_level=6)
        print(
            f"结果已保存：{dest}（加载模型至保存耗时{time.perf_counter() - started:.2f}秒）"
        )
    finally:
        # 保存失败或用户中断时，也应释放网络持有的设备及池内资源。
        model.close()



if __name__ == "__main__":
    main()
