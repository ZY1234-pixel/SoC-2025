#!/usr/bin/env python3
"""Run deployment inference on 4-dewarp/TEST_dewarp by default."""

from instance_seg.cli import parse_args


# IDE output switch:
# "debug": overlay visualizations and ROI boundary diagnostics.
# "production": original-size per-instance uint8 PNG masks, values 0 and 255 only.
OUTPUT_MODE = "debug"


def main(argv=None):
    args = parse_args(argv, output_mode=OUTPUT_MODE)
    # Help and argument validation work without importing PyTorch/CUDA.
    from instance_seg.pipeline import run

    run(args)


if __name__ == "__main__":
    main()
