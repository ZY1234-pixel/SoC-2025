"""DocFlow 命令行接口。"""
import argparse
import json
import sys

from docflow.pipeline import RecoveryPipeline
from docflow.config import RecoveryConfig


def main():
    parser = argparse.ArgumentParser(description="DocFlow —— 版面恢复工具")
    parser.add_argument("--input", "-i", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出文件路径")
    parser.add_argument("--format", "-f", default="docx", choices=["docx", "markdown", "pdf"],
                        help="输出格式（默认: docx）")
    parser.add_argument("--max-cols", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = RecoveryConfig(max_cols=args.max_cols, save_debug=args.debug)
    pipeline = RecoveryPipeline(config=config)
    result = pipeline.recover(args.input, args.output, format=args.format)
    print(f"Output saved to: {result}")


if __name__ == "__main__":
    main()
