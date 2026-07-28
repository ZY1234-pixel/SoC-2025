from __future__ import annotations

import argparse
import json

from openvino_ocr import OpenVinoOCR


parser = argparse.ArgumentParser(description="Run PP-OCRv6 OpenVINO OCR on one image.")
parser.add_argument("image")
parser.add_argument("--threads", type=int, default=10)
args = parser.parse_args()

# 模型只初始化一次。正式服务应把该实例保存在进程中，供后续页面持续复用。
ocr = OpenVinoOCR(cpu_threads=args.threads)

# 首次推理会建立 OpenVINO 运行时缓存，因此先预热一次并丢弃本次结果。
# 这里为了简化示例，直接使用待识别图片预热；项目中也可以使用固定样例图。
ocr.warmup(args.image)

# 第二次调用代表正常推理，输出文本框、识别文字、置信度及分阶段耗时。
print(json.dumps(ocr.predict(args.image), ensure_ascii=False, indent=2))
