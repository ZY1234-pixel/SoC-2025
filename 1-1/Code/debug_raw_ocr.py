"""Dump raw PaddleOCR result to check text_region format."""
from __future__ import annotations
import sys, os, json, cv2, numpy as np
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
os.chdir(str(CODE_ROOT))

# Add Paddle runtime to path BEFORE importing anything else
paddle_root = CODE_ROOT / "third_party" / "paddle_runtime"
if str(paddle_root) not in sys.path:
    sys.path.insert(0, str(paddle_root))

sys.path.insert(0, str(CODE_ROOT / "docflow_src"))

from model import RuntimePaths
from utils import ensure_runtime_paths
from test import make_engine

paths = RuntimePaths.discover()
ensure_runtime_paths(paths)
engine = make_engine(paths, None)

from dataset import collect_samples

samples = collect_samples(input_path=str(CODE_ROOT.parent / "dataset" / "newspaper_01.png"))
sample_path, _ = samples[0]
img = cv2.imread(str(sample_path))
print(f"Image shape: {img.shape}")

result, _ = engine(img, img_idx=0)
print(f"Result: {len(result)} regions")

for idx, region in enumerate(result[:3]):
    print(f"\n=== Region {idx} ===")
    print(f"type: {region.get('type')}")
    bbox = region.get('bbox')
    print(f"bbox: {bbox}")
    print(f"score: {region.get('score')}")
    res = region.get('res')
    print(f"res type: {type(res)}, len: {len(res) if isinstance(res, list) else 'N/A'}")
    if isinstance(res, list):
        for i, item in enumerate(res[:2]):
            if isinstance(item, dict):
                tr = item.get('text_region')
                txt = item.get('text', '')[:30]
                if tr is not None:
                    tr_list = tr.tolist() if hasattr(tr, 'tolist') else tr
                    print(f"  item[{i}] DICT: text_region pts={len(tr_list) if isinstance(tr_list, list) else 0}")
                    if isinstance(tr_list, list) and len(tr_list) >= 2:
                        print(f"    pts: {tr_list}")
                else:
                    print(f"  item[{i}] DICT: text_region=None")
                print(f"    text: {txt}")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                poly = item[0]
                if hasattr(poly, 'tolist'):
                    poly = poly.tolist()
                print(f"  item[{i}] LIST: poly len={len(poly) if poly else 0}")
                if poly and len(poly) >= 2:
                    print(f"    pts: {poly}")
                txt = str(item[1])[:30] if item[1] else ""
                print(f"    text: {txt}")
            else:
                print(f"  item[{i}] OTHER: type={type(item)}")
