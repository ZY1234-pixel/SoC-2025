"""全流程测试脚本通用工具函数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .model import RuntimePaths


def ensure_runtime_paths(
    paths: RuntimePaths,
    formula_enabled: bool = True,
) -> None:
    required = [
        paths.docflow_src / "docflow",
        paths.paddle_root / "ppstructure",
        paths.paddle_root / "ppocr",
        paths.paddle_root / "tools",
        paths.layout_model,
        paths.det_model,
        paths.rec_model,
        paths.rec_char_dict,
        paths.rapidocr_rec_char_dict,
        paths.models_openvino_root / "RapidAI_TableRec_openvino" / "table_cls" / "yolo_cls.xml",
        paths.models_openvino_root / "RapidAI_TableRec_openvino" / "wired_table_v2" / "unet.xml",
        paths.models_openvino_root / "RapidAI_TableRec_openvino" / "slanet_plus" / "slanet-plus.xml",
        paths.models_openvino_root / "RapidAI_TableRec_openvino" / "slanet_plus" / "character.txt",
        paths.models_openvino_root
        / "RapidAI_TableRec_openvino"
        / "ocr_cls"
        / "ch_ppocr_mobile_v2.0_cls_infer.xml",
        paths.models_openvino_root / "font_openvino" / "mobilenetv3.xml",
    ]
    if formula_enabled:
        required.append(paths.formula_model / "inference.yml")
        required.append(
            next(
                (
                    paths.formula_model / name
                    for name in ("inference.onnx", "model.onnx")
                    if (paths.formula_model / name).is_file()
                ),
                paths.formula_model / "inference.onnx",
            )
        )
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("缺少必要运行资产：\n  " + "\n  ".join(missing))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_formats(raw: str) -> List[str]:
    allowed = {"docx", "markdown", "pdf", "pdf_fidelity"}
    values = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not values:
        return ["docx", "markdown"]
    unknown = [x for x in values if x not in allowed]
    if unknown:
        raise ValueError(f"存在不支持的输出格式: {unknown}；可选: {sorted(allowed)}")
    if "md" in values:
        values = ["markdown" if x == "md" else x for x in values]
    return values


def print_list(title: str, items: Iterable[str]) -> None:
    print(title)
    for item in items:
        print(f"  - {item}")
