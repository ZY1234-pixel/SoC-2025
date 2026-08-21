"""全流程测试脚本通用工具函数。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, List

from .model import RuntimePaths


def ensure_runtime_paths(paths: RuntimePaths, table_backend: str = "rapidai") -> None:
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
    ]
    if table_backend == "slanet":
        required.append(paths.table_model)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("缺少必要运行资产：\n  " + "\n  ".join(missing))


def find_libreoffice() -> str | None:
    for name in ("libreoffice", "soffice"):
        path = shutil.which(name)
        if path:
            return path
    return None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_formats(raw: str) -> List[str]:
    allowed = {"docx", "markdown", "pdf"}
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
