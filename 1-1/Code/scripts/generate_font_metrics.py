"""Generate fixed character-width metrics from the supported font files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fontTools.ttLib import TTCollection, TTFont


FONT_SPECS = (
    ("宋体", "fonts/宋体.ttc", 0),
    ("黑体", "fonts/黑体.ttf", 0),
    ("楷体", "fonts/楷体.ttf", 0),
    ("仿宋", "fonts/仿宋.ttf", 0),
    ("Times New Roman", "fonts/times.ttf", 0),
)
ALIASES = {
    "SimSun": "宋体",
    "SimHei": "黑体",
    "KaiTi": "楷体",
    "FangSong": "仿宋",
}


def _load_font(path: Path, font_number: int):
    if path.suffix.lower() == ".ttc":
        return TTCollection(str(path)).fonts[font_number]
    return TTFont(str(path), fontNumber=font_number)


def _font_metrics(project_root: Path, source: str, font_number: int) -> dict:
    path = project_root / source
    font = _load_font(path, font_number)
    units_per_em = int(font["head"].unitsPerEm)
    cmap = font.getBestCmap()
    widths = font["hmtx"].metrics
    codepoint_widths = {}
    for codepoint, glyph_name in cmap.items():
        if not 0 <= int(codepoint) <= 0xFFFF:
            continue
        width = float(widths[glyph_name][0]) / units_per_em
        if abs(width - 1.0) <= 0.0005:
            continue
        codepoint_widths[str(int(codepoint))] = round(width, 6)
    return {
        "source": source,
        "units_per_em": units_per_em,
        "default_em": 1.0,
        "codepoint_widths": codepoint_widths,
    }


def generate(project_root: Path) -> dict:
    return {
        "format_version": 1,
        "aliases": ALIASES,
        "fonts": {
            name: _font_metrics(project_root, source, font_number)
            for name, source, font_number in FONT_SPECS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docflow_src"
        / "docflow"
        / "planning"
        / "font_metrics.json",
    )
    args = parser.parse_args()
    payload = generate(args.project_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
