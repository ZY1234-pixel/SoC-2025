"""统一的结果目录规划工具。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import re
from typing import Dict, Iterable, List, Optional, Sequence


_ILLEGAL_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_SPACE_RE = re.compile(r"\s+")
_UNDERSCORE_RE = re.compile(r"_+")


def build_run_id(now: Optional[datetime] = None) -> str:
    current = now or datetime.now()
    return current.strftime("run_%Y%m%d_%H%M%S")


def sanitize_sample_key(raw_name: str, max_len: int = 120) -> str:
    name = Path(raw_name).stem
    name = _ILLEGAL_CHARS_RE.sub("_", name)
    name = _SPACE_RE.sub("_", name)
    name = _UNDERSCORE_RE.sub("_", name)
    name = name.strip(" ._")
    if not name:
        name = "sample"
    return name[:max_len]


def resolve_run_dir(path: Path) -> Path:
    if (path / "run_manifest.json").is_file():
        return path
    candidates = sorted(
        [item for item in path.iterdir() if item.is_dir() and item.name.startswith("run_")],
        key=lambda item: item.name,
    ) if path.is_dir() else []
    if candidates:
        return candidates[-1]
    return path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


@dataclass(frozen=True)
class SampleResultLayout:
    sample_key: str
    source_path: Path
    sample_dir: Path
    json_path: Path
    recognition_path: Path
    render_plan_path: Path
    docx_path: Path
    markdown_path: Path
    pdf_path: Path
    debug_dir: Path

    def debug_image_path(self, page_index: int, suffix: str) -> Path:
        return self.debug_dir / f"page_{page_index + 1:04d}.{suffix}.jpg"


@dataclass
class ResultRunLayout:
    output_root: Path
    run_id: str
    run_dir: Path
    runtime_dir: Path
    run_manifest_path: Path
    _sample_counts: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def create(cls, output_root: Path, run_id: Optional[str] = None) -> "ResultRunLayout":
        normalized_root = output_root.resolve()
        actual_run_id = run_id or build_run_id()
        run_dir = normalized_root / actual_run_id
        suffix = 2
        while run_dir.exists():
            run_dir = normalized_root / f"{actual_run_id}__{suffix}"
            suffix += 1
        runtime_dir = run_dir / "_runtime"
        run_dir.mkdir(parents=True, exist_ok=True)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            output_root=normalized_root,
            run_id=actual_run_id,
            run_dir=run_dir,
            runtime_dir=runtime_dir,
            run_manifest_path=run_dir / "run_manifest.json",
        )

    def create_sample(self, source_path: Path, preferred_name: Optional[str] = None) -> SampleResultLayout:
        base_key = sanitize_sample_key(preferred_name or source_path.name)
        count = self._sample_counts.get(base_key, 0) + 1
        self._sample_counts[base_key] = count
        sample_key = base_key if count == 1 else f"{base_key}__{count}"
        sample_dir = self.run_dir / sample_key
        sample_dir.mkdir(parents=True, exist_ok=True)
        return SampleResultLayout(
            sample_key=sample_key,
            source_path=source_path.resolve(),
            sample_dir=sample_dir,
            json_path=sample_dir / f"{sample_key}.json",
            recognition_path=sample_dir / f"{sample_key}.recognition.json",
            render_plan_path=sample_dir / f"{sample_key}.render_plan.json",
            docx_path=sample_dir / f"{sample_key}.docx",
            markdown_path=sample_dir / f"{sample_key}.md",
            pdf_path=sample_dir / f"{sample_key}.pdf",
            debug_dir=sample_dir / "debug",
        )

    def write_run_manifest(self, payload: dict) -> None:
        write_json(self.run_manifest_path, payload)


def build_main_run_manifest(
    run_layout: ResultRunLayout,
    input_path: Path,
    formats: Iterable[str],
    layout_model_dir: Path,
    samples: Sequence[dict],
    total_pages: int,
    quality_summary: Optional[dict],
    strategy_stats: Optional[dict],
    failures: Sequence[str],
) -> dict:
    payload = {
        "run_id": run_layout.run_id,
        "run_dir": str(run_layout.run_dir),
        "input_path": str(input_path),
        "output_root": str(run_layout.output_root),
        "formats": list(formats),
        "layout_model_dir": str(layout_model_dir),
        "sample_count": len(samples),
        "page_count": total_pages,
        "failures": list(failures),
        "samples": list(samples),
    }
    if quality_summary:
        payload["quality_summary"] = dict(quality_summary)
    if strategy_stats:
        payload["strategy_stats"] = dict(strategy_stats)
    return payload


def build_eval_run_manifest(
    run_layout: ResultRunLayout,
    case_names: Iterable[str],
    dpi: int,
    samples: Sequence[dict],
) -> dict:
    return {
        "run_id": run_layout.run_id,
        "run_dir": str(run_layout.run_dir),
        "output_root": str(run_layout.output_root),
        "dpi": dpi,
        "case_names": list(case_names),
        "sample_count": len(samples),
        "samples": list(samples),
    }
