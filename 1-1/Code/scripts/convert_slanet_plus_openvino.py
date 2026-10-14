"""下载 SLANet Plus ONNX 并转换为 OpenVINO FP32 模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import onnx
import openvino as ov


SOURCE_URL = (
    "https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/main/"
    "models/TabRec/SlanetPlus/slanet-plus.onnx"
)
SOURCE_SHA256 = "d57a942af6a2f57d6a4a0372573c696a2379bf5857c45e2ac69993f3b334514b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(source_url: str, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and sha256(target) == expected_sha256:
        return
    urllib.request.urlretrieve(source_url, target)
    actual = sha256(target)
    if actual != expected_sha256:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"SLANet Plus SHA-256 mismatch: {actual}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "models" / "openvino_source" / "SlanetPlus" / "slanet-plus.onnx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "models_openvino" / "RapidAI_TableRec_openvino" / "slanet_plus",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    download(SOURCE_URL, args.source, SOURCE_SHA256)
    model = onnx.load(args.source, load_external_data=False)
    characters = model.metadata_props[0].value.splitlines()
    if len(characters) != 48:
        raise RuntimeError(f"unexpected SLANet Plus character count: {len(characters)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "slanet-plus.xml"
    if args.force or not model_path.is_file():
        ov.save_model(ov.convert_model(args.source), model_path, compress_to_fp16=False)
    (args.output_dir / "character.txt").write_text("\n".join(characters), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "name": "SLANet Plus",
                "source": SOURCE_URL,
                "source_sha256": SOURCE_SHA256,
                "openvino_model": model_path.name,
                "artifact": model_path.with_suffix(".bin").name,
                "character_file": "character.txt",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(model_path)


if __name__ == "__main__":
    main()
