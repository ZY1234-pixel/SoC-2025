"""下载 PP-FormulaNet_plus-M 的 ONNX 权重并校验 SHA-256。"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


SOURCE_URL = (
    "https://huggingface.co/jinzhenj/PP-FormulaNet_plus-M_onnx/resolve/"
    "9eb491f62e52f45571fbf88f4d0335e4eb36f9b9/inference.onnx"
)
SOURCE_SHA256 = "71b6d389cf7b857e45252a4b98cfced1a3ffca7bf24d9497d02d052a41d9493b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "models"
            / "formula"
            / "PP-FormulaNet_plus-M"
            / "PP-FormulaNet_plus-M_infer"
            / "inference.onnx"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.is_file() and not args.force and sha256(args.output) == SOURCE_SHA256:
        print(args.output)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SOURCE_URL, args.output)
    actual = sha256(args.output)
    if actual != SOURCE_SHA256:
        args.output.unlink(missing_ok=True)
        raise RuntimeError(f"PP-FormulaNet ONNX SHA-256 mismatch: {actual}")
    print(args.output)


if __name__ == "__main__":
    main()
