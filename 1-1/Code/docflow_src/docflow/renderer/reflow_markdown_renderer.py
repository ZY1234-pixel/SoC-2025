"""Markdown emission directly from DocumentAnalysis."""

from __future__ import annotations

import base64
from pathlib import Path

from docflow.model.stages import INLINE_FORMULA_MARKER_PATTERN


class ReflowMarkdownRenderer:
    @staticmethod
    def _text(element) -> str:
        formulas = element.payload.get("inline_formulas") or ()

        def replace(match) -> str:
            index = int(match.group(1))
            if index >= len(formulas):
                return match.group(0)
            return "$" + str(formulas[index].get("latex") or "").strip("$") + "$"

        return INLINE_FORMULA_MARKER_PATTERN.sub(replace, element.text)

    def render(self, analysis, output_path: str, assets_dir: str | None = None) -> None:
        output = Path(output_path)
        assets = Path(assets_dir) if assets_dir else output.parent / "assets" / output.stem
        asset_ref = assets.relative_to(output.parent).as_posix()
        parts = []
        image_index = 0
        for page in analysis.pages:
            for element in page.elements:
                if element.kind in {"header", "footer", "page_number"}:
                    continue
                if element.kind == "heading":
                    parts.append(f"# {self._text(element)}")
                elif element.kind in {"paragraph_group", "caption"}:
                    parts.append(self._text(element))
                elif element.kind == "table_group":
                    html = str(element.payload.get("html") or "")
                    if html:
                        parts.append(html)
                    else:
                        data = element.payload.get("image_base64")
                        if data:
                            assets.mkdir(parents=True, exist_ok=True)
                            image_index += 1
                            image_path = assets / f"image_{image_index:04d}.png"
                            image_path.write_bytes(base64.b64decode(data))
                            parts.append(f"![table]({asset_ref}/{image_path.name})")
                elif element.kind == "equation_group" and element.payload.get("latex"):
                    parts.append(f"$${element.payload['latex']}$$")
                else:
                    data = element.payload.get("image_base64")
                    if data:
                        assets.mkdir(parents=True, exist_ok=True)
                        image_index += 1
                        image_path = assets / f"image_{image_index:04d}.png"
                        image_path.write_bytes(base64.b64decode(data))
                        parts.append(f"![{element.kind}]({asset_ref}/{image_path.name})")
                    caption = element.payload.get("caption")
                    if caption:
                        parts.append(str(caption))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n\n".join(filter(None, parts)) + "\n", encoding="utf-8")
