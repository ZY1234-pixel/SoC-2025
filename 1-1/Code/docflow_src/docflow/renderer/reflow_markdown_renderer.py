"""Markdown emission directly from DocumentAnalysis."""

from __future__ import annotations

import base64
from pathlib import Path


class ReflowMarkdownRenderer:
    def render(self, analysis, output_path: str) -> None:
        output = Path(output_path)
        assets = output.with_name(f"{output.stem}_assets")
        parts = []
        image_index = 0
        for page in analysis.pages:
            for element in page.elements:
                if element.kind in {"header", "footer", "page_number"}:
                    continue
                if element.kind == "heading":
                    parts.append(f"# {element.text}")
                elif element.kind in {"paragraph_group", "caption"}:
                    parts.append(element.text)
                elif element.kind == "table_group":
                    parts.append(str(element.payload.get("html") or ""))
                elif element.kind == "equation_group" and element.payload.get("latex"):
                    parts.append(f"$${element.payload['latex']}$$")
                else:
                    data = element.payload.get("image_base64")
                    if data:
                        assets.mkdir(parents=True, exist_ok=True)
                        image_index += 1
                        image_path = assets / f"image_{image_index:04d}.png"
                        image_path.write_bytes(base64.b64decode(data))
                        parts.append(f"![{element.kind}]({assets.name}/{image_path.name})")
                    caption = element.payload.get("caption")
                    if caption:
                        parts.append(str(caption))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n\n".join(filter(None, parts)) + "\n", encoding="utf-8")
