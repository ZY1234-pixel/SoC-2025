from types import SimpleNamespace

import base64

from docflow.renderer.reflow_markdown_renderer import ReflowMarkdownRenderer


def test_markdown_restores_inline_formula_marker() -> None:
    element = SimpleNamespace(
        text="能量满足 {{DOCFLOW_FORMULA_0}}。",
        payload={"inline_formulas": ({"latex": r"E=mc^2"},)},
    )

    assert ReflowMarkdownRenderer._text(element) == "能量满足 $E=mc^2$。"


def test_markdown_table_without_html_falls_back_to_source_image(tmp_path) -> None:
    element = SimpleNamespace(
        kind="table_group",
        payload={"image_base64": base64.b64encode(b"png").decode("ascii")},
        text="",
    )
    analysis = SimpleNamespace(pages=(SimpleNamespace(elements=(element,)),))
    output = tmp_path / "result.md"

    ReflowMarkdownRenderer().render(analysis, str(output))

    assert "![table](assets/result/image_0001.png)" in output.read_text(encoding="utf-8")
    assert (tmp_path / "assets" / "result" / "image_0001.png").read_bytes() == b"png"
