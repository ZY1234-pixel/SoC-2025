"""LaTeX 到 Word 原生 OMML 的最小转换入口。"""

from __future__ import annotations

from functools import lru_cache

from lxml import etree
from docx.oxml.ns import qn
from docx.shared import Pt


MATH_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/math"


@lru_cache(maxsize=256)
def _latex_to_omml(latex: str) -> str:
    from latex2mathml.converter import convert
    import mathml2omml

    return mathml2omml.convert(convert(latex))


def append_latex_omml(paragraph, latex: str, font_size_pt: float | None = None) -> bool:
    """将 LaTeX 作为可编辑公式追加到段落，转换失败时返回 False。"""
    try:
        if not paragraph.runs:
            anchor = paragraph.add_run("\u200b")
            if font_size_pt is not None:
                anchor.font.size = Pt(max(float(font_size_pt), 0.5))
        omml = _latex_to_omml(str(latex).strip())
        wrapper = etree.fromstring(
            f'<root xmlns:m="{MATH_NAMESPACE}">{omml}</root>'.encode("utf-8")
        )
        _normalize_overlines(wrapper)
        _normalize_radicals(wrapper)
        _flatten_boxes(wrapper)
        paragraph._p.append(wrapper[0])
        return True
    except Exception:
        return False


def _normalize_overlines(root) -> None:
    """将 mathml2omml 的 limUpp 上划线改为兼容性更好的 officeMath accent。"""
    for element in list(root.xpath(".//m:limUpp", namespaces={"m": MATH_NAMESPACE})):
        limit = element.find(qn("m:lim"))
        base = element.find(qn("m:e"))
        if limit is None or base is None:
            continue
        mark = "".join(limit.itertext()).strip()
        if not mark:
            continue
        parent = element.getparent()
        accent = etree.Element(qn("m:acc"))
        properties = etree.SubElement(accent, qn("m:accPr"))
        character = etree.SubElement(properties, qn("m:chr"))
        character.set(qn("m:val"), mark)
        accent.append(base)
        parent.replace(element, accent)


def _normalize_radicals(root) -> None:
    """补齐标准 OMML 根号结构，避免 LibreOffice 渲染空度数框。"""
    for radical in root.xpath(".//m:rad", namespaces={"m": MATH_NAMESPACE}):
        if radical.find(qn("m:radPr")) is not None:
            continue
        properties = etree.Element(qn("m:radPr"))
        hide_degree = etree.SubElement(properties, qn("m:degHide"))
        hide_degree.set(qn("m:val"), "1")
        radical.insert(0, properties)
        radical.insert(1, etree.Element(qn("m:deg")))


def _flatten_boxes(root) -> None:
    """展开转换器额外生成的 m:box 包装，提高跨办公软件兼容性。"""
    for box in reversed(root.xpath(".//m:box", namespaces={"m": MATH_NAMESPACE})):
        contents = box.find(qn("m:e"))
        if contents is None:
            continue
        parent = box.getparent()
        index = parent.index(box)
        parent.remove(box)
        for child in reversed(contents):
            parent.insert(index, child)
