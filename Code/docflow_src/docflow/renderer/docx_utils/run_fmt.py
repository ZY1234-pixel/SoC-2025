"""python-docx Run（字符级）格式工具函数。

Migrated from original PaddleOCR recovery_to_doc lines 141-166.
"""
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn


def set_run_font(
    run,
    font_name="Times New Roman",
    east_asia="\u5b8b\u4f53",  # 宋体
    font_size=None,
    bold=False,
    italic=False,
    underline=False,
    strikethrough=False,
    superscript=False,
    subscript=False,
    color_rgb=None,
):
    """配置文本 *run* 的字体属性。

    同时设置西文字体和东亚字体（通过 ``w:rFonts`` 的 ``w:eastAsia``
    属性）。若指定 *font_size*，则四舍五入到 0.5 pt 精度，
    最小值为 5 pt。

    Args:
        run: python-docx Run 对象。
        font_name: 西文字体名（默认 ``'Times New Roman'``）。
        east_asia: CJK 字体名（默认 ``'宋体'``）。
        font_size: 字号（磅），或 None 不设置。
        bold: 是否加粗。
        italic: 是否斜体。
        color_rgb: ``RGBColor`` 或 3 元组 ``(r, g, b)``，或 None。
    """
    font = run.font
    font.name = font_name
    font.bold = bold
    font.italic = italic
    font.underline = bool(underline)
    font.strike = bool(strikethrough)
    font.superscript = bool(superscript)
    font.subscript = bool(subscript)

    # 通过 XML 元素显式设置东亚字体，以确保 CJK 字符
    # 使用正确的字体渲染
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        from docx.oxml import OxmlElement

        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:eastAsia"), east_asia)

    # 字号：四舍五入到 0.5 pt 精度，最小 5 pt
    if font_size is not None:
        size = round(font_size * 2) / 2.0  # round to nearest 0.5
        size = max(size, 5.0)
        font.size = Pt(size)

    # 颜色
    if color_rgb is not None:
        if isinstance(color_rgb, RGBColor):
            font.color.rgb = color_rgb
        else:
            font.color.rgb = RGBColor(*color_rgb)
