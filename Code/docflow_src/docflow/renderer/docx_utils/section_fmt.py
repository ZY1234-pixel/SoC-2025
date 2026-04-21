"""python-docx 节/分栏格式工具函数。

Migrated from original PaddleOCR recovery_to_doc lines 171-203.
"""
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

TWIPS_PER_PT = 20


def set_section_columns(sectPr, col_count, col_widths_pt=None, spacing_pt=18):
    """设置 Word 节中的文本分栏数。

    若 *col_widths_pt* 为 ``None`` 则设为等宽分栏；
    否则 *col_widths_pt* 应为长度等于 *col_count* 的列表，
    每个元素以磅为单位指定列宽。

    Args:
        sectPr: 节的 ``CT_SectPr`` XML 元素。
        col_count: 分栏数（1 = 单栏）。
        col_widths_pt: 可选的列宽列表（磅）。
        spacing_pt: 栏间距（磅，默认 18）。
    """
    if col_count < 1:
        col_count = 1

    # 移除已有的 <w:cols> 元素
    for old in sectPr.findall(qn("w:cols")):
        sectPr.remove(old)

    cols_elem = OxmlElement("w:cols")
    cols_elem.set(qn("w:num"), str(col_count))
    spacing_twips = str(int(spacing_pt * TWIPS_PER_PT))
    cols_elem.set(qn("w:space"), spacing_twips)

    if col_widths_pt is None:
        # 等宽分栏：由 Word 自动计算尺寸
        cols_elem.set(qn("w:equalWidth"), "1")
    else:
        # 自定义宽度：需要显式的 <w:col> 子元素
        cols_elem.set(qn("w:equalWidth"), "0")
        for i, w_pt in enumerate(col_widths_pt):
            col_el = OxmlElement("w:col")
            col_el.set(qn("w:w"), str(int(w_pt * TWIPS_PER_PT)))
            # 除最后一栏外，每栏都设置间距属性
            if i < col_count - 1:
                col_el.set(qn("w:space"), spacing_twips)
            else:
                col_el.set(qn("w:space"), "0")
            cols_elem.append(col_el)

    sectPr.append(cols_elem)
