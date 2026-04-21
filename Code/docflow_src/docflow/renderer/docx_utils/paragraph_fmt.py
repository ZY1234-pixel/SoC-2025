"""python-docx 段落格式工具函数。

Migrated from original PaddleOCR recovery_to_doc lines 87-135, 549-554.
"""
from docx.shared import Pt
from docx.enum.text import WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def reset_paragraph_format(p):
    """将段落格式重置为干净的基线状态。

    设置行距 1.05，清零所有间距/缩进，启用孤行控制，
    并通过原始 XML 禁用 autoSpaceDE / autoSpaceDN。

    Args:
        p: python-docx Paragraph 对象。

    Returns:
        ParagraphFormat 对象，可继续自定义。
    """
    pf = p.paragraph_format
    pf.line_spacing = 1.05
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.first_line_indent = Pt(0)
    pf.left_indent = Pt(0)
    pf.right_indent = Pt(0)
    pf.widow_control = True

    # 禁用东亚文字的自动间距调整
    pPr = p._element.get_or_add_pPr()

    auto_space_de = pPr.find(qn("w:autoSpaceDE"))
    if auto_space_de is None:
        auto_space_de = OxmlElement("w:autoSpaceDE")
        pPr.append(auto_space_de)
    auto_space_de.set(qn("w:val"), "0")

    auto_space_dn = pPr.find(qn("w:autoSpaceDN"))
    if auto_space_dn is None:
        auto_space_dn = OxmlElement("w:autoSpaceDN")
        pPr.append(auto_space_dn)
    auto_space_dn.set(qn("w:val"), "0")

    return pf


def set_paragraph_spacing(
    p,
    space_before=None,
    space_after=None,
    line_spacing=None,
    exact=False,
):
    """设置段落间距属性。

    Args:
        p: python-docx Paragraph 对象。
        space_before: 段前间距（Pt），或 None 保持不变。
        space_after: 段后间距（Pt），或 None 保持不变。
        line_spacing: 行距值（Pt），或 None 保持不变。
        exact: 若为 True，使用 WD_LINE_SPACING.EXACTLY 而非比例行距。
    """
    pf = p.paragraph_format
    if space_before is not None:
        pf.space_before = Pt(space_before)
    if space_after is not None:
        pf.space_after = Pt(space_after)
    if line_spacing is not None:
        if exact:
            pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            pf.line_spacing = Pt(line_spacing)
        else:
            pf.line_spacing = Pt(line_spacing)


def make_small_paragraph(container, height_pt=0.7):
    """添加一个带精确行距的空段落，用于精细间距控制。

    插入一个通过 *height_pt* 控制高度的微小“间隔”段落，
    可用于在区块之间插入精确的垂直间距。

    Args:
        container: 支持 ``add_paragraph`` 的 python-docx 容器
            （如 Document body 或 table Cell）。
        height_pt: 间隔高度（磅，默认 0.7）。

    Returns:
        新创建的 Paragraph 对象。
    """
    p = container.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(height_pt)
    return p


def add_spacing_para(container, height_pt=0.7):
    """是 :func:`make_small_paragraph` 的别名。"""
    return make_small_paragraph(container, height_pt=height_pt)
