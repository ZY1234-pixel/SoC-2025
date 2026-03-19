"""python-docx 表格与单元格格式工具函数。

Migrated from original PaddleOCR recovery_to_doc lines 230-281, 771-834.
"""
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

TWIPS_PER_PT = 20


# ---------------------------------------------------------------------------
# 单元格边框
# ---------------------------------------------------------------------------

def set_cell_border(cell, **kwargs):
    """设置表格单元格的各侧边框。

    每个关键字参数应为边框方向（``top``、``bottom``、
    ``start``、``end``、``insideH``、``insideV``），值为含可选键
    ``sz``（线宽，1/8 磅）、``val``（样式，如 ``'single'``）、
    ``color``（十六进制，如 ``'000000'``）和 ``space``（间距）的字典。

    示例::

        set_cell_border(cell, top={"sz": 4, "val": "single", "color": "000000"})
    """
    tc = cell._element
    tcPr = tc.get_or_add_tcPr()

    tcBorders = tcPr.find(qn("w:tcBorders"))
    if tcBorders is None:
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)

    for edge, attrs in kwargs.items():
        edge_elem = tcBorders.find(qn(f"w:{edge}"))
        if edge_elem is None:
            edge_elem = OxmlElement(f"w:{edge}")
            tcBorders.append(edge_elem)
        for attr_name, attr_val in attrs.items():
            edge_elem.set(qn(f"w:{attr_name}"), str(attr_val))


# ---------------------------------------------------------------------------
# 单元格底色
# ---------------------------------------------------------------------------

def set_cell_shading(cell, color_hex):
    """设置表格单元格的背景（底纹）颜色。

    Args:
        cell: python-docx 表格 Cell。
        color_hex: 六位十六进制颜色字符串（如 ``'FFFF00'``）。
    """
    tc = cell._element
    tcPr = tc.get_or_add_tcPr()

    shading = tcPr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tcPr.append(shading)

    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), color_hex)


# ---------------------------------------------------------------------------
# 单元格内边距
# ---------------------------------------------------------------------------

def set_cell_margins(cell, top=0, bottom=0, start=0, end=0):
    """设置单元格内边距（DXA/twips）。

    Args:
        cell: python-docx 表格 Cell。
        top: 上边距（twips）。
        bottom: 下边距（twips）。
        start: 左边距（twips）。
        end: 右边距（twips）。
    """
    tc = cell._element
    tcPr = tc.get_or_add_tcPr()

    tcMar = tcPr.find(qn("w:tcMar"))
    if tcMar is None:
        tcMar = OxmlElement("w:tcMar")
        tcPr.append(tcMar)

    for side, val in [("top", top), ("bottom", bottom),
                      ("start", start), ("end", end)]:
        elem = tcMar.find(qn(f"w:{side}"))
        if elem is None:
            elem = OxmlElement(f"w:{side}")
            tcMar.append(elem)
        elem.set(qn("w:w"), str(int(val)))
        elem.set(qn("w:type"), "dxa")


def set_cell_right_margin(cell, margin_pt):
    """仅设置单元格的右（end）边距（磅）。

    Args:
        cell: python-docx 表格 Cell。
        margin_pt: 右边距（磅）。
    """
    twips = int(margin_pt * TWIPS_PER_PT)
    tc = cell._element
    tcPr = tc.get_or_add_tcPr()

    tcMar = tcPr.find(qn("w:tcMar"))
    if tcMar is None:
        tcMar = OxmlElement("w:tcMar")
        tcPr.append(tcMar)

    end_elem = tcMar.find(qn("w:end"))
    if end_elem is None:
        end_elem = OxmlElement("w:end")
        tcMar.append(end_elem)
    end_elem.set(qn("w:w"), str(twips))
    end_elem.set(qn("w:type"), "dxa")


# ---------------------------------------------------------------------------
# 表格左缩进
# ---------------------------------------------------------------------------

def indent_table(table, indent_pt):
    """设置整个表格的左缩进。

    Args:
        table: python-docx Table 对象。
        indent_pt: 缩进量（磅）。
    """
    twips = int(indent_pt * TWIPS_PER_PT)
    tbl = table._tbl
    tblPr = tbl.tblPr
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr")
        tbl.insert(0, tblPr)

    tblInd = tblPr.find(qn("w:tblInd"))
    if tblInd is None:
        tblInd = OxmlElement("w:tblInd")
        tblPr.append(tblInd)
    tblInd.set(qn("w:w"), str(twips))
    tblInd.set(qn("w:type"), "dxa")


# ---------------------------------------------------------------------------
# 清除所有可见边框（布局表格）
# ---------------------------------------------------------------------------

def clear_table_borders(table):
    """移除表格的所有可见边框（用作布局表格）。

    同时将单元格边距置零，使内容紧贴边缘。

    Args:
        table: python-docx Table 对象。
    """
    tbl = table._tbl
    tblPr = tbl.tblPr
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr")
        tbl.insert(0, tblPr)

    # --- 表格级边框 ------------------------------------------------
    borders = tblPr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tblPr.append(borders)

    for edge_name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        elem = borders.find(qn(f"w:{edge_name}"))
        if elem is None:
            elem = OxmlElement(f"w:{edge_name}")
            borders.append(elem)
        elem.set(qn("w:val"), "none")
        elem.set(qn("w:sz"), "0")
        elem.set(qn("w:space"), "0")
        elem.set(qn("w:color"), "auto")

    # --- 将每个单元格的边距置零 ------------------------------------
    for row in table.rows:
        for cell in row.cells:
            set_cell_margins(cell, top=0, bottom=0, start=0, end=0)


# ---------------------------------------------------------------------------
# 通过 tblGrid 设置列宽
# ---------------------------------------------------------------------------

def set_table_col_widths(table, widths_pt):
    """通过 ``<w:tblGrid>`` 和 ``tcW`` 设置表格的显式列宽。

    Args:
        table: python-docx Table 对象。
        widths_pt: 列宽可迭代对象（磅）。长度必须与表格列数一致。
    """
    tbl = table._tbl
    widths_twips = [int(w * TWIPS_PER_PT) for w in widths_pt]

    # --- tblGrid 元素 ----------------------------------------------------
    for old in tbl.findall(qn("w:tblGrid")):
        tbl.remove(old)

    grid = OxmlElement("w:tblGrid")
    for tw in widths_twips:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(tw))
        grid.append(col)

    # 将 tblGrid 插入到 tblPr 之后
    tblPr = tbl.tblPr
    if tblPr is not None:
        tblPr.addnext(grid)
    else:
        tbl.insert(0, grid)

    # --- 逐单元格设置 tcW -------------------------------------------------------
    for row in tbl.findall(qn("w:tr")):
        cells = row.findall(qn("w:tc"))
        for i, tc in enumerate(cells):
            if i < len(widths_twips):
                tcPr = tc.find(qn("w:tcPr"))
                if tcPr is None:
                    tcPr = OxmlElement("w:tcPr")
                    tc.insert(0, tcPr)
                tcW = tcPr.find(qn("w:tcW"))
                if tcW is None:
                    tcW = OxmlElement("w:tcW")
                    tcPr.insert(0, tcW)
                tcW.set(qn("w:w"), str(widths_twips[i]))
                tcW.set(qn("w:type"), "dxa")

    # --- 表格布局设为固定宽度以提高保真度 -----------------------------
    tblPr = tbl.tblPr
    if tblPr is not None:
        layout = tblPr.find(qn("w:tblLayout"))
        if layout is None:
            layout = OxmlElement("w:tblLayout")
            tblPr.append(layout)
        layout.set(qn("w:type"), "fixed")


# ---------------------------------------------------------------------------
# 将表格缩放到最大宽度（防止溢出）
# ---------------------------------------------------------------------------

def fit_table_to_width(table, max_width_pt: float) -> None:
    """若 HTML 表格总列宽超过 *max_width_pt*，按比例缩放所有列。

    同时将表格设为固定布局并写入 tblW，确保 Word 不自动扩展。
    """
    tbl = table._tbl
    max_twips = max(1, int(max_width_pt * TWIPS_PER_PT))

    # 读取 tblGrid 中各列宽
    grid = tbl.find(qn("w:tblGrid"))
    if grid is None:
        return
    col_els = grid.findall(qn("w:gridCol"))
    if not col_els:
        return

    orig_twips = []
    for col_el in col_els:
        w = col_el.get(qn("w:w"))
        orig_twips.append(int(w) if w else 0)

    total = sum(orig_twips)
    if total <= 0:
        return

    # 按比例缩放（仅当总宽超出时才缩放）
    if total <= max_twips:
        scale = 1.0
    else:
        scale = max_twips / total

    new_twips = [max(200, int(w * scale)) for w in orig_twips]
    # 修正因取整导致的误差（分配给最后一列）
    new_twips[-1] = max(200, max_twips - sum(new_twips[:-1]))

    # 更新 tblGrid
    for col_el, nw in zip(col_els, new_twips):
        col_el.set(qn("w:w"), str(nw))

    # 更新每行每单元格的 tcW（处理跨列合并）
    for tr in tbl.findall(qn("w:tr")):
        cursor = 0
        for tc in tr.findall(qn("w:tc")):
            tcPr = tc.find(qn("w:tcPr"))
            if tcPr is None:
                tcPr = OxmlElement("w:tcPr")
                tc.insert(0, tcPr)
            # 获取 gridSpan（合并列数）
            span = 1
            gs_el = tcPr.find(qn("w:gridSpan"))
            if gs_el is not None:
                span = int(gs_el.get(qn("w:val"), "1"))
            cell_w = sum(new_twips[cursor: cursor + span])
            tcW = tcPr.find(qn("w:tcW"))
            if tcW is None:
                tcW = OxmlElement("w:tcW")
                tcPr.insert(0, tcW)
            tcW.set(qn("w:w"), str(cell_w))
            tcW.set(qn("w:type"), "dxa")
            cursor += span

    # 设置表格整体宽度 + 固定布局
    tblPr = tbl.tblPr
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr")
        tbl.insert(0, tblPr)

    tbl_w_el = tblPr.find(qn("w:tblW"))
    if tbl_w_el is None:
        tbl_w_el = OxmlElement("w:tblW")
        tblPr.append(tbl_w_el)
    tbl_w_el.set(qn("w:w"), str(max_twips))
    tbl_w_el.set(qn("w:type"), "dxa")

    layout = tblPr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tblPr.append(layout)
    layout.set(qn("w:type"), "fixed")
