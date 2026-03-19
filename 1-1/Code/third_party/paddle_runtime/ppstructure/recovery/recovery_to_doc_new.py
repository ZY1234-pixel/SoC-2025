# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
版面恢复模块：将 PaddleOCR 版面分析结果转换为高保真 Word 文档。

参考 pdf2docx 的核心实现逻辑:
  Page → Section → Column → Blocks → TextBlock / TableBlock / ImageBlock
确保页面结构（页边距、分栏、段落间距、行间距、对齐、缩进）和
样式（字体、字号、粗体、颜色、表格边框/底纹）尽可能还原。
"""

import os
import io
import cv2
import numpy as np
from copy import deepcopy

from docx import Document
from docx.shared import Inches, Pt, Emu, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.enum.section import WD_SECTION, WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ROW_HEIGHT_RULE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

from ppstructure.recovery.table_process import HtmlToDocx
from ppocr.utils.logging import get_logger

logger = get_logger()

# =====================================================================
# 常量
# =====================================================================
A4_WIDTH_PT = 595.28          # A4 宽度 (Pt)
A4_HEIGHT_PT = 841.89         # A4 高度 (Pt)
DEFAULT_MARGIN_PT = 54.0      # 默认页边距 (Pt), 约 1.9cm
MIN_LINE_SPACING_PT = 0.7     # 最小行高 (Pt)
TWIPS_PER_PT = 20             # 1 Pt = 20 twips (openxml 基本单位)
EMU_PER_PT = 12700            # 1 Pt = 12700 EMU


# =====================================================================
# 坐标映射：像素 → Word Pt
# =====================================================================
class CoordMapper:
    """将 OCR 图片像素坐标映射到 Word 页面 Pt 坐标。"""

    def __init__(self, img_w, img_h, page_width_pt=None, page_height_pt=None,
                 margin_left_pt=None, margin_right_pt=None,
                 margin_top_pt=None, margin_bottom_pt=None):
        self.img_w = img_w
        self.img_h = img_h
        ml = margin_left_pt or DEFAULT_MARGIN_PT
        mr = margin_right_pt or DEFAULT_MARGIN_PT
        mt = margin_top_pt  or DEFAULT_MARGIN_PT
        mb = margin_bottom_pt or DEFAULT_MARGIN_PT
        usable_w = (page_width_pt  or A4_WIDTH_PT)  - ml - mr
        usable_h = (page_height_pt or A4_HEIGHT_PT) - mt - mb
        self.scale   = usable_w / max(img_w, 1)   # 水平缩放
        self.h_scale = usable_h / max(img_h, 1)   # 垂直缩放

    def w(self, px):
        """像素宽度 → Pt"""
        return px * self.scale

    def h(self, px):
        """像素高度 → Pt"""
        return px * self.h_scale


# =====================================================================
# 段落格式工具 (参考 pdf2docx common/docx.py)
# =====================================================================
def _reset_paragraph_format(p):
    """
    重置段落格式：
    - 禁用中英文/数字自动间距 (autoSpaceDE/DN)  ← pdf2docx 的关键做法
    - 行间距默认 1.05 倍
    """
    pf = p.paragraph_format
    pf.line_spacing = 1.05
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.left_indent = Pt(0)
    pf.right_indent = Pt(0)
    pf.widow_control = True

    pPr = p._p.get_or_add_pPr()
    for tag_name in ('w:autoSpaceDE', 'w:autoSpaceDN'):
        el = OxmlElement(tag_name)
        el.set(qn('w:val'), '0')
        pPr.insert(0, el)
    return pf


def _set_paragraph_spacing(p, space_before=None, space_after=None,
                           line_spacing=None, exact=False):
    """设置段落间距。
    line_spacing: float 或 Pt 值
    exact=True  → 固定行距 (WD_LINE_SPACING.EXACTLY)
    exact=False → 相对倍数行距
    """
    pf = p.paragraph_format
    if space_before is not None:
        pf.space_before = Pt(max(0, space_before))
    if space_after is not None:
        pf.space_after = Pt(max(0, space_after))
    if line_spacing is not None:
        if exact:
            pf.line_spacing = Pt(line_spacing)
            pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        else:
            pf.line_spacing = line_spacing


def _make_small_paragraph(doc, height_pt=MIN_LINE_SPACING_PT):
    """创建一个高度很小的空段落，用于表格前后的间距控制。
    参考 pdf2docx Blocks.make_docx 中表格前后的处理。"""
    p = doc.add_paragraph()
    _reset_paragraph_format(p)
    _set_paragraph_spacing(p, line_spacing=height_pt, exact=True)
    return p


# =====================================================================
# Run 字体工具 (参考 pdf2docx TextSpan._set_text_format)
# =====================================================================
def _set_run_font(run, font_name='Times New Roman', east_asia='宋体',
                  font_size=None, bold=False, italic=False, color_rgb=None):
    """
    设置 Run 字体。
    显式设置 w:eastAsia 确保中文字体生效（pdf2docx 的关键做法）。
    字号取 0.5pt 精度（pdf2docx: round(size*2)/2.0）。
    """
    run.font.name = font_name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:ascii'), font_name)
    rFonts.set(qn('w:hAnsi'), font_name)
    rFonts.set(qn('w:eastAsia'), east_asia)

    if font_size is not None:
        # 只接受 x.0 或 x.5 的精度
        fs = round(font_size * 2) / 2.0
        run.font.size = Pt(max(fs, 5))
    run.font.bold = bold
    run.font.italic = italic
    if color_rgb is not None:
        run.font.color.rgb = color_rgb


# =====================================================================
# 分栏工具 (参考 pdf2docx common/docx.py set_columns)
# =====================================================================
def _set_section_columns(sectPr, col_count, col_widths_pt=None, spacing_pt=18):
    """设置节的分栏。
    col_widths_pt: 各栏宽度 (Pt) 列表，None 则等宽。
    """
    cols_el = sectPr.find(qn('w:cols'))
    if cols_el is None:
        cols_el = OxmlElement('w:cols')
        sectPr.append(cols_el)
    else:
        for child in list(cols_el):
            cols_el.remove(child)
        for attr in list(cols_el.attrib.keys()):
            del cols_el.attrib[attr]

    spacing_twips = int(spacing_pt * TWIPS_PER_PT)

    if col_count <= 1:
        cols_el.set(qn('w:num'), '1')
        return

    if col_widths_pt is None:
        cols_el.set(qn('w:num'), str(col_count))
        cols_el.set(qn('w:space'), str(spacing_twips))
        cols_el.set(qn('w:equalWidth'), '1')
    else:
        cols_el.set(qn('w:num'), str(col_count))
        cols_el.set(qn('w:equalWidth'), '0')
        for i, cw in enumerate(col_widths_pt):
            col_el = OxmlElement('w:col')
            col_el.set(qn('w:w'), str(int(cw * TWIPS_PER_PT)))
            sp = str(spacing_twips) if i < col_count - 1 else '0'
            col_el.set(qn('w:space'), sp)
            cols_el.append(col_el)


# =====================================================================
# 图片工具 (参考 pdf2docx common/docx.py add_image)
# =====================================================================
def _add_image_to_doc(doc, image_bytes, width_pt, max_width_pt=None):
    """向文档添加居中的行内图片。
    返回包含图片的段落，失败返回 None。"""
    if max_width_pt is not None:
        width_pt = min(width_pt, max_width_pt)
    stream = io.BytesIO(image_bytes)
    try:
        doc.add_picture(stream, width=Pt(width_pt))
    except Exception:
        return None
    p = doc.paragraphs[-1]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _reset_paragraph_format(p)
    # 图片段落使用单倍行距避免被固定行距裁切
    p.paragraph_format.line_spacing = 1.0
    return p


# =====================================================================
# 表格工具 (参考 pdf2docx Table / Cell / Row 的 make_docx)
# =====================================================================
def _set_cell_border(cell, **kwargs):
    """设置单元格边框。
    kwargs: top/bottom/start/end = {"sz": 8, "val": "single", "color": "#000000"}
    参考 pdf2docx docx.py set_cell_border。
    """
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn('w:tcBorders'))
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    for edge in ('start', 'top', 'end', 'bottom'):
        edge_data = kwargs.get(edge)
        if edge_data:
            el = tcBorders.find(qn(f'w:{edge}'))
            if el is None:
                el = OxmlElement(f'w:{edge}')
                tcBorders.append(el)
            for key in ('sz', 'val', 'color', 'space'):
                if key in edge_data:
                    el.set(qn(f'w:{key}'), str(edge_data[key]))


def _set_cell_shading(cell, color_hex):
    """设置单元格底纹颜色。color_hex: "FFFF00" 或 "#FFFF00" """
    c = color_hex.lstrip('#')
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{c}"/>')
    cell._tc.get_or_add_tcPr().append(shd)


def _set_cell_margins(cell, top=0, bottom=0, start=0, end=0):
    """设置单元格内边距 (DXA / twips 单位)。"""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for name, val in [('top', top), ('bottom', bottom),
                      ('start', start), ('end', end)]:
        node = OxmlElement(f'w:{name}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def _indent_table(table, indent_pt):
    """设置表格左缩进。参考 pdf2docx docx.py indent_table。"""
    tbl_pr = table._element.xpath('w:tblPr')
    if tbl_pr:
        e = OxmlElement('w:tblInd')
        e.set(qn('w:w'), str(int(TWIPS_PER_PT * indent_pt)))
        e.set(qn('w:type'), 'dxa')
        tbl_pr[0].append(e)


# =====================================================================
# 布局分析辅助
# =====================================================================
def _estimate_font_size_pt(bbox_height_px, num_lines, mapper):
    """从 bbox 高度和行数估算字体大小 (Pt)。"""
    height_pt = mapper.h(bbox_height_px)
    if num_lines <= 0:
        num_lines = 1
    line_height_pt = height_pt / num_lines
    # 近似: font_size ≈ line_height / 1.3
    fs = line_height_pt / 1.3
    return max(6, min(36, round(fs * 2) / 2.0))


def _detect_alignment(bbox, col_left, col_right, threshold_ratio=0.12):
    """根据块左右边到栏边界的距离推断对齐方式。
    参考 pdf2docx TextBlock._parse_alignment。
    """
    x1, _, x2, _ = bbox
    col_w = col_right - col_left
    if col_w <= 0:
        return WD_ALIGN_PARAGRAPH.LEFT

    d_left = abs(x1 - col_left)
    d_right = abs(col_right - x2)
    thresh = col_w * threshold_ratio

    left_ok = d_left < thresh
    right_ok = d_right < thresh

    if left_ok and right_ok:
        return WD_ALIGN_PARAGRAPH.JUSTIFY
    elif not left_ok and not right_ok:
        if abs(d_left - d_right) < thresh:
            return WD_ALIGN_PARAGRAPH.CENTER
        return WD_ALIGN_PARAGRAPH.LEFT
    elif left_ok:
        return WD_ALIGN_PARAGRAPH.LEFT
    else:
        return WD_ALIGN_PARAGRAPH.RIGHT


def _extract_text(region):
    """从 OCR region 提取纯文本。

    res 可能是以下格式之一:
    - list of {"text": str, "confidence": float, "text_region": ...}
    - str (纯文本)
    - dict {"text": ..., "html": ...}
    """
    res = region.get('res', '')
    if isinstance(res, list):
        return "".join([r.get('text', '').strip() for r in res])
    elif isinstance(res, str):
        return res
    elif isinstance(res, dict):
        return res.get('text', '')
    return ''


def _get_table_html(region):
    """从 table region 提取 HTML 字符串。"""
    res = region.get('res', {})
    if isinstance(res, dict) and 'html' in res:
        return res['html']
    return None


def _count_text_lines(region):
    """估算 region 中的文本行数。

    优先用 text_region 坐标从纵向多展性计算（避免 OCR 把一行词识别为多个横向片段导致行数虚高）。
    """
    res = region.get('res', '')
    if isinstance(res, list) and res:
        # 尝试用 text_region 纵向覆盖赋予行数
        # 收集所有行的 y 区间，将有重叠的合并，最终计算连通组数
        intervals = []
        for ln in res:
            tr = ln.get('text_region', [])
            if tr:
                y1 = min(pt[1] for pt in tr)
                y2 = max(pt[1] for pt in tr)
                if y2 > y1:
                    intervals.append((y1, y2))
        if intervals:
            intervals.sort()
            merged = [intervals[0]]
            for y1, y2 in intervals[1:]:
                if y1 < merged[-1][1] - 2:   # 有重叠（横向并排）
                    merged[-1] = (merged[-1][0], max(merged[-1][1], y2))
                else:
                    merged.append((y1, y2))
            return max(1, len(merged))
        # 无 text_region 时回退到 len(res)
        return max(1, len(res))
    text = _extract_text(region)
    return max(1, text.count('\n') + 1)


def _encode_roi_png(img, bbox, img_h, img_w):
    """从原图中裁切 bbox 区域并编码为 PNG 字节。
    返回 (bytes, w_px, h_px) 或 None。
    """
    rx1, ry1, rx2, ry2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    rx1, ry1 = max(0, rx1), max(0, ry1)
    rx2, ry2 = min(img_w, rx2), min(img_h, ry2)
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    roi = img[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None
    ok, buf = cv2.imencode('.png', roi)
    if not ok:
        return None
    return buf.tobytes(), rx2 - rx1, ry2 - ry1


# =====================================================================
# 可视化工具
# =====================================================================
def draw_sorted_layout(img, res):
    """生成版面分析可视化图，按 col_index 用不同颜色标注 bbox。
    跨栏块用粗边框 + 内框白线标记。
    返回标注后的 BGR ndarray。
    """
    PALETTE = [
        (220,  80,  50),  # col 0
        ( 50, 180,  80),  # col 1
        ( 50, 130, 220),  # col 2
        (180,  50, 180),  # col 3
        ( 50, 200, 200),  # col 4
        (200, 130,  50),  # col 5
    ]
    vis = img.copy()
    for i, b in enumerate(res):
        ci = b.get('col_index', 0)
        spanned = b.get('spanned_cols', [ci])
        color = PALETTE[ci % len(PALETTE)]
        x1, y1, x2, y2 = map(int, b['bbox'])
        thickness = 4 if len(spanned) > 1 else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if len(spanned) > 1:
            cv2.rectangle(vis, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4),
                          (255, 255, 255), 1)
        nc = b.get('col_count', 1)
        label = f"{i}:{b.get('type','?')[:5]} {nc}c-{ci}"
        if len(spanned) > 1:
            label += f"[{','.join(map(str, spanned))}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ly = max(y1 - 2, th + 4)
        cv2.rectangle(vis, (x1, ly - th - 2), (x1 + tw + 4, ly + 2),
                      color, -1)
        cv2.putText(vis, label, (x1 + 2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return vis


# =====================================================================
# 首行缩进检测 + 多段落拆分
# =====================================================================
def _detect_first_line_indent(region, mapper, min_indent_px=15):
    """从行级 text_region 检测首行缩进量，返回 Pt 值（无缩进返回 0）。

    原理：比较第一行 x1 与后续行最小 x1 之差，超过阈值则认定有首行缩进。
    参考 pdf2docx TextBlock._parse_alignment 中对 indent 的处理。
    """
    lines = region.get('res')
    if not isinstance(lines, list) or len(lines) < 2:
        return 0

    def line_x1(line):
        tr = line.get('text_region', [])
        return min(pt[0] for pt in tr) if tr else None

    first_x = line_x1(lines[0])
    if first_x is None:
        return 0
    rest_xs = [line_x1(ln) for ln in lines[1:]]
    rest_xs = [x for x in rest_xs if x is not None]
    if not rest_xs:
        return 0

    baseline_x = min(rest_xs)
    indent_px = first_x - baseline_x
    if indent_px < min_indent_px:
        return 0
    return round(mapper.w(indent_px) * 2) / 2.0


def _split_into_paragraphs(lines, mapper, min_indent_px=12):
    """将 OCR 行列表拆分为多个段落（每个段落是行子列表）。

    判断新段落起始的标准（满足其一即可）：
    1. 本行 x1 明显大于前一行 x1（首行缩进特征）
    2. 本行与上一行的垂直间距 > 平均行高 * 1.4（段落间距特征）
    """
    if not lines:
        return []

    def line_x1(line):
        tr = line.get('text_region', [])
        return min(pt[0] for pt in tr) if tr else None

    def line_y1(line):
        tr = line.get('text_region', [])
        return min(pt[1] for pt in tr) if tr else None

    def line_y2(line):
        tr = line.get('text_region', [])
        return max(pt[1] for pt in tr) if tr else None

    # 估算平均行高
    heights = []
    for ln in lines:
        y1, y2 = line_y1(ln), line_y2(ln)
        if y1 is not None and y2 is not None and y2 > y1:
            heights.append(y2 - y1)
    avg_h = (sum(heights) / len(heights)) if heights else 20

    paras = [[lines[0]]]
    for i in range(1, len(lines)):
        ln = lines[i]
        prev_ln = lines[i - 1]
        is_new = False

        # 检查是否是横向并排的行（y 轴重叠 > 平均行高的 40%）—— 这种情况不切分段落
        y1_cur  = line_y1(ln)
        y2_prev = line_y2(prev_ln)
        y1_prev = line_y1(prev_ln)
        y2_cur  = line_y2(ln)
        h_overlap = 0
        if (y1_cur is not None and y2_cur is not None
                and y1_prev is not None and y2_prev is not None):
            h_overlap = min(y2_cur, y2_prev) - max(y1_cur, y1_prev)
        if h_overlap > avg_h * 0.4:
            # 横向并排，强制并入同一段落
            paras[-1].append(ln)
            continue

        # 条件1: 首行缩进
        x1_cur  = line_x1(ln)
        x1_prev = line_x1(prev_ln)
        if x1_cur is not None and x1_prev is not None:
            if x1_cur - x1_prev >= min_indent_px:
                is_new = True

        # 条件2: 行间距过大
        if not is_new:
            if y1_cur is not None and y2_prev is not None:
                gap = y1_cur - y2_prev
                if gap > avg_h * 1.4:
                    is_new = True

        if is_new:
            paras.append([ln])
        else:
            paras[-1].append(ln)

    return paras


# =====================================================================
# 间距段落 / 通用块渲染器
# =====================================================================
def _add_spacing_para(container, height_pt=MIN_LINE_SPACING_PT):
    """向 container（Document 或 _Cell）添加指定高度的空段落。"""
    p = container.add_paragraph()
    _reset_paragraph_format(p)
    _set_paragraph_spacing(p, line_spacing=height_pt, exact=True)
    return p


def _render_block(container, region, mapper, img, img_h, img_w,
                  col_width_pt, col_left_px, col_right_px, char_width_px,
                  space_before=0, in_table_cell=False):
    """
    将单个 OCR block 渲染到 container（Document 或 _Cell）。
    统一处理 figure / table / text / unknown 四类 block。
    返回 (last_paragraph | None, ended_sentence: bool)。
    """
    bbox  = region.get('bbox', [0, 0, 0, 0])
    rtype = region.get('type', 'text').lower()

    def _region_or_page_roi():
        """优先使用 region 自带 ROI（通常来自原图裁剪），失败再回退整页按 bbox 裁剪。"""
        roi = region.get('img', None)
        if isinstance(roi, np.ndarray) and roi.size > 0:
            ok, buf = cv2.imencode('.png', roi)
            if ok:
                rh, rw = roi.shape[:2]
                return buf.tobytes(), rw, rh
        return _encode_roi_png(img, bbox, img_h, img_w)

    # ---- figure / equation ----
    if rtype in ('figure', 'equation'):
        result = _region_or_page_roi()
        if result:
            img_bytes, roi_w, roi_h_px = result
            pw_w = min(mapper.w(roi_w), col_width_pt * 0.98)
            # 同时受垂直缩放约束：确保图片在缩放后不超过 h_scale 对应的自然高度
            pw_h = mapper.h(roi_h_px) * roi_w / max(roi_h_px, 1)
            pw = min(pw_w, pw_h)
            if space_before > 3:
                _add_spacing_para(container, min(space_before, 12))
            p = container.add_paragraph()
            _reset_paragraph_format(p)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run()
            try:
                run.add_picture(io.BytesIO(img_bytes), width=Pt(pw))
            except Exception:
                pass
            _set_paragraph_spacing(p, space_after=3)
        return None, True

    # ---- table ----
    if rtype == 'table':
        _add_spacing_para(container, max(space_before, MIN_LINE_SPACING_PT))
        table_html = _get_table_html(region)
        rendered = False
        if table_html and not in_table_cell:
            try:
                parser = HtmlToDocx()
                parser.handle_table(table_html, container)
                if hasattr(container, 'tables') and container.tables:
                    t = container.tables[-1]
                    t.alignment = WD_TABLE_ALIGNMENT.CENTER
                    for row in t.rows:
                        row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
                rendered = True
            except Exception as e:
                logger.warning(f"表格HTML渲染失败，回退为图片: {e}")
        if not rendered:
            result = _region_or_page_roi()
            if result:
                img_bytes, roi_w, roi_h_px = result
                pw_w = min(mapper.w(roi_w), col_width_pt * 0.98)
                pw_h = mapper.h(roi_h_px) * roi_w / max(roi_h_px, 1)
                pw = min(pw_w, pw_h)
                p = container.add_paragraph()
                _reset_paragraph_format(p)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.line_spacing = 1.0
                run = p.add_run()
                try:
                    run.add_picture(io.BytesIO(img_bytes), width=Pt(pw))
                except Exception:
                    pass
        _add_spacing_para(container, MIN_LINE_SPACING_PT)
        return None, True

    # ---- text / title / caption 等 ----
    if rtype in ('text', 'title', 'reference', 'header', 'footer',
                 'figure_caption', 'table_caption'):
        text_stripped = _extract_text(region).strip()
        if not text_stripped:
            return None, True

        lines_raw  = region.get('res')
        num_lines  = _count_text_lines(region)
        bbox_h     = bbox[3] - bbox[1]
        font_size  = _estimate_font_size_pt(bbox_h, num_lines, mapper)
        is_title   = (rtype == 'title')
        alignment  = _detect_alignment(bbox, col_left_px, col_right_px)

        # 尝试拆分多段落（仅对含行级数据的 text 类型）
        if isinstance(lines_raw, list) and len(lines_raw) >= 2 and not is_title:
            para_groups = _split_into_paragraphs(lines_raw, mapper)
        else:
            para_groups = None

        # 基础行高（用于固定行距）
        lh_pt = None
        if num_lines > 1:
            lh = mapper.h(bbox_h) / num_lines
            if lh > font_size * 0.8:
                lh_pt = lh

        def _make_text_para(container, text, para_indent_pt, is_first, sp_before):
            """创建一个文字段落。"""
            p = container.add_paragraph()
            _reset_paragraph_format(p)
            _set_paragraph_spacing(p,
                                   space_before=sp_before if is_first else 0,
                                   space_after=4 if is_title else 1)
            if is_title:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = alignment
            if para_indent_pt > 0 and not is_title:
                p.paragraph_format.first_line_indent = Pt(para_indent_pt)
            if lh_pt is not None:
                _set_paragraph_spacing(p, line_spacing=lh_pt, exact=True)
            run = p.add_run(text)
            if is_title:
                _set_run_font(run, font_size=min(font_size * 1.15, 28),
                              bold=True, east_asia='黑体')
            elif rtype in ('figure_caption', 'table_caption'):
                _set_run_font(run, font_size=max(font_size - 0.5, 6), italic=True)
            elif rtype in ('header', 'footer'):
                _set_run_font(run, font_size=max(font_size - 1, 6),
                              color_rgb=RGBColor(128, 128, 128))
            elif rtype == 'reference':
                _set_run_font(run, font_size=max(font_size - 1, 6))
            else:
                _set_run_font(run, font_size=font_size)
            return p

        last_p = None
        if para_groups and len(para_groups) > 1:
            # 多段落模式：每组行合并为一段
            # 全局首行基准（非第一段，从上方各行最小 x1 中取）
            def grp_x1(grp):
                xs = []
                for ln in grp:
                    tr = ln.get('text_region', [])
                    if tr:
                        xs.append(min(pt[0] for pt in tr))
                return min(xs) if xs else None

            # 计算整体基线 x（所有段落非首行的最小 x1）
            all_non_first_xs = []
            for grp in para_groups:
                for ln in grp[1:]:
                    tr = ln.get('text_region', [])
                    if tr:
                        all_non_first_xs.append(min(pt[0] for pt in tr))
            # 每段的首行 x
            all_first_xs = [grp_x1([grp[0]]) for grp in para_groups]
            all_first_xs = [x for x in all_first_xs if x is not None]
            baseline_x = (min(all_non_first_xs)
                          if all_non_first_xs else
                          (min(all_first_xs) if all_first_xs else None))

            for gi, grp in enumerate(para_groups):
                grp_text = ''.join(ln.get('text', '').strip() for ln in grp)
                if not grp_text.strip():
                    continue
                # 判断本段首行缩进
                fx = grp_x1([grp[0]])
                if fx is not None and baseline_x is not None:
                    indent_px = fx - baseline_x
                    if indent_px >= 12:
                        para_indent_pt = round(mapper.w(indent_px) * 2) / 2.0
                    else:
                        para_indent_pt = 0.0
                else:
                    para_indent_pt = 0.0
                last_p = _make_text_para(
                    container, grp_text, para_indent_pt,
                    is_first=(gi == 0), sp_before=space_before)
        else:
            # 单段落模式（含 title 等）
            first_indent_pt = _detect_first_line_indent(region, mapper)
            last_p = _make_text_para(
                container, text_stripped, first_indent_pt,
                is_first=True, sp_before=space_before)

        ended = bool(text_stripped) and text_stripped[-1] in '。！？…；.!?'
        return last_p, is_title or ended

    # ---- 未知类型 → 截图 ----
    result = _region_or_page_roi()
    if result:
        img_bytes, roi_w, roi_h_px = result
        pw_w = min(mapper.w(roi_w), col_width_pt * 0.98)
        pw_h = mapper.h(roi_h_px) * roi_w / max(roi_h_px, 1)
        pw = min(pw_w, pw_h)
        if space_before > 3:
            _add_spacing_para(container, min(space_before, 12))
        p = container.add_paragraph()
        _reset_paragraph_format(p)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        run = p.add_run()
        try:
            run.add_picture(io.BytesIO(img_bytes), width=Pt(pw))
        except Exception:
            pass
    return None, True


# =====================================================================
# 布局表格工具（跨栏图片支持）
# =====================================================================
def _clear_table_borders(table):
    """移除表格所有可见边框（布局表格专用）。"""
    tblPr = table._element.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        table._element.insert(0, tblPr)
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'none')
        el.set(qn('w:sz'), '0')
        el.set(qn('w:space'), '0')
        el.set(qn('w:color'), 'auto')
        borders.append(el)
    tblPr.append(borders)
    # 单元格间距归零
    tcMar = OxmlElement('w:tblCellMar')
    for edge in ('top', 'left', 'bottom', 'right'):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:w'), '0')
        el.set(qn('w:type'), 'dxa')
        tcMar.append(el)
    tblPr.append(tcMar)


def _set_table_col_widths(table, widths_pt):
    """设置表格列宽（Pt 列表）。"""
    tbl_el = table._element
    tblGrid = tbl_el.find(qn('w:tblGrid'))
    if tblGrid is None:
        tblGrid = OxmlElement('w:tblGrid')
        tbl_el.insert(1, tblGrid)
    for child in list(tblGrid):
        tblGrid.remove(child)
    for wpt in widths_pt:
        gc = OxmlElement('w:gridCol')
        gc.set(qn('w:w'), str(int(wpt * TWIPS_PER_PT)))
        tblGrid.append(gc)
    for row in table.rows:
        for ci, cell in enumerate(row.cells):
            if ci < len(widths_pt):
                tcPr = cell._tc.get_or_add_tcPr()
                old = tcPr.find(qn('w:tcW'))
                if old is not None:
                    tcPr.remove(old)
                tcW = OxmlElement('w:tcW')
                tcW.set(qn('w:w'), str(int(widths_pt[ci] * TWIPS_PER_PT)))
                tcW.set(qn('w:type'), 'dxa')
                tcPr.append(tcW)


def _set_cell_right_margin(cell, margin_pt):
    """设置单元格右内边距（用于布局表格列间距）。"""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = tcPr.find(qn('w:tcMar'))
    if tcMar is None:
        tcMar = OxmlElement('w:tcMar')
        tcPr.append(tcMar)
    right_el = tcMar.find(qn('w:right'))
    if right_el is None:
        right_el = OxmlElement('w:right')
        tcMar.append(right_el)
    right_el.set(qn('w:w'), str(int(margin_pt * TWIPS_PER_PT)))
    right_el.set(qn('w:type'), 'dxa')


def _render_multi_col_zone_as_table(doc, zone_blocks, mapper, img, img_h, img_w,
                                    usable_width_pt):
    """
    使用无边框布局表格渲染含跨栏元素（如图片跨多栏）的多栏 zone。

    外层表格结构（以 004 的四栏布局为例）:
      [Col0 文本单元格] [Col1 文本单元格] [合并单元格: 跨栏图片/图注 + 嵌套二栏表格]

    参考 pdf2docx 对跨栏浮动图片的处理思想，在 Word 中用无边框表格实现等效效果。
    """
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT

    if not zone_blocks:
        return
    num_cols = max(b.get('col_count', 1) for b in zone_blocks)

    # 找出参与跨栏的列索引集合
    spanned_set = set()
    for b in zone_blocks:
        sc = b.get('spanned_cols', [])
        if len(sc) > 1:
            spanned_set.update(sc)

    if not spanned_set:
        # 无跨栏：降级为顺序输出
        for ci in range(num_cols):
            col_blks = sorted([b for b in zone_blocks
                               if b.get('col_index', 0) == ci],
                              key=lambda b: b['bbox'][1])
            cl = min((b['bbox'][0] for b in col_blks), default=0)
            cr = max((b['bbox'][2] for b in col_blks), default=img_w)
            prev_y = 0
            for block in col_blks:
                gap = max(0, block['bbox'][1] - prev_y)
                sp = mapper.h(gap) if (prev_y > 0 and gap > 2) else 0
                _render_block(doc, block, mapper, img, img_h, img_w,
                              usable_width_pt / max(num_cols, 1),
                              cl, cr, img_w * 0.014, space_before=sp)
                prev_y = block['bbox'][3]
        return

    standalone_cols = [ci for ci in range(num_cols) if ci not in spanned_set]
    spanned_cols    = sorted(spanned_set)

    # 列宽分配：列间距固定为可用宽的 3%（最小 18pt），剩余宽度按列数平均分配
    # gutter 通过单元格右内边距实现，不计入列宽，避免 tblCellMar=0 时间距消失
    GUTTER_PT    = max(18.0, usable_width_pt * 0.03)
    num_gutters  = num_cols - 1
    total_gutter = GUTTER_PT * num_gutters
    col_unit     = (usable_width_pt - total_gutter) / num_cols  # 每列纯内容等宽
    col_wpt      = {ci: col_unit for ci in range(num_cols)}

    # 外层表格: standalone 列各一格（纯内容宽） + 一个合并格（spanned列内容宽之和）
    outer_count  = len(standalone_cols) + 1
    outer_widths = [col_wpt[ci] for ci in standalone_cols]
    spanned_total = col_wpt[spanned_cols[0]] * len(spanned_cols) + GUTTER_PT * (len(spanned_cols) - 1)
    outer_widths.append(spanned_total)

    tbl = doc.add_table(rows=1, cols=outer_count)
    tbl.autofit = False
    _clear_table_borders(tbl)
    _set_table_col_widths(tbl, outer_widths)

    row = tbl.rows[0]
    for cell in row.cells:
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
        # 最小化默认空段落高度
        _reset_paragraph_format(cell.paragraphs[0])
        _set_paragraph_spacing(cell.paragraphs[0],
                               line_spacing=MIN_LINE_SPACING_PT, exact=True)

    # 为每个 standalone 单元格设置右内边距作为列间距
    for oi in range(len(standalone_cols)):
        _set_cell_right_margin(row.cells[oi], GUTTER_PT)

    # ── standalone 列 ──
    for oi, ci in enumerate(standalone_cols):
        cell = row.cells[oi]
        col_blks = sorted(
            [b for b in zone_blocks
             if b.get('col_index', 0) == ci
             and len(b.get('spanned_cols', [ci])) == 1],
            key=lambda b: b['bbox'][1])
        cl = min((b['bbox'][0] for b in col_blks), default=0)
        cr = max((b['bbox'][2] for b in col_blks), default=img_w)
        prev_y = 0
        for block in col_blks:
            gap = max(0, block['bbox'][1] - prev_y)
            sp = mapper.h(gap) if (prev_y > 0 and gap > 2) else 0
            _render_block(cell, block, mapper, img, img_h, img_w,
                          col_unit, cl, cr,
                          img_w * 0.014, space_before=sp,
                          in_table_cell=True)
            prev_y = block['bbox'][3]

    # ── 合并格（跨栏区域）──
    mcell = row.cells[-1]
    mwidth = outer_widths[-1]

    # 跨栏块（图片/图注）按 y 排序输出
    span_blks = sorted(
        [b for b in zone_blocks if len(b.get('spanned_cols', [])) > 1],
        key=lambda b: b['bbox'][1])
    sl = min((b['bbox'][0] for b in span_blks), default=0)
    sr = max((b['bbox'][2] for b in span_blks), default=img_w)
    prev_y = 0
    for block in span_blks:
        gap = max(0, block['bbox'][1] - prev_y)
        sp = mapper.h(gap) if (prev_y > 0 and gap > 2) else 0
        _render_block(mcell, block, mapper, img, img_h, img_w,
                      mwidth, sl, sr,
                      img_w * 0.014, space_before=sp, in_table_cell=True)
        prev_y = block['bbox'][3]

    # 各跨栏列的独立内容（跨栏块以下的文本）
    below = {ci: sorted(
                 [b for b in zone_blocks
                  if b.get('col_index', 0) == ci
                  and len(b.get('spanned_cols', [ci])) == 1],
                 key=lambda b: b['bbox'][1])
             for ci in spanned_cols}
    has_below = any(len(v) > 0 for v in below.values())

    if has_below:
        # 子表格：等宽分配（纯内容宽），列间距通过单元格右内边距实现
        n_sub = len(spanned_cols)
        sub_col_w = (mwidth - GUTTER_PT * (n_sub - 1)) / max(n_sub, 1)
        sub_widths = [sub_col_w] * n_sub
        try:
            sub_tbl = mcell.add_table(rows=1, cols=len(spanned_cols))
            sub_tbl.autofit = False
            _clear_table_borders(sub_tbl)
            _set_table_col_widths(sub_tbl, sub_widths)
            sub_row = sub_tbl.rows[0]
            for si, ci in enumerate(spanned_cols):
                sc = sub_row.cells[si]
                sc.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
                _reset_paragraph_format(sc.paragraphs[0])
                _set_paragraph_spacing(sc.paragraphs[0],
                                       line_spacing=MIN_LINE_SPACING_PT,
                                       exact=True)
                # 非最后一列加右内边距
                if si < n_sub - 1:
                    _set_cell_right_margin(sc, GUTTER_PT)
                scol_blks = below[ci]
                scl = min((b['bbox'][0] for b in scol_blks), default=0)
                scr = max((b['bbox'][2] for b in scol_blks), default=img_w)
                prev_y = 0
                for block in scol_blks:
                    gap = max(0, block['bbox'][1] - prev_y)
                    sp = mapper.h(gap) if (prev_y > 0 and gap > 2) else 0
                    _render_block(sc, block, mapper, img, img_h, img_w,
                                  sub_widths[si], scl, scr,
                                  img_w * 0.014, space_before=sp,
                                  in_table_cell=True)
                    prev_y = block['bbox'][3]
        except Exception as e:
            logger.warning(f"嵌套表格失败 ({e})，改为顺序渲染")
            for ci in spanned_cols:
                for block in below[ci]:
                    _render_block(mcell, block, mapper, img, img_h, img_w,
                                  mwidth, 0, img_w,
                                  img_w * 0.014, space_before=0,
                                  in_table_cell=True)


# =====================================================================
# 核心转换函数（zone-based 渲染）
# =====================================================================
def convert_info_docx(img, res, save_folder, img_name, save_intermediate=False):
    """
    将 PaddleOCR 版面分析结果转换为高保真 Word 文档。

    采用 zone-based 渲染策略：
      - 单栏 zone   → 直接输出段落
      - 多栏 zone（无跨栏）→ w:cols 分节
      - 多栏 zone（有跨栏）→ 无边框布局表格（参考 pdf2docx 思路）
    """
    os.makedirs(save_folder, exist_ok=True)
    save_docx_path = os.path.join(save_folder, f"{img_name}.docx")

    try:
        img_h, img_w = img.shape[:2]

        # ==============================================================
        # 1. 文档 + 默认样式
        # ==============================================================
        doc = Document()
        style_normal = doc.styles['Normal']
        style_normal.font.name = 'Times New Roman'
        style_normal._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        style_normal.font.size = Pt(10.5)
        style_normal.paragraph_format.space_after = Pt(0)
        style_normal.paragraph_format.space_before = Pt(0)
        style_normal.paragraph_format.line_spacing = 1.15

        # ==============================================================
        # 2. 页面尺寸与边距 (参考 pdf2docx Page.make_docx)
        # ==============================================================
        sect = doc.sections[0]

        # 根据图像宽高比推断纸张规格
        # 常用纸张 (width_mm, height_mm)
        PAGE_SIZES = [
            ('A3',       297, 420),
            ('A4',       210, 297),
            ('A5',       148, 210),
            ('B4',       250, 353),
            ('B5',       176, 250),
            ('Letter',   216, 279),
            ('Legal',    216, 356),
            ('Tabloid',  279, 432),
        ]
        img_ratio = img_w / max(img_h, 1)
        best_name, best_w_mm, best_h_mm = 'A4', 210, 297
        best_diff = float('inf')
        for name, wm, hm in PAGE_SIZES:
            for portrait in [True, False]:
                rw, rh = (wm, hm) if portrait else (hm, wm)
                diff = abs(img_ratio - rw / rh)
                if diff < best_diff:
                    best_diff, best_name = diff, name
                    best_w_mm, best_h_mm = rw, rh
        is_landscape = (best_w_mm > best_h_mm)
        page_w_pt = best_w_mm / 25.4 * 72
        page_h_pt = best_h_mm / 25.4 * 72

        sect.page_width  = Cm(best_w_mm / 10)
        sect.page_height = Cm(best_h_mm / 10)
        sect.orientation = WD_ORIENT.LANDSCAPE if is_landscape else WD_ORIENT.PORTRAIT

        all_x1 = [r['bbox'][0] for r in res if 'bbox' in r]
        all_x2 = [r['bbox'][2] for r in res if 'bbox' in r]
        all_y1 = [r['bbox'][1] for r in res if 'bbox' in r]
        if all_x1:
            min_x, max_x = min(all_x1), max(all_x2)
            min_y = min(all_y1) if all_y1 else 0
            ml = min(max(36, page_w_pt * min_x / max(img_w, 1)), 90)
            mr = min(max(36, page_w_pt * (1 - max_x / max(img_w, 1))), 90)
            mt = max(36, min(page_h_pt * min_y / max(img_h, 1), 100))
            mb = 54.0
        else:
            ml = mr = mt = mb = DEFAULT_MARGIN_PT

        sect.left_margin   = Pt(ml)
        sect.right_margin  = Pt(mr)
        sect.top_margin    = Pt(mt)
        sect.bottom_margin = Pt(mb)

        mapper = CoordMapper(img_w, img_h,
                             page_width_pt=page_w_pt,
                             page_height_pt=page_h_pt,
                             margin_left_pt=ml,
                             margin_right_pt=mr,
                             margin_top_pt=mt,
                             margin_bottom_pt=mb)
        usable_w_pt = page_w_pt - ml - mr

        # ==============================================================
        # 3. 可选：生成可视化排序图
        # ==============================================================
        if save_intermediate:
            debug_dir = os.path.join(save_folder, img_name, "debug")
            sorted_path = os.path.join(debug_dir, "sorted.jpg")
            if not os.path.exists(sorted_path):
                os.makedirs(debug_dir, exist_ok=True)
                vis = draw_sorted_layout(img, res)
                cv2.imwrite(sorted_path, vis)

        # ==============================================================
        # 4. 将 res 按 col_count 分组为 zones
        # ==============================================================
        zones = []
        cur_zone, cur_cols = [], None
        for block in res:
            bc = block.get('col_count', 1)
            if cur_cols is None:
                cur_cols = bc
            if bc == cur_cols:
                cur_zone.append(block)
            else:
                zones.append((cur_cols, cur_zone))
                cur_zone, cur_cols = [block], bc
        if cur_zone:
            zones.append((cur_cols, cur_zone))

        # ==============================================================
        # 5. 逐 zone 渲染
        # ==============================================================
        def _copy_page(src, dst):
            dst.page_width    = src.page_width
            dst.page_height   = src.page_height
            dst.left_margin   = src.left_margin
            dst.right_margin  = src.right_margin
            dst.top_margin    = src.top_margin
            dst.bottom_margin = src.bottom_margin

        in_multicol = False   # 是否正处于 w:cols 多栏节
        char_w_px = img_w * 0.014

        for zi, (zone_cols, zone_blocks) in enumerate(zones):
            # 预计算各列像素左右边界
            col_px = {}
            for b in zone_blocks:
                ci = b.get('col_index', 0)
                if ci not in col_px:
                    col_px[ci] = [b['bbox'][0], b['bbox'][2]]
                else:
                    col_px[ci][0] = min(col_px[ci][0], b['bbox'][0])
                    col_px[ci][1] = max(col_px[ci][1], b['bbox'][2])

            if zone_cols == 1:
                # ===== 单栏 zone =====
                if in_multicol:
                    ns = doc.add_section(WD_SECTION.CONTINUOUS)
                    _copy_page(sect, ns)
                    _set_section_columns(ns._sectPr, 1)
                    in_multicol = False
                cl = col_px.get(0, [0, img_w])[0]
                cr = col_px.get(0, [0, img_w])[1]
                prev_y = 0
                for block in zone_blocks:
                    gap = max(0, block['bbox'][1] - prev_y)
                    sp = mapper.h(gap) if (prev_y > 0 and gap > 2) else 0
                    _render_block(doc, block, mapper, img, img_h, img_w,
                                  usable_w_pt, cl, cr, char_w_px,
                                  space_before=sp)
                    prev_y = block['bbox'][3]

            else:
                # ===== 多栏 zone =====
                has_spans = any(len(b.get('spanned_cols', [])) > 1
                                for b in zone_blocks)

                if has_spans:
                    # — 布局表格方式（跨栏图片）—
                    if in_multicol:
                        ns = doc.add_section(WD_SECTION.CONTINUOUS)
                        _copy_page(sect, ns)
                        _set_section_columns(ns._sectPr, 1)
                        in_multicol = False
                    elif zi == 0:
                        _set_section_columns(sect._sectPr, 1)

                    _render_multi_col_zone_as_table(
                        doc, zone_blocks, mapper, img, img_h, img_w,
                        usable_w_pt)

                else:
                    # — w:cols 分栏方式 —
                    # 列间距：3% 页宽，最少 24pt，最多 54pt
                    col_spacing_pt = max(24.0, min(54.0, usable_w_pt * 0.03))
                    # 每列可用宽度 = (总宽 - 所有间距) / 列数
                    col_usable = ((usable_w_pt - col_spacing_pt * (zone_cols - 1))
                                  / max(zone_cols, 1))
                    col_widths_pt = [col_usable] * zone_cols
                    if zi == 0:
                        _set_section_columns(sect._sectPr, zone_cols,
                                             col_widths_pt=col_widths_pt,
                                             spacing_pt=col_spacing_pt)
                    else:
                        ns = doc.add_section(WD_SECTION.CONTINUOUS)
                        _copy_page(sect, ns)
                        _set_section_columns(ns._sectPr, zone_cols,
                                             col_widths_pt=col_widths_pt,
                                             spacing_pt=col_spacing_pt)
                    in_multicol = True
                    current_ci = -1
                    prev_y = 0
                    last_p = None
                    for block in zone_blocks:
                        bci = block.get('col_index', 0)
                        if bci != current_ci and current_ci >= 0:
                            if last_p is not None:
                                last_p.add_run().add_break(WD_BREAK.COLUMN)
                            else:
                                p = doc.add_paragraph()
                                _reset_paragraph_format(p)
                                p.add_run().add_break(WD_BREAK.COLUMN)
                            prev_y = 0
                        current_ci = bci
                        cl = col_px.get(bci, [0, img_w])[0]
                        cr = col_px.get(bci, [0, img_w])[1]
                        gap = max(0, block['bbox'][1] - prev_y)
                        sp = mapper.h(gap) if (prev_y > 0 and gap > 2) else 0
                        last_p, _ = _render_block(
                            doc, block, mapper, img, img_h, img_w,
                            col_usable, cl, cr, char_w_px, space_before=sp)
                        prev_y = block['bbox'][3]

        # 多栏收尾→追加单栏节
        if in_multicol:
            es = doc.add_section(WD_SECTION.CONTINUOUS)
            _copy_page(sect, es)
            _set_section_columns(es._sectPr, 1)

        # ==============================================================
        # 6. 删除末尾多余空段落（避免导致额外空白页）
        # ==============================================================
        # Word 新文档带一个默认空段落； CONTINUOUS 节叆可能再带来空行。
        # 删除文档最后所有没有内容的段落元素（保留最后一个， Word 必须有。
        body = doc.element.body
        paras_in_body = body.findall(qn('w:p'))
        # 从末尾往前找最后一个有内容的段落
        last_content_idx = -1
        for idx, p_el in enumerate(paras_in_body):
            runs = p_el.findall('.//' + qn('w:t'))
            pics = p_el.findall('.//' + qn('a:blip'), {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'})
            if runs or pics:
                last_content_idx = idx
        # 删除 last_content_idx 之后且不属于 sectPr 的空段落（保留最后一个）
        to_remove = []
        for idx, p_el in enumerate(paras_in_body):
            if idx > last_content_idx:
                # 检查该段落是否有 sectPr（节定义）——有的话不能删
                has_sect = p_el.find(qn('w:pPr') + '/' + qn('w:sectPr')) is not None
                if not has_sect and idx > last_content_idx + 1:
                    to_remove.append(p_el)
        for p_el in to_remove:
            body.remove(p_el)

        # ==============================================================
        # 7. 保存
        # ==============================================================
        doc.save(save_docx_path)
        logger.info(f"[{img_name}] Word 文档已保存: {save_docx_path}")

    except Exception as e:
        logger.error(f"[{img_name}] Word 转换失败: {e}")
        import traceback
        traceback.print_exc()


# =====================================================================
# 排序函数
# =====================================================================
def sorted_layout_boxes(res, w):
    """
    版面区域排序算法：宽块分 Zone + 文本列聚类。

    改进重点（相比旧版）:
      1. figure / figure_caption / table_caption / equation 从列聚类中剔除
         ── 这些块可能跨越多列，不应影响文本列检测
      2. 聚类阈值提高到 0.13，避免"副标题/字幕"单独成列
      3. MAX_COLS = 4（支持报纸四栏布局）
      4. 为 figure/caption 块检测跨列范围，标注 spanned_cols
      5. 所有块统一标注: col_count / col_index / spanned_cols / layout
    """
    MAX_COLS       = 4
    CLUSTER_THRESH = 0.13
    FIGURE_TYPES   = {'figure', 'equation', 'figure_caption', 'table_caption'}

    if len(res) <= 1:
        for r in res:
            r.update(col_count=1, col_index=0, spanned_cols=[0], layout='single')
        return res

    get_y1 = lambda b: b['bbox'][1]
    get_x1 = lambda b: b['bbox'][0]
    get_bw = lambda b: b['bbox'][2] - b['bbox'][0]

    res.sort(key=get_y1)

    # -----------------------------------------------------------------
    # Zone 切分策略：
    #   1. 宽块（>55%）单独成 zone
    #   2. 其余窄块按全局 y 重叠关系分组：
    #      先把所有窄块按 y-interval 合并成连通组（有 y 重叠则同组），
    #      然后对每个连通组再按是否存在 y 重叠进一步按 y 间隙切分。
    #      最终单个块的组即为"孤立块" zone（如 byline），
    #      多块的组整体作为需要列聚类的多栏 zone。
    # -----------------------------------------------------------------
    def _has_y_overlap(b1, b2, min_overlap=4):
        return (min(b1['bbox'][3], b2['bbox'][3])
                - max(b1['bbox'][1], b2['bbox'][1])) >= min_overlap

    def _split_narrow_batch(batch, overlap_fn):
        """将窄块批次按纵向重叠关系分成若干 zone 列表。
        用 union-find 对有纵向重叠的块连通，无重叠的块单独成组。"""
        n = len(batch)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for i in range(n):
            for j in range(i + 1, n):
                if overlap_fn(batch[i], batch[j]):
                    union(i, j)

        groups = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(batch[i])

        # 按组内最小 y1 排序各组，再按 y1 排序组内块
        result = []
        for g in sorted(groups.values(), key=lambda g: min(b['bbox'][1] for b in g)):
            result.append(sorted(g, key=get_y1))
        return result

    zones = []
    narrow_batch = []
    for box in res:
        if get_bw(box) > w * 0.55:
            if narrow_batch:
                for grp in _split_narrow_batch(narrow_batch, _has_y_overlap):
                    zones.append(grp)
                narrow_batch = []
            zones.append([box])
        else:
            narrow_batch.append(box)
    if narrow_batch:
        for grp in _split_narrow_batch(narrow_batch, _has_y_overlap):
            zones.append(grp)

    final = []
    for zone in zones:
        text_boxes = [b for b in zone if b['type'].lower() not in FIGURE_TYPES]
        fig_boxes  = [b for b in zone if b['type'].lower() in FIGURE_TYPES]

        if len(text_boxes) <= 1:
            for b in zone:
                b.update(col_count=1, col_index=0,
                         spanned_cols=[0], layout='single')
            final.extend(sorted(zone, key=get_y1))
            continue

        # 对文本块按 x1 聚类（最近邻贪心）
        text_boxes.sort(key=get_x1)
        columns, col_avgs = [], []
        for box in text_boxes:
            x1 = get_x1(box)
            best_ci, best_d = None, float('inf')
            for ci, avg in enumerate(col_avgs):
                d = abs(x1 - avg)
                if d < best_d:
                    best_d, best_ci = d, ci
            if best_ci is not None and best_d < w * CLUSTER_THRESH:
                columns[best_ci].append(box)
                col_avgs[best_ci] = (sum(get_x1(b) for b in columns[best_ci])
                                     / len(columns[best_ci]))
            else:
                columns.append([box])
                col_avgs.append(x1)

        # 按列中心 x 排序
        paired = sorted(zip(col_avgs, columns), key=lambda kv: kv[0])
        col_avgs = [p[0] for p in paired]
        columns  = [p[1] for p in paired]

        # 限制 ≤ MAX_COLS（合并间距最小的相邻列）
        while len(columns) > MAX_COLS:
            min_gap, merge_at = float('inf'), 0
            for ci in range(len(columns) - 1):
                rx  = max(b['bbox'][2] for b in columns[ci])
                lx  = min(get_x1(b) for b in columns[ci + 1])
                gap = lx - rx
                if gap < min_gap:
                    min_gap, merge_at = gap, ci
            columns[merge_at] = columns[merge_at] + columns[merge_at + 1]
            columns.pop(merge_at + 1)
            col_avgs = [sum(get_x1(b) for b in col) / len(col)
                        for col in columns]

        num_cols = len(columns)

        # 每列像素边界（用于跨栏检测）
        # 右边界使用"去掉最大一个离群值"的方式（排除 byline 等宽度异常块的影响）
        # 参考：取 x2 升序后的倒数第 2 个值作为右边界上限
        col_bounds = []
        for col in columns:
            x1_min = min(b['bbox'][0] for b in col)
            x2s = sorted(b['bbox'][2] for b in col)
            x2_robust = x2s[max(0, len(x2s) - 2)]  # 去掉最大一个离群值
            col_bounds.append((x1_min, x2_robust))

        # 标注文本块
        for ci, col in enumerate(columns):
            col.sort(key=get_y1)
            for b in col:
                b.update(col_count=num_cols, col_index=ci,
                         spanned_cols=[ci],
                         layout='double' if num_cols > 1 else 'single')

        # 标注 figure/caption 块：检测跨列范围
        for b in fig_boxes:
            bx1, _, bx2, _ = b['bbox']
            spanned = [ci for ci, (cx1, cx2) in enumerate(col_bounds)
                       if min(bx2, cx2) - max(bx1, cx1) > 0]
            if not spanned:
                # 没有与任何列重叠，取最近列
                dists = [abs((bx1 + bx2) / 2 - (cx1 + cx2) / 2)
                         for cx1, cx2 in col_bounds]
                spanned = [dists.index(min(dists))]
            b.update(col_count=num_cols, col_index=spanned[0],
                     spanned_cols=spanned,
                     layout=('multi'  if len(spanned) > 1 else
                             'double' if num_cols  > 1 else 'single'))

        # 输出顺序：按 col_index 分组，列内按 y 排序
        zone_out = []
        for ci in range(num_cols):
            col_blks = [b for b in zone if b.get('col_index', 0) == ci]
            col_blks.sort(key=get_y1)
            zone_out.extend(col_blks)
        final.extend(zone_out)

    return final
