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

import os
import cv2
from copy import deepcopy

from docx import Document
from docx import shared
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from ppstructure.recovery.table_process import HtmlToDocx

from ppocr.utils.logging import get_logger

logger = get_logger()

def convert_info_docx(img, res, save_folder, img_name):
    os.makedirs(save_folder, exist_ok=True)
    save_docx_path = os.path.join(save_folder, f"{img_name}.docx")
    
    try:
        document = Document()
        document.styles['Normal'].font.name = 'Times New Roman'
        document.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), u'宋体')
        document.styles['Normal'].font.size = Pt(11)

        current_cols = 1
        current_col_idx = -1
        current_col_x_min = 9999  # 记录当前栏最左侧的基准线
        last_p = None
        last_text_ended_sentence = True

        # 动态计算缩进阈值：大约取图片宽度的 1.5% 作为两个中文字符的宽度
        img_h, img_w, _ = img.shape
        indent_threshold = img_w * 0.015 

        for i, region in enumerate(res):
            target_cols = region.get('col_count', 1)
            target_col_idx = region.get('col_index', 0)

            # ==========================================
            # 1. 分节与分栏控制 (记录栏基准线)
            # ==========================================
            if target_cols != current_cols:
                if i == 0:
                    sectPr = document.sections[0]._sectPr
                else:
                    new_sect = document.add_section(WD_SECTION.CONTINUOUS)
                    sectPr = new_sect._sectPr
                
                cols_list = sectPr.xpath('./w:cols')
                cols = cols_list[0] if cols_list else OxmlElement('w:cols')
                cols.set(qn('w:num'), str(target_cols))
                cols.set(qn('w:space'), '425')
                cols.set(qn('w:equalWidth'), '1')
                if not cols_list: 
                    sectPr.append(cols)
                
                current_cols = target_cols
                current_col_idx = target_col_idx
                current_col_x_min = region['bbox'][0] # 新栏，重置左侧基准线

            if target_cols > 1 and target_col_idx != current_col_idx:
                if last_p is not None:
                    last_p.add_run().add_break(WD_BREAK.COLUMN)
                else:
                    last_p = document.add_paragraph()
                    last_p.add_run().add_break(WD_BREAK.COLUMN)
                current_col_idx = target_col_idx
                current_col_x_min = region['bbox'][0] # 新栏，重置左侧基准线

            # 更新当前栏的最左基准线
            x1 = region['bbox'][0]
            current_col_x_min = min(current_col_x_min, x1)

            # ==========================================
            # 2. 文本提取与智能缩进判定
            # ==========================================
            region_type = region.get('type', 'text').lower()

            if region_type in ['figure', 'figure_caption', 'equation', 'table']:
                box = region['bbox']
                rx1, ry1, rx2, ry2 = map(int, box)
                roi = img[ry1:ry2, rx1:rx2]
                tmp_img_path = os.path.join(save_folder, f"tmp_{img_name}_{i}.jpg")
                cv2.imwrite(tmp_img_path, roi)
                
                try:
                    img_width = Inches(5.5) if current_cols == 1 else Inches(1.5)
                    document.add_picture(tmp_img_path, width=img_width)
                except Exception:
                    pass
                if os.path.exists(tmp_img_path): os.remove(tmp_img_path)
                
                last_p = None 
                last_text_ended_sentence = True

            elif region_type in ['text', 'title']:
                text_content = ""
                if isinstance(region.get('res'), list):
                    # 【优化】：中文排版不要用空格拼合断行
                    text_content = "".join([r.get('text', '').strip() for r in region['res']])
                elif isinstance(region.get('res'), str):
                    text_content = region['res']
                elif isinstance(region.get('res'), dict):
                    text_content = region['res'].get('text', '')
                
                # 检查 OCR 是否抓取到了前置空格
                starts_with_space = text_content.startswith(' ') or text_content.startswith('　')
                text_content = text_content.strip()
                if not text_content: continue

                # 【核心逻辑】：物理缩进判定
                # 如果当前块的左边缘 x1，明显大于本栏的基准线，说明它缩进了！
                is_indented_by_coord = (x1 - current_col_x_min) > indent_threshold

                is_new_paragraph = (
                    last_text_ended_sentence or 
                    region_type == 'title' or 
                    last_p is None or 
                    is_indented_by_coord or 
                    starts_with_space
                )

                if is_new_paragraph:
                    last_p = document.add_paragraph(text_content)
                    if region_type == 'title':
                        for run in last_p.runs: run.bold = True
                    
                    # 【排版美化】：如果是新段落且有物理缩进，自动在 Word 中设置首行缩进 2 字符 (约 22 磅)
                    if region_type == 'text' and (is_indented_by_coord or starts_with_space):
                        last_p.paragraph_format.first_line_indent = Pt(22)
                else:
                    # 确认为同一段落的断裂句，无缝拼接
                    last_p.add_run(text_content)

                if region_type == 'title':
                    last_text_ended_sentence = True
                else:
                    end_chars = tuple("""。！？！…”"’'.\n>:-""")
                    last_text_ended_sentence = text_content.endswith(end_chars)
                        
        document.save(save_docx_path)
        print(f"[{img_name}] 高级多栏 Word 渲染完毕（已开启物理缩进识别）！")

    except Exception as e:
        print(f"[{img_name}] Word 转换失败: {e}")
    
def sorted_layout_boxes(res, w):
    """
    优化版：基于“宽块分割 (Zone)” + “左边缘聚类 (Left-edge Column)” 的排序算法
    """
    if len(res) <= 1:
        return res

    # 辅助提取坐标的函数，使代码更清晰
    def get_y1(box): return box['bbox'][1]
    def get_x1(box): return box['bbox'][0]
    def get_w(box): return box['bbox'][2] - box['bbox'][0]

    # 1. 初始按 Y 坐标排，确保整体自上而下的基本顺序
    res.sort(key=get_y1)

    zones = []
    current_zone = []
    
    # 2. 切分“水平区域 (Zones)”
    # 核心逻辑：如果遇到宽度超过页面 60% 的巨型块（如顶部0号大标题），直接将其作为横向分割线
    for box in res:
        if get_w(box) > w * 0.6: 
            if current_zone:
                zones.append(current_zone)
                current_zone = []
            zones.append([box]) # 大块独占一个 Zone
        else:
            current_zone.append(box)
            
    if current_zone:
        zones.append(current_zone)

    final_sorted_res = []

    # 3. 在每个区域内，切分“垂直专栏 (Columns)”
    for zone in zones:
        if len(zone) <= 1:
            final_sorted_res.extend(zone)
            continue
            
        # 核心修正：严格按块的【左边缘 (x1)】进行排序，忽略右边缘。
        # 这彻底避免了跨栏的宽图片扰乱后续排版
        zone.sort(key=get_x1)
        
        columns = []
        current_col = [zone[0]]
        col_x1_avg = get_x1(zone[0])
        
        for box in zone[1:]:
            # 容差：如果当前块的左边缘，与当前列的平均左边缘相差不到页面宽度的 10%
            if abs(get_x1(box) - col_x1_avg) < (w * 0.1):
                current_col.append(box)
                # 动态更新该列的平均左边缘，提高对轻微缩进的宽容度
                col_x1_avg = sum(get_x1(b) for b in current_col) / len(current_col)
            else:
                columns.append(current_col)
                current_col = [box]
                col_x1_avg = get_x1(box)
        
        if current_col:
            columns.append(current_col)
            
        # 4. 在同一个专栏内，严格按 Y 坐标自上而下排序
        for col in columns:
            col.sort(key=get_y1)
            final_sorted_res.extend(col)
            
    # 5. 添加 layout 字段
    for item in final_sorted_res:
        item["layout"] = "single" 
        
    return final_sorted_res

# def sorted_layout_boxes(res, w):
#     """
#     Sort text boxes in order from top to bottom, left to right
#     args:
#         res(list):ppstructure results
#     return:
#         sorted results(list)
#     """
#     num_boxes = len(res)
#     if num_boxes == 1:
#         res[0]["layout"] = "single"
#         return res

#     sorted_boxes = sorted(res, key=lambda x: (x["bbox"][1], x["bbox"][0]))
#     _boxes = list(sorted_boxes)

#     new_res = []
#     res_left = []
#     res_right = []
#     i = 0

#     while True:
#         if i >= num_boxes:
#             break
#         if i == num_boxes - 1:
#             if (
#                 _boxes[i]["bbox"][1] > _boxes[i - 1]["bbox"][3]
#                 and _boxes[i]["bbox"][0] < w / 2
#                 and _boxes[i]["bbox"][2] > w / 2
#             ):
#                 new_res += res_left
#                 new_res += res_right
#                 _boxes[i]["layout"] = "single"
#                 new_res.append(_boxes[i])
#             else:
#                 if _boxes[i]["bbox"][2] > w / 2:
#                     _boxes[i]["layout"] = "double"
#                     res_right.append(_boxes[i])
#                     new_res += res_left
#                     new_res += res_right
#                 elif _boxes[i]["bbox"][0] < w / 2:
#                     _boxes[i]["layout"] = "double"
#                     res_left.append(_boxes[i])
#                     new_res += res_left
#                     new_res += res_right
#             res_left = []
#             res_right = []
#             break
#         elif _boxes[i]["bbox"][0] < w / 4 and _boxes[i]["bbox"][2] < 3 * w / 4:
#             _boxes[i]["layout"] = "double"
#             res_left.append(_boxes[i])
#             i += 1
#         elif _boxes[i]["bbox"][0] > w / 4 and _boxes[i]["bbox"][2] > w / 2:
#             _boxes[i]["layout"] = "double"
#             res_right.append(_boxes[i])
#             i += 1
#         else:
#             new_res += res_left
#             new_res += res_right
#             _boxes[i]["layout"] = "single"
#             new_res.append(_boxes[i])
#             res_left = []
#             res_right = []
#             i += 1
#     if res_left:
#         new_res += res_left
#     if res_right:
#         new_res += res_right
#     return new_res
