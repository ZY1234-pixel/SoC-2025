"""版面分析层：坐标映射、区域拆分、列检测、
paragraph detection, reading-order sorting, and style inference."""

from docflow.model.page import CoordMapper
from docflow.layout.sorter import sort_layout
from docflow.layout.paragraph_detector import split_into_paragraphs, detect_first_line_indent
from docflow.layout.style_inferrer import infer_block_styles
