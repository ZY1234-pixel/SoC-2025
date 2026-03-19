"""版面恢复管线的配置项。"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RecoveryConfig:
    """版面恢复管线的全部配置选项。"""
    # 版面分析
    max_cols: int = 4
    column_cluster_thresh: float = 0.13
    column_confidence_min: float = 0.55
    zone_strip_height_ratio: float = 0.12
    wide_block_thresh: float = 0.55
    paragraph_indent_px: float = 12.0
    paragraph_list_marker_enabled: bool = True
    align_justify_min_lines: int = 3

    # DOCX 渲染
    default_font: str = "Times New Roman"
    default_cjk_font: str = "\u5b8b\u4f53"
    default_font_size_pt: float = 10.5
    title_font: str = "\u9ed1\u4f53"
    default_line_spacing: float = 1.05
    docx_prefer_native_columns: bool = True
    docx_column_gap_twips: int = 720
    docx_preserve_visual_line_breaks: bool = True
    docx_preserve_breaks_on_ambiguous_justify: bool = True

    # 页面布局
    min_margin_pt: float = 36.0
    max_margin_pt: float = 90.0
    default_margin_pt: float = 54.0
    bottom_margin_pt: float = 54.0

    # 调试
    save_debug: bool = False
    debug_dir: Optional[str] = None
