"""Pixel-level appearance evidence used by document analysis."""

from docflow.appearance.color_inferrer import infer_background_extent, infer_crop_style, infer_table_row_fills
from docflow.appearance.font_classifier import FONT_FAMILY_BY_LABEL, FontClassifier

__all__ = [
    "FONT_FAMILY_BY_LABEL",
    "FontClassifier",
    "infer_background_extent",
    "infer_crop_style",
    "infer_table_row_fills",
]
