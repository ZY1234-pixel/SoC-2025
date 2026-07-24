"""Pixel-level style evidence used by document analysis."""

from docflow.layout.color_inferrer import infer_crop_style, infer_table_row_fills
from docflow.layout.font_classifier import FONT_FAMILY_BY_LABEL, FontClassifier

__all__ = [
    "FONT_FAMILY_BY_LABEL",
    "FontClassifier",
    "infer_crop_style",
    "infer_table_row_fills",
]
