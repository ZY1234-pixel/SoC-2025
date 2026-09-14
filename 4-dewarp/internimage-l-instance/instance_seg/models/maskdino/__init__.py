"""Register only the deployed MaskDINO model and heads."""
from . import modeling
from .config import add_maskdino_config
from .maskdino import MaskDINO
