"""Standalone Detectron2/MaskDINO configuration loader."""

from typing import Iterable
from pathlib import Path

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from .models.maskdino import add_maskdino_config


from .settings import DEFAULT_CONFIG, PACKAGE_ROOT


def load_config(config_file: Path = DEFAULT_CONFIG, opts: Iterable[str] = ()):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(str(Path(config_file).resolve()))
    cfg.merge_from_list(list(opts))
    # IDEs often launch with a working directory other than the repository root.
    # Keep all project paths deterministic and independent of that setting.
    weights = Path(cfg.MODEL.WEIGHTS)
    output_dir = Path(cfg.OUTPUT_DIR)
    if not weights.is_absolute():
        cfg.MODEL.WEIGHTS = str((PACKAGE_ROOT / weights).resolve())
    if not output_dir.is_absolute():
        cfg.OUTPUT_DIR = str((PACKAGE_ROOT / output_dir).resolve())
    cfg.freeze()
    return cfg
