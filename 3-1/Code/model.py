"""Model factory for ESDNet and ESDNet-Lite."""

from __future__ import annotations

import sys
from pathlib import Path

ARCH_DIR = Path(__file__).resolve().parent / 'archs'
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))

from ESDNet_arch import ESDNet  # noqa: E402
from ESDNet_lite_arch import ESDNetLite  # noqa: E402

MODEL_CONFIGS = {
    'full': {
        'model_type': 'full',
        'en_feature_num': 48,
        'en_inter_num': 32,
        'de_feature_num': 64,
        'de_inter_num': 32,
        'sam_number': 1,
    },
    'lite-s': {
        'model_type': 'lite',
        'en_feature_num': 16,
        'en_inter_num': 8,
        'de_feature_num': 20,
        'de_inter_num': 8,
        'sam_number': 1,
    },
    'lite-xs': {
        'model_type': 'lite',
        'en_feature_num': 12,
        'en_inter_num': 6,
        'de_feature_num': 16,
        'de_inter_num': 6,
        'sam_number': 1,
    },
}


def get_model_config(preset: str) -> dict:
    key = preset.lower()
    if key not in MODEL_CONFIGS:
        raise ValueError(f'Unsupported model preset: {preset}')
    return dict(MODEL_CONFIGS[key])


def build_model(preset: str):
    config = get_model_config(preset)
    model_type = config.pop('model_type')
    if model_type == 'full':
        return ESDNet(**config)
    return ESDNetLite(**config)
