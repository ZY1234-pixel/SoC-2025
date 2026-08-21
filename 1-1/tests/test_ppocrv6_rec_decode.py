from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PADDLE_RUNTIME = PROJECT_ROOT / "Code" / "third_party" / "paddle_runtime"
if str(PADDLE_RUNTIME) not in sys.path:
    sys.path.insert(0, str(PADDLE_RUNTIME))

from ppocr.postprocess.rec_postprocess import CTCLabelDecode


def test_ppocrv6_dict_leading_empty_line_does_not_shift_ctc_indices() -> None:
    dict_path = (
        PROJECT_ROOT
        / "Code"
        / "models_openvino"
        / "PP-OCRv6_small_rec_openvino"
        / "ppocrv6_dict.txt"
    )
    decoder = CTCLabelDecode(character_dict_path=str(dict_path), use_space_char=True)

    assert len(decoder.character) == 18710
    assert decoder.character[62] == "T"
    assert decoder.character[76] == "h"
    assert decoder.character[73] == "e"
    assert decoder.character[-1] == " "

    preds = np.zeros((1, 4, len(decoder.character)), dtype=np.float32)
    for step, char_index in enumerate([62, 76, 73, len(decoder.character) - 1]):
        preds[0, step, char_index] = 1.0

    decoded = decoder(preds)
    assert decoded[0][0] == "The "
