"""SLANet Plus 结构模型，使用 OpenVINO 执行并复用现有 OCR 结果。"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

import numpy as np

from docflow.inference import OpenVINOInferSession


_SESSION_PATCH_LOCK = Lock()


class SLANetPlusOpenVINOInferSession(OpenVINOInferSession):
    """把 rapid_table 的 ONNX session 接口映射到 OpenVINO。"""

    def __init__(self, config) -> None:
        model_path = Path(config["model_dir_or_path"])
        super().__init__({"model_path": str(model_path)})
        character_path = model_path.with_name("character.txt")
        if not character_path.is_file():
            raise FileNotFoundError(f"SLANet Plus character file not found: {character_path}")
        self._characters = character_path.read_text(encoding="utf-8").splitlines()

    def __call__(self, input_content):
        value = np.asarray(input_content, dtype=np.float32)
        if value.ndim == 3:
            value = value[None, ...]
        if value.ndim != 4:
            raise ValueError(f"SLANet Plus expects NCHW input, got {value.shape}")
        return self.run(None, {self.get_input_names()[0]: value})

    def have_key(self, key: str = "character") -> bool:
        return key == "character"

    def get_character_list(self, key: str = "character"):
        if key != "character":
            raise KeyError(key)
        return list(self._characters)


class SLANetPlusTableEngine:
    """适配 Current RapidAITableRecognizer 的 SLANet Plus 调用接口。"""

    def __init__(self, model_path: str | Path) -> None:
        import rapid_table.inference_engine.onnxruntime as onnxruntime_engine
        from rapid_table import RapidTable
        from rapid_table.utils.typings import EngineType, ModelType, RapidTableInput

        model_path = Path(model_path)
        with _SESSION_PATCH_LOCK:
            original = onnxruntime_engine.OrtInferSession
            onnxruntime_engine.OrtInferSession = SLANetPlusOpenVINOInferSession
            try:
                self.table = RapidTable(
                    RapidTableInput(
                        model_type=ModelType.SLANETPLUS,
                        model_dir_or_path=model_path,
                        engine_type=EngineType.ONNXRUNTIME,
                        use_ocr=False,
                    )
                )
            finally:
                onnxruntime_engine.OrtInferSession = original

    def __call__(
        self,
        image: np.ndarray,
        ocr_result: list | None = None,
        enhance_box_line: bool = True,
        rotated_fix: bool = False,
        col_threshold: int = 15,
        row_threshold: int = 10,
    ):
        del enhance_box_line, rotated_fix, col_threshold, row_threshold
        started = time.perf_counter()
        structures, cell_bboxes = self.table.table_structure([image])
        logic_points = self.table.table_matcher.decode_logic_points(structures)
        dt_boxes, rec_results = self._format_ocr(ocr_result or [], image.shape[:2])
        htmls = self.table.table_matcher(
            structures,
            cell_bboxes,
            [dt_boxes],
            [rec_results],
        )
        elapsed = time.perf_counter() - started
        return (
            htmls[0] if htmls else "",
            elapsed,
            cell_bboxes[0] if cell_bboxes else np.empty((0, 8), dtype=np.float32),
            logic_points[0] if logic_points else np.empty((0, 4), dtype=np.int32),
            None,
        )

    @staticmethod
    def _format_ocr(ocr_result: list, image_shape: tuple[int, int]):
        height, width = image_shape
        valid_items = [item for item in ocr_result if len(item) > 1 and str(item[1]).strip()]
        if not valid_items:
            return np.empty((0, 4), dtype=np.float32), []
        polygons = np.asarray(
            [np.asarray(item[0], dtype=np.float32).reshape(-1, 2) for item in valid_items],
            dtype=np.float32,
        )
        min_coords = np.maximum(polygons[..., :2].min(axis=1), 0)
        max_coords = np.minimum(polygons[..., :2].max(axis=1), [width, height])
        boxes = np.hstack([min_coords, max_coords]).astype(np.float32)
        results = [
            (str(item[1]), float(item[2]) if len(item) > 2 else 1.0)
            for item in valid_items
        ]
        return boxes, results
