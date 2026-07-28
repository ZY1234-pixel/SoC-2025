# Copyright (c) 2020 PaddlePaddle Authors.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import pyclipper


ROOT = Path(__file__).resolve().parent


class OpenVinoOCR:
    """使用 OpenVINO CPU 后端执行 PP-OCRv6 small DET 和 REC。"""

    def __init__(
        self,
        model_dir: str | Path = ROOT / "models",
        cpu_threads: int = 10,
        rec_batch_size: int = 6,
        score_threshold: float = 0.5,
    ) -> None:
        """加载模型；实例应在进程启动时创建并持续复用。"""
        model_dir = Path(model_dir)
        required = {
            "det": model_dir / "det" / "model.xml",
            "rec": model_dir / "rec" / "model.xml",
            "dict": model_dir / "rec" / "ppocrv6_dict.txt",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing deployment files: " + ", ".join(missing))
        if cpu_threads < 0 or rec_batch_size < 1:
            raise ValueError("cpu_threads must be >= 0 and rec_batch_size must be >= 1")

        core = ov.Core()
        common = {ov.properties.inference_num_threads: cpu_threads} if cpu_threads else {}
        # DET 固定 FP32 以保持检测框稳定；REC 由 OpenVINO 自动选择 CPU 精度。
        det_config = {**common, ov.properties.hint.inference_precision: ov.Type.f32}
        self.det = core.compile_model(required["det"], "CPU", det_config)
        self.rec = core.compile_model(required["rec"], "CPU", common)
        self.characters = self._load_characters(required["dict"])
        self.rec_batch_size = rec_batch_size
        self.score_threshold = score_threshold

    def predict(self, image: str | Path | np.ndarray) -> dict:
        """执行 DET、裁剪、REC 和解码；计时不含图片读取与模型加载。"""
        image = self._read_image(image)
        started = time.perf_counter()

        det_started = time.perf_counter()
        boxes = self._detect(image)
        det_seconds = time.perf_counter() - det_started

        crops = [self._crop(image, box.copy()) for box in boxes]
        rec_started = time.perf_counter()
        rec_results = self._recognize(crops)
        rec_seconds = time.perf_counter() - rec_started

        lines = [
            {
                "box": box.astype(int).tolist(),
                "text": text,
                "score": round(float(score), 6),
            }
            for box, (text, score) in zip(boxes, rec_results)
            if score >= self.score_threshold
        ]
        return {
            "lines": lines,
            "timings": {
                "det_seconds": round(det_seconds, 6),
                "rec_seconds": round(rec_seconds, 6),
                "total_seconds": round(time.perf_counter() - started, 6),
            },
        }

    def warmup(self, image: str | Path | np.ndarray) -> None:
        """执行一次完整 OCR，使正式请求不包含首次运行开销。"""
        self.predict(image)

    @staticmethod
    def _read_image(image: str | Path | np.ndarray) -> np.ndarray:
        """读取图片路径，或检查调用方传入的 OpenCV BGR 数组。"""
        if isinstance(image, np.ndarray):
            result = image
        else:
            path = Path(image)
            if not path.is_file():
                raise FileNotFoundError(path)
            # imdecode 可以正常读取 Windows 中文路径。
            result = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if result is None or result.ndim != 3 or result.shape[2] != 3:
            raise ValueError("image must be a valid three-channel BGR image")
        return result

    @staticmethod
    def _load_characters(path: Path) -> list[str]:
        """读取 REC 字符字典，并在索引 0 补入 CTC 空白符。"""
        characters = path.read_text(encoding="utf-8").splitlines()
        if characters and characters[0] == "":
            characters = characters[1:]
        if " " not in characters:
            characters.append(" ")
        return [""] + characters

    def _detect(self, image: np.ndarray) -> list[np.ndarray]:
        """检测文本框，恢复原图坐标并按从上到下、从左到右排序。"""
        resized, shape = self._resize_det(image)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        tensor = tensor.transpose(2, 0, 1)[None]
        prediction = self.det([tensor])[self.det.output(0)][0, 0]
        boxes = self._boxes_from_bitmap(prediction, prediction > 0.3, shape[1], shape[0])
        boxes = [self._order_box(box, image.shape) for box in boxes]
        boxes = [
            box
            for box in boxes
            if np.linalg.norm(box[0] - box[1]) > 3
            and np.linalg.norm(box[0] - box[3]) > 3
        ]
        boxes.sort(key=lambda box: (box[0][1], box[0][0]))
        for i in range(len(boxes) - 1):
            for j in range(i, -1, -1):
                same_line = abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 10
                if same_line and boxes[j + 1][0][0] < boxes[j][0][0]:
                    boxes[j], boxes[j + 1] = boxes[j + 1], boxes[j]
                else:
                    break
        return boxes

    @staticmethod
    def _resize_det(image: np.ndarray, limit: int = 960) -> tuple[np.ndarray, tuple[int, int]]:
        """保持宽高比，将 DET 输入长边限制为 960，宽高对齐到 32。"""
        height, width = image.shape[:2]
        ratio = min(1.0, limit / max(height, width))
        resize_h = max(int(round(int(height * ratio) / 32) * 32), 32)
        resize_w = max(int(round(int(width * ratio) / 32) * 32), 32)
        return cv2.resize(image, (resize_w, resize_h)), (height, width)

    @staticmethod
    def _mini_box(contour: np.ndarray) -> tuple[np.ndarray, float]:
        """计算轮廓的最小外接旋转矩形，并统一四个顶点顺序。"""
        rect = cv2.minAreaRect(contour)
        points = sorted(cv2.boxPoints(rect).tolist(), key=lambda point: point[0])
        left_top, left_bottom = (
            (points[0], points[1]) if points[1][1] > points[0][1] else (points[1], points[0])
        )
        right_top, right_bottom = (
            (points[2], points[3]) if points[3][1] > points[2][1] else (points[3], points[2])
        )
        return np.array([left_top, right_top, right_bottom, left_bottom]), min(rect[1])

    @staticmethod
    def _box_score(bitmap: np.ndarray, box: np.ndarray) -> float:
        """计算四边形内的平均文本概率。"""
        height, width = bitmap.shape
        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, width - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, width - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, height - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, height - 1))
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
        local = box.copy()
        local[:, 0] -= xmin
        local[:, 1] -= ymin
        cv2.fillPoly(mask, local.reshape(1, -1, 2).astype(np.int32), 1)
        return float(cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0])

    def _boxes_from_bitmap(
        self, prediction: np.ndarray, bitmap: np.ndarray, dest_width: int, dest_height: int
    ) -> list[np.ndarray]:
        """执行 DB 后处理，将概率图转换为原图坐标下的四边形文本框。"""
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        height, width = bitmap.shape
        for contour in contours[:1000]:
            box, short_side = self._mini_box(contour)
            if short_side < 3 or self._box_score(prediction, box) < 0.6:
                continue

            # 按面积与周长向外扩框，避免裁掉文字边缘。
            contour_box = box.astype(np.float32)
            perimeter = cv2.arcLength(contour_box, True)
            if perimeter == 0:
                continue
            distance = cv2.contourArea(contour_box) * 1.5 / perimeter
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(box.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = offset.Execute(distance)
            if len(expanded) != 1:
                continue

            box, short_side = self._mini_box(np.asarray(expanded).reshape(-1, 1, 2))
            if short_side < 5:
                continue
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int32))
        return boxes

    @staticmethod
    def _order_box(points: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
        """把顶点整理为左上、右上、右下、左下，并裁剪到图像范围。"""
        ordered = np.zeros((4, 2), np.float32)
        sums = points.sum(axis=1)
        ordered[0], ordered[2] = points[np.argmin(sums)], points[np.argmax(sums)]
        remaining = np.delete(points, (np.argmin(sums), np.argmax(sums)), axis=0)
        differences = np.diff(remaining, axis=1).ravel()
        ordered[1], ordered[3] = remaining[np.argmin(differences)], remaining[np.argmax(differences)]
        ordered[:, 0] = np.clip(ordered[:, 0], 0, image_shape[1] - 1)
        ordered[:, 1] = np.clip(ordered[:, 1], 0, image_shape[0] - 1)
        return ordered

    @staticmethod
    def _crop(image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """透视裁剪文本框；竖排文本旋转为 REC 使用的横向文本条。"""
        width = max(1, int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))))
        height = max(1, int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))))
        target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        matrix = cv2.getPerspectiveTransform(points.astype(np.float32), target)
        crop = cv2.warpPerspective(
            image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return np.rot90(crop) if crop.shape[0] / crop.shape[1] >= 1.5 else crop

    def _recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """按宽高比分批执行 REC，并将结果恢复到原检测框顺序。"""
        if not crops:
            return []
        ratios = np.array([crop.shape[1] / crop.shape[0] for crop in crops])
        order = np.argsort(ratios)
        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        for start in range(0, len(crops), self.rec_batch_size):
            indexes = order[start : start + self.rec_batch_size]
            max_ratio = max(320 / 48, max(ratios[index] for index in indexes))
            batch = np.stack([self._resize_rec(crops[index], max_ratio) for index in indexes])
            decoded = self._ctc_decode(self.rec([batch])[self.rec.output(0)])
            for index, result in zip(indexes, decoded):
                results[int(index)] = result
        return results

    @staticmethod
    def _resize_rec(image: np.ndarray, max_ratio: float) -> np.ndarray:
        """保持宽高比缩放到 48 像素高，并在右侧补零到批次宽度。"""
        target_h = 48
        target_w = int(target_h * max_ratio)
        ratio = image.shape[1] / image.shape[0]
        resized_w = min(target_w, int(math.ceil(target_h * ratio)))
        resized = cv2.resize(image, (resized_w, target_h)).astype(np.float32)
        resized = resized.transpose(2, 0, 1) / 127.5 - 1.0
        padded = np.zeros((3, target_h, target_w), np.float32)
        padded[:, :, :resized_w] = resized
        return padded

    def _ctc_decode(self, prediction: np.ndarray) -> list[tuple[str, float]]:
        """移除 CTC 空白符与连续重复字符，返回文本和平均置信度。"""
        indexes = prediction.argmax(axis=2)
        probabilities = prediction.max(axis=2)
        results = []
        for sequence, scores in zip(indexes, probabilities):
            keep = sequence != 0
            keep[1:] &= sequence[1:] != sequence[:-1]
            selected = sequence[keep]
            if len(selected) and int(selected.max()) >= len(self.characters):
                raise ValueError("recognition dictionary does not match model output")
            text = "".join(self.characters[int(index)] for index in selected)
            confidence = float(scores[keep].mean()) if keep.any() else 0.0
            results.append((text, confidence))
        return results
