# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import numpy as np
import time
import yaml

import tools.infer.utility as utility
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppstructure.utility import parse_args
from picodet_postprocess import PicoDetPostProcess

logger = get_logger()


class LayoutPredictor(object):
    # PP-DocLayout 系列 23 类 -> ppstructure 恢复链路使用的兼容标签
    _DOC_LAYOUT_LABEL_MAP = {
        "title": "title",
        "plain text": "text",
        "plain_text": "text",
        "abandon": None,
        "figure": "figure",
        "figure_caption": "figure_caption",
        "table": "table",
        "table_caption": "table_caption",
        "table_footnote": "table_footnote",
        "isolate_formula": "equation",
        "formula_caption": "figure_caption",
        "doc_title": "title",
        "paragraph_title": "title",
        "text": "text",
        "number": "footer",
        "abstract": "text",
        "content": "text",
        "reference": "reference",
        "footnote": "reference",
        "header": "header",
        "footer": "footer",
        "algorithm": "text",
        "formula": "equation",
        "formula_number": "equation",
        "image": "figure",
        "figure_title": "figure_caption",
        "table_title": "table_caption",
        "seal": "figure",
        "chart_title": "figure_caption",
        "chart": "figure",
        "header_image": "figure",
        "footer_image": "figure",
        "aside_text": "text",
    }

    _NCNN_MAX_DET = 300
    _V10_REG_MAX = 16
    _V10_STRIDES = (8, 16, 32)

    def __init__(self, args):
        resize_size = [800, 608]
        inference_cfg = os.path.join(args.layout_model_dir or "", "inference.yml")
        layout_model_path = str(args.layout_model_dir or "")
        if os.path.isfile(inference_cfg):
            try:
                with open(inference_cfg, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                for op in cfg.get("Preprocess", []):
                    resize_cfg = None
                    if isinstance(op, dict):
                        if "Resize" in op and isinstance(op.get("Resize"), dict):
                            resize_cfg = op.get("Resize")
                        elif op.get("type") == "Resize":
                            resize_cfg = op
                    if not isinstance(resize_cfg, dict):
                        continue
                    target_size = resize_cfg.get("target_size")
                    if isinstance(target_size, (list, tuple)) and len(target_size) == 2:
                        resize_size = [int(target_size[0]), int(target_size[1])]
                        break
                    size = resize_cfg.get("size")
                    if isinstance(size, (list, tuple)) and len(size) == 2:
                        resize_size = [int(size[0]), int(size[1])]
                        break
            except Exception as e:
                logger.warning(
                    "Failed to parse %s, fallback resize to %s. err=%s",
                    inference_cfg,
                    resize_size,
                    e,
                )

        if not os.path.isfile(inference_cfg):
            inferred_imgsz = self._infer_ncnn_imgsz(args.layout_model_dir)
            if inferred_imgsz is not None:
                resize_size = [inferred_imgsz, inferred_imgsz]
        if layout_model_path.lower().endswith(".onnx"):
            inferred_onnx_size = self._infer_onnx_input_size(layout_model_path)
            if inferred_onnx_size is not None:
                resize_size = inferred_onnx_size

        pre_process_list = [
            {"Resize": {"size": resize_size}},
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image"]}},
        ]
        postprocess_params = {
            "name": "PicoDetPostProcess",
            "layout_dict_path": args.layout_dict_path,
            "score_threshold": args.layout_score_threshold,
            "nms_threshold": args.layout_nms_threshold,
        }

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        self.ncnn_param_path, self.ncnn_model_path = self._discover_ncnn_model(
            args.layout_model_dir
        )
        self.use_ncnn = self.ncnn_param_path is not None and self.ncnn_model_path is not None
        self.ncnn_input_size = resize_size
        self.ncnn_conf_threshold = min(float(args.layout_score_threshold), 0.25)
        self.enable_tiled_recall = self._env_flag("DOCFLOW_LAYOUT_TILE_RECALL", True)
        self.tile_overlap_ratio = self._env_float(
            "DOCFLOW_LAYOUT_TILE_OVERLAP", default=0.18, minimum=0.05, maximum=0.45
        )
        self.tile_trigger_ratio = self._env_float(
            "DOCFLOW_LAYOUT_TILE_TRIGGER_RATIO", default=1.05, minimum=1.0, maximum=3.0
        )
        self.tile_margin_ratio = self._env_float(
            "DOCFLOW_LAYOUT_TILE_MARGIN_RATIO", default=0.02, minimum=0.0, maximum=0.15
        )
        self.tile_max_passes = int(
            max(1, min(16, self._env_float("DOCFLOW_LAYOUT_TILE_MAX_PASSES", default=12, minimum=1, maximum=16)))
        )
        self.predictor = None
        self.input_tensor = None
        self.output_tensors = None
        self.config = None
        self.use_onnx = False
        self.input_names = None
        self.ncnn_net = None
        if self.use_ncnn:
            try:
                import ncnn
            except ImportError as exc:
                raise RuntimeError(
                    "Detected NCNN layout model assets, but Python package `ncnn` is not installed."
                ) from exc

            self.ncnn = ncnn
            self.ncnn_net = ncnn.Net()
            self.ncnn_net.load_param(self.ncnn_param_path)
            self.ncnn_net.load_model(self.ncnn_model_path)
        else:
            (
                self.predictor,
                self.input_tensor,
                self.output_tensors,
                self.config,
            ) = utility.create_predictor(args, "layout", logger)
            self.use_onnx = bool(args.use_onnx) or str(args.layout_model_dir or "").lower().endswith(".onnx")
            self.input_names = None if self.use_onnx else self.predictor.get_input_names()

    @staticmethod
    def _env_flag(name, default):
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "off", "no"}

    @staticmethod
    def _env_float(name, default, minimum, maximum):
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return max(minimum, min(maximum, value))

    def _map_label(self, label):
        return self._DOC_LAYOUT_LABEL_MAP.get(str(label).lower(), str(label).lower())

    @staticmethod
    def _infer_ncnn_imgsz(layout_model_dir):
        if not layout_model_dir:
            return None
        metadata_path = os.path.join(layout_model_dir, "metadata.yaml")
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = yaml.safe_load(f) or {}
                imgsz = metadata.get("imgsz")
                if isinstance(imgsz, (list, tuple)) and len(imgsz) >= 2:
                    return int(imgsz[0])
            except Exception:
                pass
        match = re.search(r"imgsz(\d+)", os.path.basename(layout_model_dir))
        return int(match.group(1)) if match else None

    @staticmethod
    def _discover_ncnn_model(layout_model_dir):
        if not layout_model_dir or not os.path.isdir(layout_model_dir):
            return None, None
        stems = ["model.ncnn", "model", "inference"]
        for stem in stems:
            param_path = os.path.join(layout_model_dir, f"{stem}.param")
            model_path = os.path.join(layout_model_dir, f"{stem}.bin")
            if os.path.isfile(param_path) and os.path.isfile(model_path):
                return param_path, model_path
        return None, None

    @staticmethod
    def _infer_onnx_input_size(layout_model_path):
        if not layout_model_path or not os.path.isfile(layout_model_path):
            return None
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(
                layout_model_path,
                providers=["CPUExecutionProvider"],
            )
            shape = session.get_inputs()[0].shape
            if len(shape) < 4:
                return None
            height, width = shape[2], shape[3]
            if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
                return [int(width), int(height)]
        except Exception:
            return None
        return None

    def _letterbox_image(self, img):
        new_shape = (self.ncnn_input_size[1], self.ncnn_input_size[0])
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, (r, r), (left, top)

    def _decode_v10_raw_head(self, preds):
        nc = len(self.postprocess_op.labels)
        box_logits = preds[:, : self._V10_REG_MAX * 4].astype(np.float32)
        cls_logits = preds[:, self._V10_REG_MAX * 4 :].astype(np.float32)
        if cls_logits.shape[1] != nc:
            return None, None
        total_positions = preds.shape[0]
        input_w, input_h = self.ncnn_input_size[0], self.ncnn_input_size[1]
        expected_counts, anchor_points, stride_values = [], [], []
        for stride in self._V10_STRIDES:
            feat_h = input_h // stride
            feat_w = input_w // stride
            count = feat_h * feat_w
            expected_counts.append(count)
            sy = np.arange(feat_h, dtype=np.float32) + 0.5
            sx = np.arange(feat_w, dtype=np.float32) + 0.5
            grid_x, grid_y = np.meshgrid(sx, sy)
            anchor_points.append(np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1))
            stride_values.append(np.full((count, 1), stride, dtype=np.float32))
        if sum(expected_counts) != total_positions:
            return None, None
        anchor_points = np.concatenate(anchor_points, axis=0)
        stride_values = np.concatenate(stride_values, axis=0)
        box_logits = box_logits.reshape(-1, 4, self._V10_REG_MAX)
        box_logits = box_logits - np.max(box_logits, axis=2, keepdims=True)
        np.exp(box_logits, out=box_logits)
        box_logits /= np.sum(box_logits, axis=2, keepdims=True)
        reg_range = np.arange(self._V10_REG_MAX, dtype=np.float32)
        distances = np.sum(box_logits * reg_range[None, None, :], axis=2)
        x1y1 = anchor_points - distances[:, :2]
        x2y2 = anchor_points + distances[:, 2:]
        centers = (x1y1 + x2y2) / 2.0
        wh = x2y2 - x1y1
        boxes = np.concatenate([centers, wh], axis=1) * stride_values
        scores = 1.0 / (1.0 + np.exp(-cls_logits))
        return boxes, scores

    def _postprocess_predecoded_cxcywh_outputs(self, raw_preds, ori_shape, ratio_pad):
        preds = np.asarray(raw_preds)
        if preds.ndim != 2:
            return None
        if preds.shape[0] in (
            4 + len(self.postprocess_op.labels),
            self._V10_REG_MAX * 4 + len(self.postprocess_op.labels),
        ) and preds.shape[1] not in (
            4 + len(self.postprocess_op.labels),
            self._V10_REG_MAX * 4 + len(self.postprocess_op.labels),
        ):
            preds = preds.T
        box_dims = preds.shape[1] - len(self.postprocess_op.labels)
        if box_dims == 4:
            boxes = preds[:, :4]
            scores = preds[:, 4:]
        elif box_dims == self._V10_REG_MAX * 4:
            boxes, scores = self._decode_v10_raw_head(preds)
            if boxes is None:
                return None
        else:
            return None
        max_det = min(self._NCNN_MAX_DET, preds.shape[0], preds.shape[0] * scores.shape[1])
        max_scores = scores.max(axis=1)
        topk_candidate_idx = np.argsort(max_scores)[-max_det:][::-1]
        boxes = boxes[topk_candidate_idx]
        scores = scores[topk_candidate_idx]
        flat_scores = scores.reshape(-1)
        topk_score_idx = np.argsort(flat_scores)[-max_det:][::-1]
        labels = topk_score_idx % scores.shape[1]
        box_idx = topk_score_idx // scores.shape[1]
        boxes = boxes[box_idx]
        scores = flat_scores[topk_score_idx]
        boxes = boxes.astype(np.float32).copy()
        boxes[:, 0] -= boxes[:, 2] / 2.0
        boxes[:, 1] -= boxes[:, 3] / 2.0
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes[:, :4] /= gain
        ori_h, ori_w = ori_shape[:2]
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, ori_w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, ori_h)
        results = []
        for box, score, label_idx in zip(boxes, scores, labels):
            if float(score) <= self.ncnn_conf_threshold:
                continue
            raw_label = self.postprocess_op.labels[int(label_idx)]
            mapped_label = self._map_label(raw_label)
            if mapped_label is None:
                continue
            results.append({"bbox": box, "label": mapped_label, "score": float(score)})
        return results

    def _predict_with_ncnn(self, img):
        letterboxed, ratio, pad = self._letterbox_image(img)
        chw = np.ascontiguousarray(letterboxed[..., ::-1].transpose(2, 0, 1), dtype=np.float32)
        chw /= 255.0
        mat = self.ncnn.Mat(chw)
        extractor = self.ncnn_net.create_extractor()
        input_name = self.ncnn_net.input_names()[0]
        output_name = self.ncnn_net.output_names()[0]
        extractor.input(input_name, mat)
        ret, out = extractor.extract(output_name)
        if ret != 0:
            raise RuntimeError(f"NCNN layout inference failed with code {ret}")
        return np.array(out), (ratio, pad)

    def _parse_predecoded_outputs(self, outputs, ori_shape):
        """兼容 PP-DocLayout-S 等直接输出 NMS 后 boxes 的模型。"""
        if not isinstance(outputs, list) or len(outputs) < 2:
            return None

        boxes_arr = None
        nums_arr = None
        for out in outputs:
            if not isinstance(out, np.ndarray):
                continue
            if out.ndim == 2 and out.shape[1] >= 6:
                boxes_arr = out
            elif out.ndim == 1 and out.size > 0:
                nums_arr = out

        if boxes_arr is None or nums_arr is None:
            return None

        h, w = ori_shape[:2]
        keep_n = int(nums_arr[0])
        keep_n = max(0, min(keep_n, boxes_arr.shape[0]))
        labels = self.postprocess_op.labels
        results = []
        for dt in boxes_arr[:keep_n]:
            clsid = int(dt[0])
            score = float(dt[1])
            if score < self.postprocess_op.score_threshold:
                continue
            bbox = dt[2:6].astype(np.float32)
            bbox[0::2] = np.clip(bbox[0::2], 0, w)
            bbox[1::2] = np.clip(bbox[1::2], 0, h)
            raw_label = labels[clsid] if 0 <= clsid < len(labels) else str(clsid)
            results.append(
                {"bbox": bbox, "label": self._map_label(raw_label), "score": score}
            )
        return results

    @staticmethod
    def _tile_starts(length, tile_length, overlap_ratio):
        if tile_length >= length:
            return [0]
        stride = max(1, int(round(tile_length * (1.0 - overlap_ratio))))
        starts = [0]
        while starts[-1] + tile_length < length:
            next_start = min(starts[-1] + stride, length - tile_length)
            if next_start <= starts[-1]:
                break
            starts.append(next_start)
        return starts

    @staticmethod
    def _clip_bbox(bbox, width, height):
        clipped = np.asarray(bbox, dtype=np.float32).copy()
        clipped[0::2] = np.clip(clipped[0::2], 0, width)
        clipped[1::2] = np.clip(clipped[1::2], 0, height)
        return clipped

    @staticmethod
    def _bbox_area(bbox):
        return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))

    @classmethod
    def _bbox_iou(cls, bbox_a, bbox_b):
        x1 = max(float(bbox_a[0]), float(bbox_b[0]))
        y1 = max(float(bbox_a[1]), float(bbox_b[1]))
        x2 = min(float(bbox_a[2]), float(bbox_b[2]))
        y2 = min(float(bbox_a[3]), float(bbox_b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0.0:
            return 0.0
        area_a = cls._bbox_area(bbox_a)
        area_b = cls._bbox_area(bbox_b)
        denom = area_a + area_b - inter
        if denom <= 1e-6:
            return 0.0
        return inter / denom

    @classmethod
    def _bbox_containment(cls, bbox_a, bbox_b):
        x1 = max(float(bbox_a[0]), float(bbox_b[0]))
        y1 = max(float(bbox_a[1]), float(bbox_b[1]))
        x2 = min(float(bbox_a[2]), float(bbox_b[2]))
        y2 = min(float(bbox_a[3]), float(bbox_b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0.0:
            return 0.0
        area_a = cls._bbox_area(bbox_a)
        area_b = cls._bbox_area(bbox_b)
        return inter / max(1e-6, min(area_a, area_b))

    @classmethod
    def _same_detection(cls, candidate, kept):
        if candidate["label"] != kept["label"]:
            return False
        bbox_a = candidate["bbox"]
        bbox_b = kept["bbox"]
        return cls._bbox_iou(bbox_a, bbox_b) >= 0.45 or cls._bbox_containment(bbox_a, bbox_b) >= 0.88

    def _filter_tile_edge_box(
        self,
        bbox,
        *,
        tile_width,
        tile_height,
        is_left_edge,
        is_top_edge,
        is_right_edge,
        is_bottom_edge,
    ):
        margin_x = max(8.0, float(tile_width) * self.tile_margin_ratio)
        margin_y = max(8.0, float(tile_height) * self.tile_margin_ratio)
        if not is_left_edge and float(bbox[0]) <= margin_x:
            return True
        if not is_top_edge and float(bbox[1]) <= margin_y:
            return True
        if not is_right_edge and float(tile_width - bbox[2]) <= margin_x:
            return True
        if not is_bottom_edge and float(tile_height - bbox[3]) <= margin_y:
            return True
        return False

    def _merge_layout_results(self, primary_results, fallback_results, image_shape):
        if not fallback_results:
            return primary_results
        image_height, image_width = image_shape[:2]
        merged = []
        combined = []
        for region in list(primary_results) + list(fallback_results):
            label = region.get("label")
            if label is None:
                continue
            bbox = self._clip_bbox(region["bbox"], image_width, image_height)
            if self._bbox_area(bbox) <= 1.0:
                continue
            combined.append(
                {
                    "bbox": bbox,
                    "label": label,
                    "score": float(region.get("score", 0.0)),
                }
            )

        combined.sort(
            key=lambda item: (
                float(item["score"]),
                self._bbox_area(item["bbox"]),
            ),
            reverse=True,
        )
        for candidate in combined:
            if any(self._same_detection(candidate, kept) for kept in merged):
                continue
            merged.append(candidate)
        return merged

    def _should_run_tiled_recall(self, img):
        if not self.enable_tiled_recall:
            return False
        if self.tile_max_passes <= 1:
            return False
        img_h, img_w = img.shape[:2]
        input_w = max(int(self.ncnn_input_size[0]), 1)
        input_h = max(int(self.ncnn_input_size[1]), 1)
        return (
            img_w > input_w * self.tile_trigger_ratio
            or img_h > input_h * self.tile_trigger_ratio
        )

    def _predict_tiled_recall(self, img):
        img_h, img_w = img.shape[:2]
        tile_w = min(max(int(self.ncnn_input_size[0]), 32), img_w)
        tile_h = min(max(int(self.ncnn_input_size[1]), 32), img_h)
        x_starts = self._tile_starts(img_w, tile_w, self.tile_overlap_ratio)
        y_starts = self._tile_starts(img_h, tile_h, self.tile_overlap_ratio)
        if len(x_starts) * len(y_starts) <= 1:
            return [], 0.0

        fallback_results = []
        elapsed = 0.0
        pass_count = 0
        for y0 in y_starts:
            for x0 in x_starts:
                if pass_count >= self.tile_max_passes:
                    return fallback_results, elapsed
                crop = img[y0 : y0 + tile_h, x0 : x0 + tile_w]
                local_results, local_elapsed = self._predict_single(crop)
                elapsed += local_elapsed
                pass_count += 1
                is_left_edge = x0 == 0
                is_top_edge = y0 == 0
                is_right_edge = x0 + tile_w >= img_w
                is_bottom_edge = y0 + tile_h >= img_h
                for region in local_results:
                    label = region.get("label")
                    if label is None:
                        continue
                    bbox = np.asarray(region["bbox"], dtype=np.float32).copy()
                    if self._filter_tile_edge_box(
                        bbox,
                        tile_width=tile_w,
                        tile_height=tile_h,
                        is_left_edge=is_left_edge,
                        is_top_edge=is_top_edge,
                        is_right_edge=is_right_edge,
                        is_bottom_edge=is_bottom_edge,
                    ):
                        continue
                    bbox[0] += x0
                    bbox[2] += x0
                    bbox[1] += y0
                    bbox[3] += y0
                    fallback_results.append(
                        {
                            "bbox": bbox,
                            "label": label,
                            "score": float(region.get("score", 0.0)),
                        }
                    )
        return fallback_results, elapsed

    def _predict_single(self, img):
        if self.use_ncnn:
            starttime = time.time()
            outputs, ratio_pad = self._predict_with_ncnn(img)
            results = self._postprocess_predecoded_cxcywh_outputs(outputs, img.shape, ratio_pad)
            elapse = time.time() - starttime
            if results is None:
                raise RuntimeError("Unsupported NCNN layout output shape")
            return results, elapse

        ori_im = img.copy()
        data = {"image": img}
        data = transform(data, self.preprocess_op)
        img = data[0]

        if img is None:
            return None, 0

        img = np.expand_dims(img, axis=0)
        img = img.copy()

        preds, elapse = 0, 1
        starttime = time.time()
        ori_h, ori_w = ori_im.shape[:2]
        in_h, in_w = img.shape[2], img.shape[3]
        scale_factor = np.array(
            [[in_h / float(ori_h), in_w / float(ori_w)]], dtype=np.float32
        )

        np_score_list, np_boxes_list = [], []
        if self.use_onnx:
            input_dict = {}
            input_names = (
                self.input_tensor if isinstance(self.input_tensor, list) else [self.input_tensor.name]
            )
            for name in input_names:
                if name == "image":
                    input_dict[name] = img
                elif name == "scale_factor":
                    input_dict[name] = scale_factor
                else:
                    input_dict[name] = img
            outputs = self.predictor.run(self.output_tensors, input_dict)
            predecoded = self._parse_predecoded_outputs(outputs, ori_im.shape)
            if predecoded is not None:
                elapse = time.time() - starttime
                return predecoded, elapse
            num_outs = int(len(outputs) / 2)
            for out_idx in range(num_outs):
                np_score_list.append(outputs[out_idx])
                np_boxes_list.append(outputs[out_idx + num_outs])
        else:
            if self.input_names and len(self.input_names) > 1:
                for name in self.input_names:
                    handle = self.predictor.get_input_handle(name)
                    if name == "image":
                        handle.copy_from_cpu(img)
                    elif name == "scale_factor":
                        handle.copy_from_cpu(scale_factor)
                    else:
                        # 未知输入名兜底喂 image，避免输入未赋值导致推理失败
                        handle.copy_from_cpu(img)
            else:
                self.input_tensor.copy_from_cpu(img)
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            outputs = [
                self.predictor.get_output_handle(name).copy_to_cpu()
                for name in output_names
            ]
            predecoded = self._parse_predecoded_outputs(outputs, ori_im.shape)
            if predecoded is not None:
                elapse = time.time() - starttime
                return predecoded, elapse
            num_outs = int(len(output_names) / 2)
            for out_idx in range(num_outs):
                np_score_list.append(
                    self.predictor.get_output_handle(
                        output_names[out_idx]
                    ).copy_to_cpu()
                )
                np_boxes_list.append(
                    self.predictor.get_output_handle(
                        output_names[out_idx + num_outs]
                    ).copy_to_cpu()
                )
        preds = dict(boxes=np_score_list, boxes_num=np_boxes_list)

        post_preds = self.postprocess_op(ori_im, img, preds)
        for it in post_preds:
            it["label"] = self._map_label(it["label"])
        elapse = time.time() - starttime
        return post_preds, elapse

    def __call__(self, img):
        base_results, base_elapsed = self._predict_single(img)
        if not self._should_run_tiled_recall(img):
            return base_results, base_elapsed
        fallback_results, fallback_elapsed = self._predict_tiled_recall(img)
        if not fallback_results:
            return base_results, base_elapsed + fallback_elapsed
        merged = self._merge_layout_results(base_results, fallback_results, img.shape)
        return merged, base_elapsed + fallback_elapsed


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    layout_predictor = LayoutPredictor(args)
    count = 0
    total_time = 0

    repeats = 50
    for image_file in image_file_list:
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue

        layout_res, elapse = layout_predictor(img)

        logger.info("result: {}".format(layout_res))

        if count > 0:
            total_time += elapse
        count += 1
        logger.info("Predict time of {}: {}".format(image_file, elapse))


if __name__ == "__main__":
    main(parse_args())
