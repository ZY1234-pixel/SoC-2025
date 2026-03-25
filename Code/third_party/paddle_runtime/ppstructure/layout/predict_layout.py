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
        "table": "table",
        "table_title": "table_caption",
        "seal": "figure",
        "chart_title": "figure_caption",
        "chart": "figure",
        "header_image": "figure",
        "footer_image": "figure",
        "aside_text": "text",
    }

    def __init__(self, args):
        resize_size = [800, 608]
        inference_cfg = os.path.join(args.layout_model_dir or "", "inference.yml")
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
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            self.config,
        ) = utility.create_predictor(args, "layout", logger)
        self.use_onnx = args.use_onnx
        self.input_names = None if self.use_onnx else self.predictor.get_input_names()

    def _map_label(self, label):
        return self._DOC_LAYOUT_LABEL_MAP.get(str(label).lower(), str(label).lower())

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

    def __call__(self, img):
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
