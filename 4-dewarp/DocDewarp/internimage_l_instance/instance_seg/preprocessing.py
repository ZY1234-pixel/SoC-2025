"""First-pass PIL letterbox and ROI OpenCV letterbox preprocessing."""

import copy
from typing import Any, Dict, List

import cv2

import numpy as np
import torch
from PIL import Image
from detectron2.data import detection_utils as utils
from detectron2.data.transforms import (
    PadTransform,
    ResizeTransform,
    TransformList,
)


class DocLetterboxDatasetMapper:
    """Preserve all documents while matching the audited semantic preprocessing."""

    def __init__(
        self,
        is_train: bool,
        image_size: int = 1024,
        image_format: str = "RGB",
        pad_value: int = 128,
        random_flip: bool = True,
    ) -> None:
        if is_train:
            raise ValueError("Deployment mapper supports inference only")
        self.is_train = False
        self.image_size = int(image_size)
        self.image_format = image_format
        self.pad_value = int(pad_value)
        self.random_flip = bool(random_flip and is_train)

    @classmethod
    def from_config(cls, cfg, is_train: bool = False) -> Dict[str, Any]:
        return {
            "is_train": is_train,
            "image_size": cfg.INPUT.IMAGE_SIZE,
            "image_format": cfg.INPUT.FORMAT,
            "pad_value": 128,
            "random_flip": cfg.INPUT.RANDOM_FLIP == "horizontal",
        }

    def _transforms(self, height: int, width: int) -> TransformList:
        transforms: List[Any] = []
        scale = min(self.image_size / width, self.image_size / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        transforms.append(
            ResizeTransform(
                height, width, new_height, new_width, interp=Image.Resampling.LANCZOS
            )
        )
        left = (self.image_size - new_width) // 2
        top = (self.image_size - new_height) // 2
        transforms.append(
            PadTransform(
                left,
                top,
                self.image_size - new_width - left,
                self.image_size - new_height - top,
                orig_w=new_width,
                orig_h=new_height,
                pad_value=self.pad_value,
                seg_pad_value=0,
            )
        )
        return TransformList(transforms)

    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        # In standalone inference the caller also needs the original image for
        # ROI crops.  Accept a preloaded image so the same camera file is not
        # decoded twice.  Training/evaluation records do not provide this key
        # and keep the original read path unchanged.
        image = dataset_dict.get("_preloaded_image")
        if image is None:
            record = copy.deepcopy(dataset_dict)
            image = utils.read_image(record["file_name"], format=self.image_format)
        else:
            record = copy.deepcopy(
                {
                    key: value
                    for key, value in dataset_dict.items()
                    if key != "_preloaded_image"
                }
            )
            image = np.asarray(image)
        utils.check_image_size(record, image)
        original_height, original_width = image.shape[:2]
        transforms = self._transforms(original_height, original_width)
        transformed = transforms.apply_image(image)
        if transformed.shape[:2] != (self.image_size, self.image_size):
            raise RuntimeError(
                f"Letterbox produced unexpected shape {transformed.shape}"
            )

        valid_pixels = np.ones((original_height, original_width), dtype=np.uint8)
        padding_mask = ~transforms.apply_segmentation(valid_pixels).astype(bool)
        record["image"] = torch.as_tensor(
            np.ascontiguousarray(transformed.transpose(2, 0, 1))
        )
        record["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        scale = min(self.image_size / original_width, self.image_size / original_height)
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        record["letterbox_meta"] = {
            "scale": scale,
            "left": (self.image_size - resized_width) // 2,
            "top": (self.image_size - resized_height) // 2,
            "resized_width": resized_width,
            "resized_height": resized_height,
            "original_width": original_width,
            "original_height": original_height,
        }

        record.pop("annotations", None)
        return record


def inference_record_from_bgr(
    image_bgr: np.ndarray,
    image_size: int,
    image_id: int,
) -> Dict[str, Any]:
    """Create the same RGB/[0,1] letterbox contract for an in-memory ROI."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 BGR image, received {image_bgr.shape}")
    height, width = image_bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Cannot letterbox an empty image of shape {image_bgr.shape}")

    scale = min(image_size / width, image_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4
    )
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas = np.full((image_size, image_size, 3), 128, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    padding_mask = np.ones((image_size, image_size), dtype=bool)
    padding_mask[top : top + resized_height, left : left + resized_width] = False
    return {
        "image": torch.as_tensor(np.ascontiguousarray(canvas.transpose(2, 0, 1))),
        "padding_mask": torch.as_tensor(np.ascontiguousarray(padding_mask)),
        "height": height,
        "width": width,
        "image_id": int(image_id),
        "letterbox_meta": {
            "scale": scale,
            "left": left,
            "top": top,
            "resized_width": resized_width,
            "resized_height": resized_height,
            "original_width": width,
            "original_height": height,
        },
    }
