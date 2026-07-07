import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from inference_config import (
    BLEND_ALPHA,
    BOOK_THRESHOLD,
    DOWNSAMPLE_FACTOR,
    EDGE_WIDTH,
    INPUT_SHAPE,
    MIX_TYPE,
    MODEL_PATH,
    NUM_CLASSES,
    OUTPUT_TYPE,
)
from nets.deeplabv3_plus import DeepLab
from utils.utils import cvtColor, preprocess_input, resize_image


def mask_to_edge(mask, edge_width=2):
    edge_width = max(1, int(edge_width))
    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge = np.zeros_like(mask)
    cv2.drawContours(edge, contours, -1, 255, thickness=edge_width)
    return edge.astype(np.uint8)


class DeeplabV3:
    def __init__(
        self,
        model_path=MODEL_PATH,
        input_shape=INPUT_SHAPE,
        downsample_factor=DOWNSAMPLE_FACTOR,
        book_threshold=BOOK_THRESHOLD,
        blend_alpha=BLEND_ALPHA,
        mix_type=MIX_TYPE,
        num_classes=NUM_CLASSES,
    ):
        self.model_path = model_path
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.downsample_factor = downsample_factor
        self.book_threshold = book_threshold
        self.blend_alpha = blend_alpha
        self.mix_type = mix_type
        self.colors = np.array([(0, 0, 0), (128, 0, 0)], dtype=np.uint8)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model weights not found: {self.model_path}")

        net = DeepLab(num_classes=self.num_classes, downsample_factor=self.downsample_factor)
        state_dict = torch.load(self.model_path, map_location=self.device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if all(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key[7:]: value for key, value in state_dict.items()}

        net.load_state_dict(state_dict)
        net.eval()
        net.to(self.device)
        return net

    def predict_mask(self, image):
        return self._predict_mask(cvtColor(image))

    def _predict_mask(self, image):
        original_h, original_w = np.array(image).shape[:2]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.array(image_data, dtype=np.float32)
        image_data = preprocess_input(image_data)
        image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).to(self.device)
            output = self.net(images)[0]
            prediction = F.softmax(output.permute(1, 2, 0), dim=-1).cpu().numpy()

        y1 = int((self.input_shape[0] - nh) // 2)
        x1 = int((self.input_shape[1] - nw) // 2)
        prediction = prediction[y1:y1 + nh, x1:x1 + nw]
        prediction = cv2.resize(prediction, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        mask = (prediction[..., 1] >= self.book_threshold).astype(np.uint8)
        return mask

    def detect_image(self, image, output_type=None, edge_width=None):
        output_type = OUTPUT_TYPE if output_type is None else output_type
        edge_width = EDGE_WIDTH if edge_width is None else edge_width

        image = cvtColor(image)
        mask = self._predict_mask(image)
        if output_type == "edge":
            return Image.fromarray(mask_to_edge(mask, edge_width=edge_width))

        if output_type != "mask":
            raise ValueError("output_type only supports 'mask' or 'edge'")

        if self.mix_type == 1:
            return Image.fromarray((mask * 255).astype(np.uint8))

        if self.mix_type != 0:
            raise ValueError("mix_type only supports 0(blend image) or 1(0-255 mask)")

        overlay = self.colors[mask.reshape(-1)].reshape(mask.shape[0], mask.shape[1], 3)
        overlay = Image.fromarray(overlay)
        return Image.blend(image, overlay, self.blend_alpha)
