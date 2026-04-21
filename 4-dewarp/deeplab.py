import copy
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
from utils.utils import cvtColor, preprocess_input, resize_image

def refine_mask(pr):
    """
    pr: (H, W) numpy, 类别ID
    return: 优化后的mask (0/1)
    """
    # 取book类别
    mask = (pr == 1).astype(np.uint8)*255
    # 最大连通域（去掉杂点）
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)
    # 填洞（书页内部空洞填满）
    h, w = mask.shape
    flood = mask.copy()
    tmp = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, tmp, (0, 0), 1)
    flood_inv = cv2.bitwise_not(flood*255)//255
    mask = cv2.bitwise_or(mask, flood_inv)

    # 形态学平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 填缝
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去毛刺

    return mask.astype(np.uint8)

class DeeplabV3_ONNX(object):

    _defaults = {
        "model_path": "./model/deeplabv3p.onnx",
        "num_classes": 2,
        "input_shape": [640, 640],
        "mix_type": 0,
        "cuda": True,
    }

    def __init__(self, **kwargs):

        self.__dict__.update(self._defaults)
        for k, v in kwargs.items():
            setattr(self, k, v)

        # colors
        self.colors = [(0, 0, 0), (128, 0, 0)]

        # ONNX runtime
        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if self.cuda else
            ['CPUExecutionProvider']
        )

        self.session = ort.InferenceSession(self.model_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("ONNX loaded:", self.model_path)

    # -------------------------
    # 推理
    # -------------------------
    def detect_image(self, image, count=False, name_classes=None):

        old_img = copy.deepcopy(image)
        image = cvtColor(image)
        image = Image.fromarray(np.uint8(image))

        orininal_w, orininal_h = image.size

        image_data, nw, nh = resize_image(
            image,
            (self.input_shape[1], self.input_shape[0])
        )

        image_data = np.array(image_data, dtype=np.float32)
        image_data = preprocess_input(image_data)
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, 0)

        # ---------------- inference ----------------
        out = self.session.run(
            [self.output_name],
            {self.input_name: image_data}
        )[0]


        # ---------------- format ----------------
        if out.ndim == 4:
            out = out[0]

        if out.shape[0] == self.num_classes:
            pr = out.transpose(1, 2, 0)
        else:
            pr = out

        # ---------------- softmax ----------------
        pr = pr - np.max(pr, axis=-1, keepdims=True)
        pr = np.exp(pr)
        pr = pr / np.sum(pr, axis=-1, keepdims=True)

        # ---------------- crop ----------------
        top = (self.input_shape[0] - nh) // 2
        left = (self.input_shape[1] - nw) // 2
        pr = pr[top:top + nh, left:left + nw]

        # ---------------- resize ----------------
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

        # ---------------- argmax ----------------
        pr = np.argmax(pr, axis=-1).astype(np.uint8)

        pr = refine_mask(pr)

        # ---------------- mix_type ----------------
        if self.mix_type == 0:
            seg = np.array(self.colors, np.uint8)[pr]
            seg = Image.fromarray(seg)
            image = Image.blend(old_img, seg, 0.7)

        elif self.mix_type == 1:
            seg = np.array(self.colors, np.uint8)[pr]
            image = Image.fromarray(seg)

        elif self.mix_type == 2:
            seg = (pr != 0)[..., None] * np.array(old_img, np.float32)
            image = Image.fromarray(np.uint8(seg))

        return image