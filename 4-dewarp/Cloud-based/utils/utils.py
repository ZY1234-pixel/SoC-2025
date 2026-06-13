import numpy as np
from PIL import Image, ImageOps


def cvtColor(image):
    if isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image)
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    return image.convert("RGB")


def resize_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image, nw, nh


def preprocess_input(image):
    return image / 255.0
