import os

from PIL import Image

from deeplab import DeeplabV3
from inference_config import IMAGE_DIR, IMAGE_EXTENSIONS, SAVE_DIR


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    deeplab = DeeplabV3()

    for image_name in os.listdir(IMAGE_DIR):
        if not image_name.lower().endswith(IMAGE_EXTENSIONS):
            continue

        image_path = os.path.join(IMAGE_DIR, image_name)
        save_name = os.path.splitext(image_name)[0] + ".png"
        save_path = os.path.join(SAVE_DIR, save_name)

        image = Image.open(image_path)
        result = deeplab.detect_image(image)
        result.save(save_path)
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
