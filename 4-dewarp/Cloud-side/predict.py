import os
from pathlib import Path

from PIL import Image

from deeplab import DeeplabV3
from inference_config import IMAGE_DIR, IMAGE_EXTENSIONS, SAVE_DIR


def main():
    input_dir = Path(IMAGE_DIR)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    deeplab = DeeplabV3()

    for image_path in sorted(input_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        save_name = image_path.relative_to(input_dir).with_suffix(".png")
        save_path = Path(SAVE_DIR) / save_name
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as image:
            result = deeplab.detect_image(image)
            result.save(save_path)
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
