import os

from PIL import Image

from deeplab import DeeplabV3


IMAGE_EXTENSIONS = (".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    image_dir = os.path.join(BASE_DIR, "img")
    save_dir = os.path.join(BASE_DIR, "img_out")

    os.makedirs(save_dir, exist_ok=True)
    deeplab = DeeplabV3()

    for image_name in os.listdir(image_dir):
        if not image_name.lower().endswith(IMAGE_EXTENSIONS):
            continue

        image_path = os.path.join(image_dir, image_name)
        save_path = os.path.join(save_dir, image_name)

        image = Image.open(image_path)
        result = deeplab.detect_image(image)
        result.save(save_path)
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
