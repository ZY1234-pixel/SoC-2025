import os

from PIL import Image

from deeplab import DeeplabV3


IMAGE_EXTENSIONS = (".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Change this value to choose output:
# "mask": save segmentation mask
# "edge": save contour edge
OUTPUT_TYPE = "mask"
EDGE_WIDTH = 2
# 0: blend mask with original image, only works when OUTPUT_TYPE is "mask"
# 1: save 0-255 mask/edge image
MIX_TYPE = 0


def main():
    image_dir = os.path.join(BASE_DIR, "img")
    save_dir = os.path.join(BASE_DIR, "img_out")

    os.makedirs(save_dir, exist_ok=True)
    deeplab = DeeplabV3(mix_type=MIX_TYPE)

    for image_name in os.listdir(image_dir):
        if not image_name.lower().endswith(IMAGE_EXTENSIONS):
            continue

        image_path = os.path.join(image_dir, image_name)
        save_name = os.path.splitext(image_name)[0] + ".png"
        save_path = os.path.join(save_dir, save_name)

        image = Image.open(image_path)
        result = deeplab.detect_image(
            image,
            output_type=OUTPUT_TYPE,
            edge_width=EDGE_WIDTH,
        )
        result.save(save_path)
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
