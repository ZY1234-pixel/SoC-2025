import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input/output paths for batch prediction.
IMAGE_DIR = os.path.join(BASE_DIR, "img")
SAVE_DIR = os.path.join(BASE_DIR, "img_out")
IMAGE_EXTENSIONS = (".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")

# Model settings.
MODEL_PATH = os.path.join(BASE_DIR, "best_epoch_weights.pth")
NUM_CLASSES = 2
INPUT_SHAPE = (1024, 1024)
DOWNSAMPLE_FACTOR = 8
BOOK_THRESHOLD = 0.60

# Output settings.
# OUTPUT_TYPE:
#   "mask": save document segmentation result
#   "edge": save document contour edge
OUTPUT_TYPE = "mask"
EDGE_WIDTH = 2

# MIX_TYPE is used only when OUTPUT_TYPE == "mask":
#   0: blend mask with original image
#   1: save 0-255 black/white mask
MIX_TYPE = 0
BLEND_ALPHA = 0.7
