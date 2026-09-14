import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input/output paths for batch prediction.
IMAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "TEST_dewarp"))
SAVE_DIR = os.path.join(BASE_DIR, "img_out")
IMAGE_EXTENSIONS = (".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")

# Model settings.
# Select the model architecture.  The checkpoint must be trained with the
# same backbone: best_hd95_epoch_85.pth is MobileNetV3, while
# best_epoch_weights.pth is Xception.
BACKBONE = "mobilenetv3"  # "mobilenetv3" or "xception"
MODEL_PATH = os.path.join(BASE_DIR, "best_hd95_epoch_85.pth")
NUM_CLASSES = 2
INPUT_SHAPE = (1024, 1024)
# Must match training: best_hd95_epoch_85.pth uses output stride 8.
# Stride/dilation are architecture settings and are not stored in the weights.
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
