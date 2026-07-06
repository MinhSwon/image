import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
POSITIVE_DIR = os.path.join(DATASET_DIR, "positive")
NEGATIVE_DIR = os.path.join(DATASET_DIR, "negative")
VIDEO_DIR = os.path.join(DATASET_DIR, "videos")
TEST_IMAGE_DIR = os.path.join(DATASET_DIR, "test_images")

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

CUSTOM_MODEL_PATH = os.path.join(MODEL_DIR, "custom_hog_svm.pkl")

# Kích thước chuẩn HOG cho người đi bộ trong OpenCV
WINDOW_SIZE = (64, 128)  # width, height

# Tham số HOG dùng khi tự train
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

# Tham số realtime detector
DEFAULT_HIT_THRESHOLD = 0.0
DEFAULT_WIN_STRIDE = (8, 8)
DEFAULT_PADDING = (16, 16)
DEFAULT_SCALE = 1.05
DEFAULT_GROUP_THRESHOLD = 2

# Resize frame để chạy mượt hơn
DEFAULT_MAX_WIDTH = 900
