import cv2
import numpy as np
from skimage.feature import hog

from config import (
    WINDOW_SIZE,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    HOG_BLOCK_NORM,
)


def preprocess_for_hog(image):
    """
    Resize về 64x128, chuyển grayscale.
    """
    image = cv2.resize(image, WINDOW_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray


def extract_hog_feature(image):
    """
    Trích xuất vector HOG.
    """
    gray = preprocess_for_hog(image)

    feature = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        transform_sqrt=True,
        visualize=False,
        feature_vector=True,
    )

    return feature.astype(np.float32)


def extract_hog_feature_from_path(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    return extract_hog_feature(image)
