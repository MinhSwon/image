import argparse
import os
import sys

import cv2
import numpy as np
from skimage.feature import hog

from config import (
    OUTPUT_DIR,
    WINDOW_SIZE,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    HOG_BLOCK_NORM,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def normalize_to_uint8(img):
    img = img.astype(np.float32)
    min_val = img.min()
    max_val = img.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - min_val) / (max_val - min_val)
    img = (img * 255).astype(np.uint8)
    return img


def visualize(image_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    resized = cv2.resize(image, WINDOW_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Sobel gradient
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(gx, gy)
    orientation = cv2.phase(gx, gy, angleInDegrees=True)

    magnitude_img = normalize_to_uint8(magnitude)
    orientation_img = normalize_to_uint8(orientation)

    # HOG visualization
    _, hog_img = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        transform_sqrt=True,
        visualize=True,
        feature_vector=True,
    )

    hog_img = normalize_to_uint8(hog_img)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "input_resized.jpg"), resized)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "gradient_magnitude.jpg"), magnitude_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "gradient_orientation.jpg"), orientation_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "hog_visualization.jpg"), hog_img)

    print("Đã lưu ảnh minh họa vào outputs/:")
    print("- input_resized.jpg")
    print("- gradient_magnitude.jpg")
    print("- gradient_orientation.jpg")
    print("- hog_visualization.jpg")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Sobel Gradient and HOG")
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh đầu vào")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize(args.image)
