import glob
import os
import sys

import joblib
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import POSITIVE_DIR, NEGATIVE_DIR, CUSTOM_MODEL_PATH
from features import extract_hog_feature_from_path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def collect_images(folder):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def main():
    if not os.path.exists(CUSTOM_MODEL_PATH):
        raise RuntimeError("Chưa có custom model. Chạy python src/train_custom_svm.py trước.")

    model = joblib.load(CUSTOM_MODEL_PATH)

    paths = []
    labels = []

    for p in collect_images(POSITIVE_DIR):
        paths.append(p)
        labels.append(1)

    for p in collect_images(NEGATIVE_DIR):
        paths.append(p)
        labels.append(0)

    X = []
    y = []

    for path, label in tqdm(list(zip(paths, labels))):
        try:
            X.append(extract_hog_feature_from_path(path))
            y.append(label)
        except Exception as e:
            print(f"Bỏ qua {path}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    pred = model.predict(X)

    print("\n===== ĐÁNH GIÁ TOÀN BỘ DATASET =====")
    print("Accuracy:", accuracy_score(y, pred))
    print(classification_report(y, pred, target_names=["background", "person"]))
    print("Confusion matrix:")
    print(confusion_matrix(y, pred))


if __name__ == "__main__":
    main()
