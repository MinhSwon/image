import glob
import os

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

from config import (
    CUSTOM_MODEL_PATH,
    MODEL_DIR,
    NEGATIVE_DIR,
    OUTPUT_DIR,
    POSITIVE_DIR,
)
from features import extract_hog_feature_from_path


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def collect_images(folder):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def build_dataset():
    pos_paths = collect_images(POSITIVE_DIR)
    neg_paths = collect_images(NEGATIVE_DIR)

    if len(pos_paths) == 0:
        raise RuntimeError(f"No positive images found in {POSITIVE_DIR}")

    if len(neg_paths) == 0:
        raise RuntimeError(f"No negative images found in {NEGATIVE_DIR}")

    X = []
    y = []

    print(f"Positive images: {len(pos_paths)}")
    print(f"Negative images: {len(neg_paths)}")

    print("Extracting HOG features for positive images...")
    for path in tqdm(pos_paths):
        try:
            X.append(extract_hog_feature_from_path(path))
            y.append(1)
        except Exception as e:
            print(f"Skip {path}: {e}")

    print("Extracting HOG features for negative images...")
    for path in tqdm(neg_paths):
        try:
            X.append(extract_hog_feature_from_path(path))
            y.append(0)
        except Exception as e:
            print(f"Skip {path}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def save_training_report(y, y_train, y_test, accuracy, report, matrix):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "custom_svm_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CUSTOM HOG + SVM TRAINING REPORT\n")
        f.write(f"Positive images: {int(np.sum(y == 1))}\n")
        f.write(f"Negative images: {int(np.sum(y == 0))}\n")
        f.write(f"Train samples: {len(y_train)}\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(str(matrix))
        f.write("\n")

    return report_path


def main():
    X, y = build_dataset()

    if len(np.unique(y)) < 2:
        raise RuntimeError("Dataset must contain both positive and negative images.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(C=1.0, class_weight="balanced", max_iter=20000)),
    ])

    """
    Huấn luyện mô hình Linear Support Vector Machine (Linear SVM).
    
    Cơ sở lý thuyết & Cấu hình Pipeline (Scikit-Learn):
    1. StandardScaler (Chuẩn hóa Z-score): 
       - Đặc trưng HOG đôi khi có thang đo (scale) chênh lệch nhỏ dù đã chuẩn hóa L2-Hys cục bộ.
       - StandardScaler đưa trung bình của từng feature về 0 và độ lệch chuẩn về 1. 
         Việc này giúp thuật toán tối ưu hóa (gradient descent/quang học) của SVM hội tụ 
         nhanh hơn và tìm được siêu phẳng (hyperplane) chính xác hơn.
    
    2. LinearSVC: 
       - Sử dụng kernel tuyến tính (Linear kernel) vì với không gian số chiều lớn của HOG 
         (3780 chiều), dữ liệu thường đã phân tách tuyến tính. Kernel tuyến tính giúp 
         huấn luyện và dự đoán rất nhanh so với RBF.
       - C=1.0: Tham số Regularization. Càng nhỏ thì biên (margin) càng lớn nhưng chấp nhận 
         nhiều điểm phân loại sai (Soft-margin). C=1.0 là giá trị cân bằng chuẩn.
       - class_weight="balanced": Rất quan trọng! Tập dữ liệu INRIA thường mất cân bằng 
         (số lượng ảnh nền negative nhiều hơn ảnh người positive). Tham số này tự động điều chỉnh 
         trọng số phạt (penalty) tỷ lệ nghịch với tần suất xuất hiện của lớp, giúp mô hình không bị 
         lệch (bias) về phía lớp nền.
       - max_iter=20000: Tăng số vòng lặp tối đa để đảm bảo mô hình hội tụ trên không gian lớn.
    """
    print("Training Linear SVM...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, target_names=["background", "person"])
    matrix = confusion_matrix(y_test, pred)

    print("\n===== TEST RESULT =====")
    print("Accuracy:", accuracy)
    print(report)
    print("Confusion matrix:")
    print(matrix)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, CUSTOM_MODEL_PATH)

    report_path = save_training_report(y, y_train, y_test, accuracy, report, matrix)

    print(f"\nSaved model: {CUSTOM_MODEL_PATH}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
