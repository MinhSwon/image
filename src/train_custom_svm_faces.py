import os
import glob
import cv2
import numpy as np
import joblib
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from config import NEGATIVE_DIR, WINDOW_SIZE, MODELS_DIR
from features import extract_hog_feature

def load_face_data():
    """
    Tải dữ liệu khuôn mặt từ LFW (Labeled Faces in the Wild) làm Positive samples.
    """
    print("Đang tải dữ liệu khuôn mặt LFW (Positive samples)...")
    # Lấy các khuôn mặt, resize=0.4 để ảnh nhỏ lại cho nhanh
    lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.4, data_home="dataset/external_raw")
    images = lfw_people.images
    
    features = []
    labels = []
    
    print(f"Bắt đầu trích xuất HOG từ {len(images)} ảnh khuôn mặt...")
    for img in images:
        # LFW images là ảnh xám float32 từ 0-1, ta chuyển về uint8 0-255
        img_uint8 = (img * 255).astype(np.uint8)
        # Chuyển sang BGR 3 kênh để tương thích với hàm preprocess_for_hog
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        # Extract
        try:
            feat = extract_hog_feature(img_bgr)
            features.append(feat)
            labels.append(1) # Lớp 1: Khuôn mặt
        except Exception as e:
            pass
            
    return features, labels

def load_negative_data(max_samples=2000):
    """
    Tải ảnh nền từ INRIA làm Negative samples.
    """
    print("Đang tải dữ liệu ảnh nền INRIA (Negative samples)...")
    neg_paths = glob.glob(os.path.join(NEGATIVE_DIR, "*.jpg")) + glob.glob(os.path.join(NEGATIVE_DIR, "*.png"))
    
    features = []
    labels = []
    
    count = 0
    for path in neg_paths:
        if count >= max_samples:
            break
        img = cv2.imread(path)
        if img is None:
            continue
            
        # Với ảnh nền, ta cắt ngẫu nhiên 1 vùng 64x128 để extract HOG
        h, w = img.shape[:2]
        if h > WINDOW_SIZE[1] and w > WINDOW_SIZE[0]:
            y = np.random.randint(0, h - WINDOW_SIZE[1])
            x = np.random.randint(0, w - WINDOW_SIZE[0])
            crop = img[y:y+WINDOW_SIZE[1], x:x+WINDOW_SIZE[0]]
            
            try:
                feat = extract_hog_feature(crop)
                features.append(feat)
                labels.append(0) # Lớp 0: Nền
                count += 1
            except:
                pass
                
    return features, labels

def train_face_model():
    # 1. Load Data
    pos_feat, pos_labels = load_face_data()
    neg_feat, neg_labels = load_negative_data(max_samples=max(len(pos_feat)*2, 1000)) # Lấy số âm bản gấp đôi
    
    X = np.array(pos_feat + neg_feat)
    y = np.array(pos_labels + neg_labels)
    
    if len(np.unique(y)) < 2:
        print("Lỗi: Không đủ dữ liệu 2 lớp để huấn luyện!")
        return
        
    print(f"Tổng số mẫu: {len(X)} (Khuôn mặt: {sum(y==1)}, Nền: {sum(y==0)})")
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(C=1.0, class_weight="balanced", max_iter=20000)),
    ])
    
    print("Đang huấn luyện mô hình Face Detection SVM...")
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, target_names=["background", "face"])
    
    print("\n===== KẾT QUẢ HUẤN LUYỆN FACE DETECTION =====")
    print("Độ chính xác (Accuracy):", accuracy)
    print(report)
    
    # 5. Save Model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "custom_hog_svm_face_model.pkl")
    joblib.dump(model, model_path)
    print(f"Mô hình đã được lưu tại: {model_path}")

if __name__ == "__main__":
    train_face_model()
