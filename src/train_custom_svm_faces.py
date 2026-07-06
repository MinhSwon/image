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

from config import NEGATIVE_DIR, WINDOW_SIZE, MODEL_DIR
from features import extract_hog_feature

def load_face_data():
    """
    Tải dữ liệu khuôn mặt từ LFW (Labeled Faces in the Wild) làm Positive samples.
    """
    print("Đang tải dữ liệu khuôn mặt LFW (Positive samples)...")
    # Để nguyên độ phân giải gốc (resize=1.0) để HOG bắt được các đường nét sắc sảo trên mặt
    lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=1.0, data_home="dataset/external_raw")
    images = lfw_people.images
    
    features = []
    labels = []
    
    print(f"Bắt đầu trích xuất HOG từ {len(images)} ảnh khuôn mặt...")
    for img in images:
        img_uint8 = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        try:
            feat = extract_hog_feature(img_bgr, window_size=(64, 64))
            features.append(feat)
            labels.append(1)
        except Exception as e:
            pass
            
    return features, labels

def load_negative_data(max_samples=5000):
    """
    Tải ảnh nền từ INRIA làm Negative samples, cộng thêm các ảnh trống (màu trơn) 
    để chống nhận diện sai các bức tường.
    """
    print("Đang tải dữ liệu ảnh nền INRIA và tạo ảnh trống (Negative samples)...")
    neg_paths = glob.glob(os.path.join(NEGATIVE_DIR, "*.jpg")) + glob.glob(os.path.join(NEGATIVE_DIR, "*.png"))
    
    features = []
    labels = []
    
    # 1. Thêm khoảng 500 ảnh trống (Solid colors) để chữa bệnh "nhận diện tường thành mặt"
    print("Đang tạo các mẫu nền trống (Tường trắng, đen, xám)...")
    for _ in range(500):
        color = np.random.randint(0, 256)
        blank_img = np.full((64, 64, 3), color, dtype=np.uint8)
        try:
            feat = extract_hog_feature(blank_img, window_size=(64, 64))
            features.append(feat)
            labels.append(0)
        except:
            pass

    # 2. Lấy nhiều crop ngẫu nhiên từ ảnh INRIA
    count = len(features)
    print("Đang cắt ngẫu nhiên từ ảnh phong cảnh...")
    for path in neg_paths:
        if count >= max_samples:
            break
        img = cv2.imread(path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        win_w, win_h = 64, 64
        
        # Cắt 5 vùng ngẫu nhiên cho mỗi ảnh phong cảnh để làm phong phú dữ liệu
        for _ in range(5):
            if count >= max_samples:
                break
            if h > win_h and w > win_w:
                y = np.random.randint(0, h - win_h)
                x = np.random.randint(0, w - win_w)
                crop = img[y:y+win_h, x:x+win_w]
                
                try:
                    feat = extract_hog_feature(crop, window_size=(64, 64))
                    features.append(feat)
                    labels.append(0)
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
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "custom_hog_svm_face_model.pkl")
    joblib.dump(model, model_path)
    print(f"Mô hình đã được lưu tại: {model_path}")

if __name__ == "__main__":
    train_face_model()
