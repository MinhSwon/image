import cv2
import argparse
from custom_detector import CustomHOGSVMSlidingWindowDetector

def main():
    parser = argparse.ArgumentParser(description="Real-time Face Detection using Custom HOG + SVM")
    parser.add_argument("--source", type=str, default="0", help="Nguồn video (0 cho webcam, hoặc đường dẫn file .mp4)")
    args = parser.parse_args()

    # Khởi tạo detector khuôn mặt
    print("Đang tải mô hình Face Detection SVM...")
    try:
        detector = CustomHOGSVMSlidingWindowDetector(
            model_path="models/custom_hog_svm_face_model.pkl",
            decision_threshold=0.5, # Hạ ngưỡng xuống để dễ bắt mặt hơn (vì tường đã bị loại trừ nhờ Train lại)
            window_step=16,         # Quét dày hơn (16 pixel/bước) để không lọt lưới khuôn mặt
            scales=[1.25, 1.0, 0.8, 0.6, 0.4, 0.25], # Bổ sung nhiều kích thước từ to đến nhỏ
            window_size=(64, 64)
        )
    except Exception as e:
        print(f"Lỗi tải mô hình: {e}")
        print("Vui lòng chạy 'python src/train_custom_svm_faces.py' trước để huấn luyện mô hình.")
        return

    # Xử lý input source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print("Không thể mở video source.")
        return

    print("Nhấn 'q' để thoát.")
    
    import time
    
    frame_count = 0
    last_boxes = []
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (480, 360))
        frame_count += 1
        
        # Cập nhật HOG mỗi 3 frame
        if frame_count % 3 == 0:
            boxes, scores, _ = detector.detect(frame)
            last_boxes = boxes
        else:
            boxes = last_boxes
        
        # Tính FPS trung bình để không bị lỗi số ảo
        now = time.time()
        elapsed = now - prev_time
        fps = 1.0 / (elapsed + 1e-5)
        prev_time = now

        # Vẽ Bounding box
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Custom HOG+SVM Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
