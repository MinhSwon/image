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
            decision_threshold=0.5, # Có thể tinh chỉnh
            window_step=16,         # Tăng bước nhảy để chạy nhanh hơn trên Realtime
            scales=[1.0, 0.75, 0.5]
        )
    except Exception as e:
        print(f"Lỗi tải mô hình: {e}")
        print("Vui lòng chạy 'python src/train_custom_svm_faces.py' trước để huấn luyện mô hình.")
        return

    # Xử lý input source
    if args.source.isdigit():
        # Dùng CAP_DSHOW để tránh lỗi MSMF (-1072875772) trên Windows Webcam
        cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print("Không thể mở video source.")
        return

    print("Nhấn 'q' để thoát.")
    
    import time
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Để chạy realtime mượt hơn với Sliding Window (vốn rất nặng),
        # ta thu nhỏ khung hình xuống một chút
        frame = cv2.resize(frame, (640, 480))
        
        start_time = time.time()
        
        # Phát hiện khuôn mặt
        boxes, scores, _ = detector.detect(frame)
        
        end_time = time.time()
        fps = 1.0 / (end_time - start_time + 1e-5)

        # Vẽ Bounding box
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Hiển thị FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Custom HOG+SVM Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
