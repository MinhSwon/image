# Nội Dung Gợi Ý Slide Thuyết Trình 15-20 Phút

## Slide 1: Tiêu đề

Phát hiện người đi bộ bằng Image Gradient, HOG và SVM trên Video/Webcam.

## Slide 2: Mục tiêu đề tài

- Tìm hiểu Image Gradient, Magnitude, Orientation.
- Tìm hiểu HOG descriptor.
- Dùng SVM để phân loại person/background.
- Xây dựng demo realtime có bounding box và nhãn.
- Tự tạo dataset và train thử custom Linear SVM.

## Slide 3: Quy trình tổng quát

```text
Input frame -> Grayscale -> Gradient -> HOG -> SVM -> Bounding Box -> Display/Record
```

## Slide 4: Image Gradient

Nội dung trình bày:

- Sobel theo trục x và y.
- `Gx`, `Gy`.
- Magnitude: độ mạnh của biên.
- Orientation: hướng thay đổi cường độ sáng.

Minh họa bằng:

- `outputs/gradient_magnitude.jpg`
- `outputs/gradient_orientation.jpg`

## Slide 5: HOG

Nội dung trình bày:

- Resize window `64x128`.
- Cell `8x8`.
- Block `2x2`.
- Histogram 9 hướng.
- Block normalization `L2-Hys`.

Minh họa:

- `outputs/hog_visualization.jpg`

## Slide 6: SVM

Nội dung trình bày:

- HOG biến ảnh thành vector đặc trưng.
- SVM học ranh giới giữa `person` và `background`.
- Linear SVM có hàm quyết định:

```text
f(x) = w^T x + b
```

## Slide 7: Thiết kế chương trình

Các module:

- `detector.py`: OpenCV HOG + SVM pretrained.
- `custom_detector.py`: custom sliding-window detector.
- `run_realtime.py`: chạy webcam/video.
- `features.py`: trích xuất HOG.
- `train_custom_svm.py`: train Linear SVM.
- `visualize_gradient_hog.py`: xuất ảnh minh họa.

## Slide 8: Dataset

Dataset hiện tại:

- Video: `dataset/videos/walking.mp4`
- Positive: 80 ảnh crop người.
- Negative: 120 ảnh nền.
- Test images: 5 ảnh.

Cách tạo:

```bash
python src/bootstrap_dataset_from_video.py --source dataset/videos/walking.mp4 --clear-generated
```

## Slide 9: Demo chính

Lệnh chạy:

```bash
python src/run_realtime.py --source dataset/videos/walking.mp4
```

Hoặc webcam:

```bash
python src/run_realtime.py --source 0
```

## Slide 10: Custom SVM

Train model:

```bash
python src/train_custom_svm.py
```

Kết quả train/test:

```text
Accuracy: 0.95
Confusion matrix:
[[22  2]
 [ 0 16]]
```

Đánh giá toàn bộ dataset:

```text
Accuracy: 0.99
Confusion matrix:
[[118   2]
 [  0  80]]
```

## Slide 11: So sánh pretrained và custom

OpenCV pretrained:

- Nhanh hơn.
- Ổn định hơn.
- Phù hợp demo realtime.

Custom SVM:

- Thể hiện khả năng tự tạo dataset và tự huấn luyện.
- Dễ giải thích về mặt học máy.
- Chậm hơn vì dùng sliding window.
- Cần nhiều dữ liệu hơn để giảm nhiễu.

## Slide 12: Kết quả

Kết quả đạt được:

- Phát hiện người trên video/webcam.
- Có bounding box, nhãn và FPS.
- Có lưu video kết quả.
- Có ảnh minh họa Gradient và HOG.
- Có custom model `models/custom_hog_svm.pkl`.

## Slide 13: Hạn chế

- HOG + SVM nhạy với background nhiều cạnh.
- Khó phát hiện người bị che khuất hoặc quá nhỏ.
- Custom dataset còn ít.
- Sliding window custom chạy chậm hơn deep learning hoặc detector tối ưu.

## Slide 14: Hướng phát triển

- Bổ sung thêm ảnh từ INRIA Person Dataset.
- Tăng số lượng negative hard samples.
- Tối ưu sliding window.
- Thử các mô hình hiện đại như YOLO để so sánh.

## Slide 15: Kết luận

Project đáp ứng yêu cầu Chủ đề 2:

- Có cơ sở lý thuyết Image Gradient, HOG, SVM.
- Có demo realtime trên video/webcam.
- Có bounding box và nhãn.
- Có minh họa các bước trung gian.
- Có tự tạo dataset và train custom SVM.
