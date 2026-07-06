# Nội Dung Gợi Ý Slide Thuyết Trình 15-20 Phút

## Slide 1: Tiêu đề

Phát hiện người đi bộ bằng Image Gradient, HOG và SVM trên Video/Webcam.

## Slide 2: Mục tiêu đề tài

- Tìm hiểu Image Gradient, Magnitude, Orientation.
- Tìm hiểu HOG descriptor.
- Dùng SVM để phân loại person/background.
- Xây dựng demo realtime có bounding box và nhãn.
- Tải INRIA subset từ Hugging Face và train custom Linear SVM.

## Slide 3: Quy trình tổng quát

```text
Input frame -> Grayscale -> Gradient -> HOG -> SVM -> Bounding Box -> Display/Record
```

## Slide 4: Image Gradient

- Sobel theo trục x và y.
- `Gx`, `Gy`.
- Magnitude: độ mạnh của biên.
- Orientation: hướng thay đổi cường độ sáng.

Minh họa:

- `outputs/gradient_magnitude.jpg`
- `outputs/gradient_orientation.jpg`

## Slide 5: HOG

- Resize window `64x128`.
- Cell `8x8`.
- Block `2x2`.
- Histogram 9 hướng.
- Block normalization `L2-Hys`.

Minh họa:

- `outputs/hog_visualization.jpg`

## Slide 6: SVM

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
- `run_realtime.py`: chạy webcam/video bất kỳ.
- `features.py`: trích xuất HOG.
- `download_inria_hf_subset.py`: tải subset INRIA.
- `train_custom_svm.py`: train Linear SVM.
- `visualize_gradient_hog.py`: xuất ảnh minh họa.

## Slide 8: Dataset

Dataset hiện tại:

- Nguồn train: INRIA Person mirror trên Hugging Face.
- Positive: 300 ảnh pedestrian.
- Negative: 300 negative patches.
- Video `walking.mp4` chỉ dùng làm demo.

Cách tạo:

```bash
python src/download_inria_hf_subset.py --max-positive 300 --max-negative 300 --negative-patches-per-image 2 --clear-inria --clear-video-bootstrap
```

## Slide 9: Demo

Video bất kỳ:

```bash
python src/run_realtime.py --source path/to/video.mp4
```

Webcam:

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
Accuracy: 0.9667
Confusion matrix:
[[59  1]
 [ 3 57]]
```

Đánh giá toàn bộ dataset:

```text
Accuracy: 0.9933
Confusion matrix:
[[299   1]
 [  3 297]]
```

## Slide 11: So sánh pretrained và custom

OpenCV pretrained:

- Nhanh hơn.
- Ổn định hơn.
- Phù hợp demo realtime.

Custom SVM:

- Được train từ nguồn ngoài INRIA.
- Không phụ thuộc video demo cố định.
- Dễ giải thích về mặt học máy.
- Chậm hơn vì dùng sliding window.

## Slide 12: Kết quả

- Phát hiện người trên video/webcam.
- Có bounding box, nhãn và FPS.
- Có lưu video kết quả.
- Có ảnh minh họa Gradient và HOG.
- Có custom model `models/custom_hog_svm.pkl`.

## Slide 13: Hạn chế

- HOG + SVM nhạy với background nhiều cạnh.
- Khó phát hiện người bị che khuất hoặc quá nhỏ.
- INRIA subset trong repo chỉ là subset 600 ảnh.
- Sliding window custom chạy chậm hơn deep learning hoặc detector tối ưu.

## Slide 14: Hướng phát triển

- Tải full INRIA dataset hoặc bổ sung hard negatives.
- Thêm nhiều video ở bối cảnh khác nhau.
- Tối ưu sliding window.
- Thử các mô hình hiện đại như YOLO để so sánh.

## Slide 15: Kết luận

Project đáp ứng yêu cầu Chủ đề 2:

- Có cơ sở lý thuyết Image Gradient, HOG, SVM.
- Có demo realtime trên video/webcam.
- Có bounding box và nhãn.
- Có minh họa các bước trung gian.
- Có tải dataset ngoài, custom model và kết quả đánh giá.
