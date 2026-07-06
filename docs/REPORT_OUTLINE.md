# Dàn Ý Báo Cáo: Image Gradient & HOG + SVM

## 1. Giới thiệu

Đề tài xây dựng chương trình phát hiện người đi bộ trong video/webcam bằng phương pháp xử lý ảnh truyền thống: Image Gradient, HOG và SVM. Hệ thống đọc từng frame từ video hoặc webcam, trích xuất đặc trưng hình dạng bằng HOG, dùng SVM để phân loại vùng ảnh, sau đó vẽ bounding box và nhãn trực tiếp lên luồng video.

Mục tiêu:

- Hiểu Image Gradient, Magnitude và Orientation.
- Hiểu quy trình tạo đặc trưng HOG.
- Hiểu vai trò của SVM trong bài toán phân loại person/background.
- Xây dựng demo chạy realtime trên video/webcam hoặc video bất kỳ.
- Tải dataset ngoài từ INRIA/Hugging Face và train custom Linear SVM.

## 2. Cơ sở lý thuyết

### 2.1 Image Gradient

Image Gradient biểu diễn mức thay đổi cường độ sáng của ảnh theo không gian. Với ảnh xám `I(x, y)`, gradient gồm hai thành phần:

```text
Gx = dI/dx
Gy = dI/dy
```

Biên độ gradient:

```text
M(x, y) = sqrt(Gx^2 + Gy^2)
```

Hướng gradient:

```text
theta(x, y) = atan2(Gy, Gx)
```

Script minh họa: `src/visualize_gradient_hog.py`.

### 2.2 HOG

HOG mô tả hình dạng đối tượng dựa trên phân bố hướng gradient.

Các bước:

1. Resize vùng ảnh về kích thước chuẩn `64x128`.
2. Chuyển ảnh sang grayscale.
3. Tính gradient theo trục x và y.
4. Tính magnitude và orientation.
5. Chia ảnh thành các cell `8x8`.
6. Với mỗi cell, tạo histogram hướng gradient.
7. Gom nhiều cell thành block `2x2`.
8. Chuẩn hóa block bằng `L2-Hys`.
9. Ghép các block thành vector đặc trưng HOG.

### 2.3 SVM

SVM nhận vector HOG và phân loại vùng ảnh thành:

- `person`
- `background`

Với Linear SVM:

```text
f(x) = w^T x + b
```

Trong project có hai hướng:

- OpenCV pretrained HOG + SVM: dùng để demo realtime ổn định.
- Custom HOG + Linear SVM: model tự train từ INRIA subset.

## 3. Thiết kế chương trình

Luồng xử lý demo chính:

1. Đọc frame từ video hoặc webcam bằng `cv2.VideoCapture`.
2. Resize frame để tăng tốc.
3. Chọn ROI nếu người dùng truyền `--roi-pct`.
4. Dùng HOG + SVM để phát hiện người.
5. Lọc box theo tỷ lệ cao/rộng và kích thước.
6. Áp dụng Non-Maximum Suppression để bỏ box trùng.
7. Dùng tracker đơn giản để giảm nhấp nháy box.
8. Vẽ bounding box, nhãn `person`, score và FPS.
9. Hiển thị realtime hoặc lưu video kết quả.

Các file chính:

- `src/detector.py`: detector pretrained OpenCV, NMS, vẽ kết quả.
- `src/custom_detector.py`: custom sliding-window detector.
- `src/run_realtime.py`: chạy webcam/video bất kỳ, ghi video.
- `src/features.py`: trích xuất HOG feature.
- `src/download_inria_hf_subset.py`: tải subset INRIA từ Hugging Face.
- `src/bootstrap_dataset_from_video.py`: tạo dataset từ video, chỉ giữ như tùy chọn phụ.
- `src/train_custom_svm.py`: train Linear SVM.
- `src/evaluate_custom_svm.py`: đánh giá model.
- `src/visualize_gradient_hog.py`: xuất ảnh Gradient/HOG.

## 4. Dataset và thí nghiệm

Nguồn dữ liệu train:

- Dataset ngoài: INRIA Person mirror trên Hugging Face.
- Positive: 300 ảnh pedestrian từ `data_ped/pedestrians`.
- Negative: 300 negative patches crop từ `data_ped/no_pedestrians`.
- Video mẫu `dataset/videos/walking.mp4` chỉ dùng để demo, không dùng để train model custom.

Cách tạo dataset:

```bash
python src/download_inria_hf_subset.py --max-positive 300 --max-negative 300 --negative-patches-per-image 2 --clear-inria --clear-video-bootstrap
```

Cách train custom SVM:

```bash
python src/train_custom_svm.py
```

Kết quả train/test hiện tại:

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

## 5. Kết quả

Các kết quả đã có:

- Chương trình phát hiện người trên video/webcam hoặc video bất kỳ.
- Bounding box và nhãn được vẽ trực tiếp trên frame.
- FPS được hiển thị.
- Có thể lưu video kết quả.
- Có ảnh minh họa Gradient Magnitude, Gradient Orientation và HOG.
- Có custom model `models/custom_hog_svm.pkl`.
- Có report `outputs/custom_svm_report.txt`.

Lệnh demo pretrained:

```bash
python src/run_realtime.py --source path/to/video.mp4
```

Lệnh demo custom SVM:

```bash
python src/run_realtime.py --source path/to/video.mp4 --model custom --custom-max-width 360 --custom-step 48 --custom-scales 1.0,0.75,0.55,0.40 --custom-threshold 0.8
```

## 6. Nhận xét

Ưu điểm:

- Phương pháp dễ giải thích, phù hợp môn Nhập môn Xử lý ảnh số.
- Không cần GPU.
- Demo pretrained chạy ổn với video bất kỳ.
- Custom SVM đã train từ nguồn ngoài, không phụ thuộc video demo.

Hạn chế:

- HOG + SVM nhạy với góc nhìn, che khuất và nền nhiều cạnh.
- Custom SVM còn chậm vì sliding window thuần Python.
- INRIA subset trong project chỉ là subset 600 ảnh để giữ dung lượng vừa phải.

## 7. Kết luận

Project đáp ứng yêu cầu Chủ đề 2: trình bày được Image Gradient, HOG và SVM; xây dựng được demo phát hiện người đi bộ trên video/webcam; vẽ bounding box và nhãn realtime; đồng thời có phần tải dataset ngoài từ INRIA/Hugging Face, train custom Linear SVM và đánh giá kết quả.
