# HOG + SVM Realtime Pedestrian Detection

Dự án cuối kỳ cho **Chủ đề 2: Image Gradient & HOG + SVM - Phát hiện đối tượng**.

Project đã có đủ các phần chính:

- Demo phát hiện người đi bộ realtime bằng webcam hoặc video.
- Vẽ bounding box, nhãn `person`, confidence score và FPS.
- Minh họa Image Gradient: Sobel, Magnitude, Orientation.
- Minh họa HOG visualization.
- Tạo dataset positive/negative từ video mẫu.
- Train Linear SVM tùy chỉnh bằng HOG feature.
- Chạy được 2 chế độ:
  - `opencv`: HOG + SVM pretrained của OpenCV, dùng để demo realtime ổn định.
  - `custom`: HOG + Linear SVM do project tự train, dùng để chứng minh phần học máy truyền thống.

## 1. Cài đặt

```bash
python -m pip install -r requirements.txt
```

Nếu muốn dùng môi trường ảo:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Demo chính: OpenCV HOG + SVM pretrained

Chạy webcam:

```bash
python src/run_realtime.py --source 0
```

Chạy video mẫu:

```bash
python src/run_realtime.py --source dataset/videos/walking.mp4
```

Lưu video kết quả:

```bash
python src/run_realtime.py --source dataset/videos/walking.mp4 --record outputs/result.mp4
```

Chạy không mở cửa sổ OpenCV, chỉ ghi video:

```bash
python src/run_realtime.py --source dataset/videos/walking.mp4 --no-display --record outputs/result.mp4
```

Phím khi mở cửa sổ:

- `q`: thoát
- `s`: lưu frame hiện tại vào `outputs/`
- `p`: pause/resume

## 3. Tạo dataset từ video mẫu

Script này lấy crop người từ video làm `positive`, lấy vùng nền ít overlap làm `negative`, và lưu vài ảnh test để minh họa Gradient/HOG.

```bash
python src/bootstrap_dataset_from_video.py --source dataset/videos/walking.mp4 --clear-generated
```

Kết quả hiện tại trong project:

- `dataset/positive/`: 80 ảnh positive
- `dataset/negative/`: 120 ảnh negative
- `dataset/test_images/`: 5 ảnh test

## 4. Train custom HOG + SVM

```bash
python src/train_custom_svm.py
```

Kết quả sinh ra:

- `models/custom_hog_svm.pkl`
- `outputs/custom_svm_report.txt`

Kết quả train/test hiện tại:

- Accuracy trên tập test: `0.95`
- Confusion matrix:

```text
[[22  2]
 [ 0 16]]
```

Đánh giá toàn bộ dataset:

```bash
python src/evaluate_custom_svm.py
```

Kết quả đánh giá toàn bộ dataset hiện tại:

- Accuracy: `0.99`
- Confusion matrix:

```text
[[118   2]
 [  0  80]]
```

## 5. Demo bằng custom SVM

Custom SVM dùng sliding window nên chậm hơn detector pretrained. Nên dùng frame nhỏ hơn và bước trượt lớn hơn khi demo.

```bash
python src/run_realtime.py --source dataset/videos/walking.mp4 --model custom --custom-max-width 360 --custom-step 48 --custom-scales 1.0,0.75,0.55,0.40 --custom-threshold 0.8
```

Nếu chỉ muốn ghi video ngắn để kiểm tra:

```bash
python src/run_realtime.py --source dataset/videos/walking.mp4 --model custom --custom-max-width 360 --custom-step 48 --custom-scales 1.0,0.75,0.55,0.40 --custom-threshold 0.8 --detect-every 2 --no-display --max-frames 50 --record outputs/custom_result_short.mp4
```

## 6. Xuất ảnh minh họa Gradient và HOG

```bash
python src/visualize_gradient_hog.py --image dataset/test_images/test_frame_000.jpg
```

Kết quả trong `outputs/`:

- `input_resized.jpg`
- `gradient_magnitude.jpg`
- `gradient_orientation.jpg`
- `hog_visualization.jpg`

## 7. Gợi ý khi thuyết trình

Nên nói rõ:

- Demo realtime chính dùng OpenCV HOG + SVM pretrained để đảm bảo tốc độ và độ ổn định.
- Project có thêm phần tự tạo dataset và tự train Linear SVM để đáp ứng tiêu chí mở rộng.
- Custom SVM có accuracy tốt trên dataset bootstrap, nhưng vì dữ liệu còn nhỏ nên chưa ổn định bằng pretrained model.
- Đây là đặc điểm hợp lý của HOG + SVM truyền thống: dễ hiểu, không cần GPU, nhưng nhạy với góc nhìn, che khuất, kích thước người và nền nhiều cạnh.
