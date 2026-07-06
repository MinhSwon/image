# HOG + SVM Realtime Pedestrian Detection

Dự án cuối kỳ cho **Chủ đề 2: Image Gradient & HOG + SVM - Phát hiện đối tượng**.

Project đã hoàn thiện các phần chính:

- Demo phát hiện người đi bộ realtime bằng webcam hoặc video bất kỳ.
- Vẽ bounding box, nhãn `person`, confidence score và FPS.
- Minh họa Image Gradient: Sobel, Magnitude, Orientation.
- Minh họa HOG visualization.
- Tải dataset ngoài từ **INRIA Person mirror trên Hugging Face**.
- Train Linear SVM tùy chỉnh bằng HOG feature.
- Chạy được 2 chế độ:
  - `opencv`: HOG + SVM pretrained của OpenCV, dùng để demo realtime ổn định.
  - `custom`: HOG + Linear SVM do project tự train từ INRIA subset.

Nguồn dataset ngoài dùng trong project:

- Hugging Face: https://huggingface.co/datasets/marcelarosalesj/inria-person
- Dataset viewer cho thấy mirror này có khoảng 2.49k rows, 2 class và tổng file khoảng 239MB.

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

## 2. Demo chính với video bất kỳ

Thay `path/to/video.mp4` bằng video của bạn:

```bash
python src/run_realtime.py --source path/to/video.mp4
```

Chạy video mẫu đi kèm project:

```bash
python src/run_realtime.py --source dataset/videos/walking.mp4
```

Chạy webcam:

```bash
python src/run_realtime.py --source 0
```

Lưu video kết quả:

```bash
python src/run_realtime.py --source path/to/video.mp4 --record outputs/result_new_video.mp4
```

Chạy không mở cửa sổ OpenCV, chỉ ghi video:

```bash
python src/run_realtime.py --source path/to/video.mp4 --no-display --record outputs/result_new_video.mp4
```

Phím khi mở cửa sổ:

- `q`: thoát
- `s`: lưu frame hiện tại vào `outputs/`
- `p`: pause/resume

## 3. Tạo dataset ngoài từ INRIA/Hugging Face

Dataset train hiện tại **không lấy từ video demo**. Nó được tải từ mirror INRIA trên Hugging Face.

Lệnh tạo lại subset 300 positive + 300 negative:

```bash
python src/download_inria_hf_subset.py --max-positive 300 --max-negative 300 --negative-patches-per-image 2 --clear-inria --clear-video-bootstrap
```

Kết quả hiện tại:

- `dataset/positive/`: 300 ảnh pedestrian từ INRIA.
- `dataset/negative/`: 300 negative patches từ ảnh không có pedestrian.
- `dataset/videos/walking.mp4`: chỉ dùng làm video demo, không dùng để train.

Raw ảnh tải về được cache ở `dataset/external_raw/inria_hf/` nhưng thư mục này được `.gitignore` để tránh đẩy trùng dữ liệu lên Git.

## 4. Train custom HOG + SVM

```bash
python src/train_custom_svm.py
```

Kết quả sinh ra:

- `models/custom_hog_svm.pkl`
- `outputs/custom_svm_report.txt`

Kết quả train/test hiện tại:

- Accuracy trên tập test: `0.9667`
- Confusion matrix:

```text
[[59  1]
 [ 3 57]]
```

Đánh giá toàn bộ dataset:

```bash
python src/evaluate_custom_svm.py
```

Kết quả toàn bộ dataset hiện tại:

- Accuracy: `0.9933`
- Confusion matrix:

```text
[[299   1]
 [  3 297]]
```

## 5. Demo bằng custom SVM

Custom SVM dùng sliding window nên chậm hơn detector pretrained. Nên dùng frame nhỏ và bước trượt lớn khi demo.

Với video bất kỳ:

```bash
python src/run_realtime.py --source path/to/video.mp4 --model custom --custom-max-width 360 --custom-step 48 --custom-scales 1.0,0.75,0.55,0.40 --custom-threshold 0.8
```

Với video mẫu:

```bash
python src/run_realtime.py --source dataset/videos/walking.mp4 --model custom --custom-max-width 360 --custom-step 48 --custom-scales 1.0,0.75,0.55,0.40 --custom-threshold 0.8
```

Nếu chỉ muốn ghi video ngắn để kiểm tra:

```bash
python src/run_realtime.py --source path/to/video.mp4 --model custom --custom-max-width 360 --custom-step 48 --custom-scales 1.0,0.75,0.55,0.40 --custom-threshold 0.8 --detect-every 2 --no-display --max-frames 50 --record outputs/custom_result_new_video.mp4
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

- Dataset train custom SVM đã chuyển sang nguồn ngoài INRIA/Hugging Face, không còn lấy từ video demo.
- Demo realtime chính dùng OpenCV HOG + SVM pretrained để đảm bảo tốc độ và độ ổn định.
- Project có thêm custom Linear SVM tự train để đáp ứng tiêu chí mở rộng.
- Custom SVM có kết quả tốt trên INRIA subset nhưng vẫn chậm hơn pretrained detector vì dùng sliding window thuần Python.
