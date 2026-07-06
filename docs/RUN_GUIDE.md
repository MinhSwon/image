# Hướng Dẫn Chạy Project

## 1. Cài thư viện

```bash
python -m pip install -r requirements.txt
```

## 2. Chạy video bất kỳ

```bash
python src/run_realtime.py --source path/to/video.mp4
```

Ví dụ với video mẫu:

```bash
python src/run_realtime.py --source dataset/videos/walking.mp4
```

## 3. Demo webcam

```bash
python src/run_realtime.py --source 0
```

Nếu không mở được webcam, thử:

```bash
python src/run_realtime.py --source 1
```

## 4. Lưu video kết quả

```bash
python src/run_realtime.py --source path/to/video.mp4 --record outputs/result_new_video.mp4
```

Nếu chạy trong môi trường không mở được cửa sổ OpenCV:

```bash
python src/run_realtime.py --source path/to/video.mp4 --no-display --record outputs/result_new_video.mp4
```

## 5. Tạo dataset ngoài từ INRIA/Hugging Face

```bash
python src/download_inria_hf_subset.py --max-positive 300 --max-negative 300 --negative-patches-per-image 2 --clear-inria --clear-video-bootstrap
```

Kết quả hiện tại:

- Positive: 300 ảnh pedestrian từ INRIA.
- Negative: 300 negative patches từ ảnh no_pedestrians.
- Video mẫu chỉ dùng để demo, không dùng để train.

## 6. Train custom HOG + SVM

```bash
python src/train_custom_svm.py
```

Output:

- `models/custom_hog_svm.pkl`
- `outputs/custom_svm_report.txt`

Kết quả train/test hiện tại:

```text
Accuracy: 0.9667
Confusion matrix:
[[59  1]
 [ 3 57]]
```

## 7. Đánh giá custom SVM

```bash
python src/evaluate_custom_svm.py
```

Kết quả hiện tại trên toàn bộ dataset:

```text
Accuracy: 0.9933
Confusion matrix:
[[299   1]
 [  3 297]]
```

## 8. Demo custom SVM

Custom SVM dùng sliding window nên chậm hơn pretrained detector. Lệnh gợi ý với video bất kỳ:

```bash
python src/run_realtime.py --source path/to/video.mp4 --model custom --custom-max-width 360 --custom-step 48 --custom-scales 1.0,0.75,0.55,0.40 --custom-threshold 0.8
```

Ghi video ngắn không mở cửa sổ:

```bash
python src/run_realtime.py --source path/to/video.mp4 --model custom --custom-max-width 360 --custom-step 48 --custom-scales 1.0,0.75,0.55,0.40 --custom-threshold 0.8 --detect-every 2 --no-display --max-frames 50 --record outputs/custom_result_new_video.mp4
```

## 9. Xuất ảnh Gradient và HOG

```bash
python src/visualize_gradient_hog.py --image dataset/test_images/test_frame_000.jpg
```

Output:

- `outputs/input_resized.jpg`
- `outputs/gradient_magnitude.jpg`
- `outputs/gradient_orientation.jpg`
- `outputs/hog_visualization.jpg`

## 10. Lỗi thường gặp

Nếu phát hiện quá ít người:

```bash
python src/run_realtime.py --source path/to/video.mp4 --hit -0.3
```

Nếu quá nhiều box nhiễu:

```bash
python src/run_realtime.py --source path/to/video.mp4 --hit 0.3
```

Nếu custom SVM quá nhiễu, tăng threshold:

```bash
python src/run_realtime.py --source path/to/video.mp4 --model custom --custom-threshold 1.0
```
