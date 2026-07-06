# Hướng Dẫn Chạy Project

## 1. Cài thư viện

```bash
python -m pip install -r requirements.txt
```

## 2. Demo realtime bằng video

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
python src/run_realtime.py --source dataset/videos/walking.mp4 --record outputs/result.mp4
```

Nếu chạy trong môi trường không mở được cửa sổ OpenCV:

```bash
python src/run_realtime.py --source dataset/videos/walking.mp4 --no-display --record outputs/result.mp4
```

## 5. Tạo dataset positive/negative từ video mẫu

```bash
python src/bootstrap_dataset_from_video.py --source dataset/videos/walking.mp4 --clear-generated
```

Kết quả hiện tại:

- Positive: 80 ảnh
- Negative: 120 ảnh
- Test images: 5 ảnh

## 6. Train custom HOG + SVM

```bash
python src/train_custom_svm.py
```

Output:

- `models/custom_hog_svm.pkl`
- `outputs/custom_svm_report.txt`

## 7. Đánh giá custom SVM

```bash
python src/evaluate_custom_svm.py
```

Kết quả hiện tại trên toàn bộ dataset:

```text
Accuracy: 0.99
Confusion matrix:
[[118   2]
 [  0  80]]
```

## 8. Demo custom SVM

Custom SVM dùng sliding window nên chậm hơn pretrained detector. Lệnh gợi ý:

```bash
python src/run_realtime.py --source dataset/videos/walking.mp4 --model custom --custom-max-width 360 --custom-step 48 --custom-scales 1.0,0.75,0.55,0.40 --custom-threshold 0.8
```

Ghi video ngắn không mở cửa sổ:

```bash
python src/run_realtime.py --source dataset/videos/walking.mp4 --model custom --custom-max-width 360 --custom-step 48 --custom-scales 1.0,0.75,0.55,0.40 --custom-threshold 0.8 --detect-every 2 --no-display --max-frames 50 --record outputs/custom_result_short.mp4
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
python src/run_realtime.py --source dataset/videos/walking.mp4 --hit -0.3
```

Nếu quá nhiều box nhiễu:

```bash
python src/run_realtime.py --source dataset/videos/walking.mp4 --hit 0.3
```

Nếu custom SVM quá nhiễu, tăng threshold:

```bash
python src/run_realtime.py --source dataset/videos/walking.mp4 --model custom --custom-threshold 1.0
```
