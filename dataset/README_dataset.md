# Dataset

Dataset dùng cho Chủ đề 2: HOG + SVM phát hiện người đi bộ.

## Cấu trúc

```text
dataset/
  positive/     ảnh/crop có người đi bộ
  negative/     ảnh nền không có người
  test_images/  ảnh dùng minh họa Gradient/HOG
  videos/       video đầu vào
```

## Dữ liệu hiện có

- `dataset/videos/walking.mp4`: video mẫu để demo.
- `dataset/positive/`: 80 ảnh positive được tạo từ video mẫu.
- `dataset/negative/`: 120 ảnh negative được lấy từ vùng nền ít overlap với người.
- `dataset/test_images/`: 5 frame dùng để xuất ảnh Gradient/HOG.

## Cách tạo lại dataset

```bash
python src/bootstrap_dataset_from_video.py --source dataset/videos/walking.mp4 --clear-generated
```

Script dùng detector HOG + SVM pretrained của OpenCV để bootstrap crop người từ video, sau đó lấy các vùng nền làm negative. Đây là cách tạo dataset nhanh để phục vụ báo cáo và train custom SVM.

Nếu muốn dataset mạnh hơn, nhóm nên tự quay/thêm ảnh:

- Positive: người toàn thân, nhiều khoảng cách, nhiều góc nhìn.
- Negative: nền đường, tường, cây, xe, hành lang, sân trường nhưng không có người.

## Train custom SVM

```bash
python src/train_custom_svm.py
```

Model được lưu ở:

```text
models/custom_hog_svm.pkl
```

Report được lưu ở:

```text
outputs/custom_svm_report.txt
```
