# Dataset

Dataset dùng cho Chủ đề 2: HOG + SVM phát hiện người đi bộ.

## Cấu trúc

```text
dataset/
  positive/      ảnh pedestrian dùng train SVM
  negative/      ảnh nền/negative patches dùng train SVM
  test_images/   ảnh dùng minh họa Gradient/HOG
  videos/        video đầu vào để demo
  external_raw/  raw cache tải từ nguồn ngoài, không commit lên Git
```

## Dữ liệu train hiện tại

Dataset train hiện tại **không lấy từ video demo**. Project dùng subset từ mirror INRIA Person trên Hugging Face:

```text
https://huggingface.co/datasets/marcelarosalesj/inria-person
```

Hiện có:

- `dataset/positive/`: 300 ảnh pedestrian từ thư mục `data_ped/pedestrians`.
- `dataset/negative/`: 300 negative patches crop từ thư mục `data_ped/no_pedestrians`.
- `dataset/videos/walking.mp4`: chỉ dùng làm video demo.
- `dataset/test_images/`: ảnh minh họa Gradient/HOG.

## Tạo lại dataset INRIA subset

```bash
python src/download_inria_hf_subset.py --max-positive 300 --max-negative 300 --negative-patches-per-image 2 --clear-inria --clear-video-bootstrap
```

Script sẽ:

1. Liệt kê file từ Hugging Face dataset API.
2. Tải ảnh pedestrian làm positive.
3. Tải ảnh no_pedestrians rồi crop ngẫu nhiên các vùng nền tỷ lệ gần 64x128 làm negative.
4. Xóa các ảnh `auto_*` tạo từ video cũ nếu dùng `--clear-video-bootstrap`.

Raw ảnh tải về nằm ở:

```text
dataset/external_raw/inria_hf/
```

Thư mục này được `.gitignore` để tránh đẩy dữ liệu thô trùng lặp lên Git.

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
