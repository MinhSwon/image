import cv2
import numpy as np
from skimage.feature import hog

from config import (
    WINDOW_SIZE,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    HOG_BLOCK_NORM,
)


def preprocess_for_hog(image):
    """
    Tiền xử lý ảnh trước khi trích xuất đặc trưng HOG.
    
    Quy trình:
    1. Resize ảnh về kích thước chuẩn (mặc định 64x128). Kích thước này tỷ lệ 1:2
       phù hợp với dáng đứng của người đi bộ.
    2. Chuyển đổi sang ảnh xám (Grayscale). Do HOG chủ yếu quan tâm đến gradient
       (sự thay đổi cường độ sáng tạo nên hình dáng) chứ không quan tâm đến màu sắc.
    3. Cân bằng lược đồ xám (Histogram Equalization) để tăng độ tương phản,
       giúp các đường biên trở nên rõ ràng hơn ở cả vùng tối và sáng.
    """
    image = cv2.resize(image, WINDOW_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray


def extract_hog_feature(image):
    """
    Trích xuất vector đặc trưng HOG (Histogram of Oriented Gradients).
    
    Tham số HOG được sử dụng:
    - orientations (9): Chia vòng tròn góc (0-180 độ đối với unsigned gradient) thành 9 bin,
      mỗi bin 20 độ. Thực nghiệm chứng minh 9 bin là tối ưu để nắm bắt hình dạng.
    - pixels_per_cell (8x8): Chia ảnh thành các ô nhỏ 8x8 pixel. Kích thước này đủ nhỏ 
      để bắt các chi tiết nhưng đủ lớn để không bị nhiễu.
    - cells_per_block (2x2): Gom 4 ô (cells) thành 1 khối (block). 
    - block_norm (L2-Hys): Chuẩn hóa L2-Hysteresis trên mỗi block giúp giảm thiểu 
      sự phụ thuộc vào độ sáng và độ tương phản cục bộ.
    - transform_sqrt (True): Áp dụng căn bậc hai lên ảnh gốc (Gamma compression)
      để giảm bớt sự biến đổi ánh sáng gắt (shadows/illuminations).
      
    Kết quả trả về: Vector 1D có kích thước 3780 chiều (đối với ảnh 64x128).
    """
    gray = preprocess_for_hog(image)

    feature = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        transform_sqrt=True,
        visualize=False,
        feature_vector=True,
    )

    return feature.astype(np.float32)


def extract_hog_feature_from_path(image_path):
    """
    Tiện ích đọc ảnh từ đường dẫn file và trích xuất HOG.
    Hỗ trợ xử lý lỗi nếu file không tồn tại hoặc bị hỏng.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    return extract_hog_feature(image)
