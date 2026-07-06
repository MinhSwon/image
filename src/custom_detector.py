import os

import cv2
import joblib
import numpy as np

from config import CUSTOM_MODEL_PATH, WINDOW_SIZE
from detector import (
    non_max_suppression_with_scores,
    parse_roi_pct,
    remove_inner_boxes,
    resize_keep_aspect,
)
from features import extract_hog_feature


class CustomHOGSVMSlidingWindowDetector:
    """
    Bộ phát hiện tùy chỉnh sử dụng HOG + Linear SVM (Custom Model).
    
    Cơ sở lý thuyết và Thuật toán:
    Vì mô hình Custom SVM tự huấn luyện chỉ nhận đầu vào cố định là cửa sổ 64x128 pixel, 
    nhóm đã triển khai giải thuật Quét cửa sổ trượt (Sliding Window) đa tỷ lệ (Multi-scale)
    bằng Python thuần:
    
    1. Kim tự tháp ảnh (Image Pyramid): Ảnh đầu vào được thu nhỏ liên tục qua các tỷ lệ 
       (scales) khác nhau. Việc thu nhỏ ảnh tương đương với việc phóng to cửa sổ quét, 
       giúp phát hiện người ở các khoảng cách xa gần khác nhau.
    2. Cửa sổ trượt (Sliding Window): Tại mỗi tỷ lệ ảnh, một cửa sổ cố định 64x128 
       trượt khắp ảnh từ trái sang phải, từ trên xuống dưới với bước nhảy `window_step`.
    3. HOG + SVM: Mỗi vùng ảnh con bị cắt bởi cửa sổ sẽ được tính toán vector HOG, 
       sau đó đưa qua hàm quyết định `decision_function` của SVM.
    4. Nếu điểm quyết định (score) lớn hơn `decision_threshold`, hệ thống đánh dấu đó là người.
    
    Lưu ý: Giải thuật này viết bằng Python lồng vòng lặp nên chạy khá chậm, nhưng nó 
    giúp chứng minh toàn vẹn quy trình học máy truyền thống từ xử lý thô đến học máy.
    """

    def __init__(
        self,
        model_path=CUSTOM_MODEL_PATH,
        max_width=640,
        roi_pct=None,
        window_step=32,
        scales=None,
        decision_threshold=0.0,
        nms_thresh=0.35,
        batch_size=128,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Custom model not found: {model_path}. "
                "Run python src/train_custom_svm.py first."
            )

        self.model = joblib.load(model_path)
        self.model_path = model_path
        self.max_width = max_width
        self.roi_pct = roi_pct
        self.window_step = max(4, int(window_step))
        self.scales = scales or [1.0, 0.85, 0.70, 0.55, 0.45, 0.35]
        self.decision_threshold = decision_threshold
        self.nms_thresh = nms_thresh
        self.batch_size = max(1, int(batch_size))

    def _flush_batch(self, features, coords, boxes, scores):
        if not features:
            return

        X = np.array(features, dtype=np.float32)

        if hasattr(self.model, "decision_function"):
            raw_scores = self.model.decision_function(X)
        else:
            raw_scores = self.model.predict(X)

        raw_scores = np.array(raw_scores).reshape(-1)

        for coord, score in zip(coords, raw_scores):
            score = float(score)
            if score < self.decision_threshold:
                continue

            boxes.append(coord)
            scores.append(score)

        features.clear()
        coords.clear()

    def detect(self, frame):
        work_frame, scale_back = resize_keep_aspect(frame, self.max_width)
        work_h, work_w = work_frame.shape[:2]

        roi_box_work = None
        roi_box_original = None

        if self.roi_pct is not None:
            x1, y1, x2, y2 = parse_roi_pct(self.roi_pct, work_w, work_h)
            roi_frame = work_frame[y1:y2, x1:x2].copy()
            roi_box_work = (x1, y1, x2, y2)
            roi_box_original = (
                int(x1 * scale_back),
                int(y1 * scale_back),
                int(x2 * scale_back),
                int(y2 * scale_back),
            )
        else:
            roi_frame = work_frame

        roi_h, roi_w = roi_frame.shape[:2]
        win_w, win_h = WINDOW_SIZE

        found_boxes = []
        found_scores = []

        for scale in self.scales:
            scaled_w = int(roi_w * scale)
            scaled_h = int(roi_h * scale)

            if scaled_w < win_w or scaled_h < win_h:
                continue

            scaled = cv2.resize(roi_frame, (scaled_w, scaled_h))
            batch_features = []
            batch_coords = []

            max_x = scaled_w - win_w
            max_y = scaled_h - win_h

            for y in range(0, max_y + 1, self.window_step):
                for x in range(0, max_x + 1, self.window_step):
                    window = scaled[y:y + win_h, x:x + win_w]
                    feature = extract_hog_feature(window)

                    mapped_x = int(x / scale)
                    mapped_y = int(y / scale)
                    mapped_w = int(win_w / scale)
                    mapped_h = int(win_h / scale)

                    if roi_box_work is not None:
                        rx1, ry1, _, _ = roi_box_work
                        mapped_x += rx1
                        mapped_y += ry1

                    batch_features.append(feature)
                    batch_coords.append((mapped_x, mapped_y, mapped_w, mapped_h))

                    if len(batch_features) >= self.batch_size:
                        self._flush_batch(
                            batch_features,
                            batch_coords,
                            found_boxes,
                            found_scores,
                        )

            self._flush_batch(batch_features, batch_coords, found_boxes, found_scores)

        boxes_after_nms, scores_after_nms = non_max_suppression_with_scores(
            found_boxes,
            found_scores,
            overlap_thresh=self.nms_thresh,
        )

        boxes_after_nms, scores_after_nms = remove_inner_boxes(
            boxes_after_nms,
            scores_after_nms,
            contain_thresh=0.70,
        )

        final_boxes = []
        for x, y, w, h in boxes_after_nms:
            final_boxes.append((
                int(x * scale_back),
                int(y * scale_back),
                int(w * scale_back),
                int(h * scale_back),
            ))

        return final_boxes, scores_after_nms[:len(final_boxes)], roi_box_original
