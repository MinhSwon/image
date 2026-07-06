import cv2
import numpy as np

from config import (
    DEFAULT_HIT_THRESHOLD,
    DEFAULT_WIN_STRIDE,
    DEFAULT_PADDING,
    DEFAULT_SCALE,
    DEFAULT_GROUP_THRESHOLD,
    DEFAULT_MAX_WIDTH,
)


def resize_keep_aspect(frame, max_width=DEFAULT_MAX_WIDTH):
    """
    Resize frame để chạy nhanh hơn nhưng giữ tỉ lệ ảnh.
    Trả về frame_resized và scale_back để map bbox về ảnh gốc.
    """
    h, w = frame.shape[:2]

    if max_width is None or max_width <= 0 or w <= max_width:
        return frame.copy(), 1.0

    scale = max_width / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))
    scale_back = w / float(new_w)

    return resized, scale_back


def parse_roi_pct(roi_pct, width, height):
    """
    ROI theo phần trăm: x1 y1 x2 y2.
    Ví dụ: 0.1 0.1 0.9 0.98.
    """
    x1p, y1p, x2p, y2p = roi_pct

    x1 = int(x1p * width)
    y1 = int(y1p * height)
    x2 = int(x2p * width)
    y2 = int(y2p * height)

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI không hợp lệ.")

    return x1, y1, x2, y2


def non_max_suppression_indices(boxes, scores, overlap_thresh=0.35):
    """
    NMS loại bỏ các bbox trùng nhau.
    boxes: list[(x, y, w, h)]
    scores: list[float]
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0, xx2 - xx1 + 1)
        inter_h = np.maximum(0, yy2 - yy1 + 1)
        inter = inter_w * inter_h

        overlap = inter / areas[order[1:]]

        inds = np.where(overlap <= overlap_thresh)[0]
        order = order[inds + 1]

    return [int(i) for i in keep]


def non_max_suppression(boxes, scores, overlap_thresh=0.35):
    keep = non_max_suppression_indices(boxes, scores, overlap_thresh)
    return [tuple(map(int, boxes[i])) for i in keep]


def non_max_suppression_with_scores(boxes, scores, overlap_thresh=0.35):
    keep = non_max_suppression_indices(boxes, scores, overlap_thresh)
    return [tuple(map(int, boxes[i])) for i in keep], [scores[i] for i in keep]

def remove_inner_boxes(boxes, scores=None, contain_thresh=0.75):
    """
    Loại bỏ box nhỏ nằm phần lớn bên trong box lớn hơn.
    Nếu một box nhỏ bị chứa trong box lớn hơn, giữ box lớn.
    """
    if len(boxes) == 0:
        return [], []

    if scores is None:
        scores = [0.0] * len(boxes)

    keep = []

    for i, box_i in enumerate(boxes):
        xi, yi, wi, hi = box_i
        area_i = wi * hi
        remove_i = False

        for j, box_j in enumerate(boxes):
            if i == j:
                continue

            xj, yj, wj, hj = box_j
            area_j = wj * hj

            # Chỉ xét trường hợp box_i nhỏ hơn box_j
            if area_i >= area_j:
                continue

            ix1 = max(xi, xj)
            iy1 = max(yi, yj)
            ix2 = min(xi + wi, xj + wj)
            iy2 = min(yi + hi, yj + hj)

            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter_area = inter_w * inter_h

            # Nếu phần lớn box nhỏ nằm trong box lớn thì bỏ box nhỏ
            contained_ratio = inter_area / float(area_i + 1e-6)

            if contained_ratio >= contain_thresh:
                remove_i = True
                break

        if not remove_i:
            keep.append(i)

    new_boxes = [boxes[i] for i in keep]
    new_scores = [scores[i] for i in keep]

    return new_boxes, new_scores

class HOGSVMPedestrianDetector:
    """
    Detector HOG + SVM cho người đi bộ.

    OpenCV HOGDescriptor_getDefaultPeopleDetector() là bộ SVM đã huấn luyện sẵn.
    HOG window mặc định là 64x128, phù hợp với người đứng/đi bộ toàn thân.
    """

    def __init__(
        self,
        hit_threshold=DEFAULT_HIT_THRESHOLD,
        win_stride=DEFAULT_WIN_STRIDE,
        padding=DEFAULT_PADDING,
        scale=DEFAULT_SCALE,
        group_threshold=DEFAULT_GROUP_THRESHOLD,
        max_width=DEFAULT_MAX_WIDTH,
        roi_pct=None,
        nms_thresh=0.35,
    ):
        self.hit_threshold = hit_threshold
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.group_threshold = group_threshold
        self.max_width = max_width
        self.roi_pct = roi_pct
        self.nms_thresh = nms_thresh

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        """
        Trả về:
        - boxes: bbox đã map về frame gốc
        - scores: điểm tin cậy
        - roi_box: vùng ROI trên frame gốc hoặc None
        """
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

        # OpenCV Python có phiên bản không nhận keyword finalThreshold,
        # nên truyền tham số theo vị trí.
        try:
            boxes, weights = self.hog.detectMultiScale(
                roi_frame,
                self.hit_threshold,
                self.win_stride,
                self.padding,
                self.scale,
                self.group_threshold,
                False,
            )
        except TypeError:
            boxes, weights = self.hog.detectMultiScale(
                roi_frame,
                self.hit_threshold,
                self.win_stride,
                self.padding,
                self.scale,
                self.group_threshold,
            )

        weights = np.array(weights).reshape(-1)

        filtered_boxes = []
        filtered_scores = []

        roi_h, roi_w = roi_frame.shape[:2]

        for i, (x, y, w, h) in enumerate(boxes):
            score = float(weights[i]) if i < len(weights) else 0.0
            aspect_ratio = h / float(w)

            # Người đi bộ thường có bbox cao hơn rộng.
            if aspect_ratio < 1.3 or aspect_ratio > 4.8:
                continue

            # Bỏ bbox quá nhỏ so với frame.
            if h < roi_h * 0.12:
                continue

            # Bỏ bbox gần như chiếm toàn ảnh.
            if h > roi_h * 0.98:
                continue

            # Nếu có ROI, cộng offset để đưa về work_frame.
            if roi_box_work is not None:
                rx1, ry1, _, _ = roi_box_work
                x += rx1
                y += ry1

            filtered_boxes.append((x, y, w, h))
            filtered_scores.append(score)

        boxes_after_nms, scores_after_nms = non_max_suppression_with_scores(
            filtered_boxes,
            filtered_scores,
            overlap_thresh=self.nms_thresh,
        )

        # Loại box nhỏ nằm trong box lớn
        boxes_after_nms, scores_after_nms = remove_inner_boxes(
            boxes_after_nms,
            scores_after_nms,
            contain_thresh=0.70,
        )

        # Map về frame gốc nếu đã resize.
        final_boxes = []
        for b in boxes_after_nms:
            x, y, w, h = b
            final_boxes.append((
                int(x * scale_back),
                int(y * scale_back),
                int(w * scale_back),
                int(h * scale_back),
            ))

        # Score dùng để hiển thị gần đúng.
        final_scores = scores_after_nms[:len(final_boxes)]

        return final_boxes, final_scores, roi_box_original


def draw_detections(frame, boxes, scores=None, roi_box=None, fps=None):
    """
    Vẽ bbox, nhãn person và FPS lên frame.
    """
    output = frame.copy()

    if roi_box is not None:
        x1, y1, x2, y2 = roi_box
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            output,
            "ROI",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    for idx, (x, y, w, h) in enumerate(boxes):
        label = "person"

        if scores is not None and idx < len(scores):
            label = f"person {scores[idx]:.2f}"

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(
            output,
            label,
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

    if fps is not None:
        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        output,
        "HOG + SVM Pedestrian Detection",
        (20, output.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    return output
