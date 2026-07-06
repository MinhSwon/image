import argparse
import os
import sys
import time
import cv2

from config import CUSTOM_MODEL_PATH, OUTPUT_DIR
from detector import HOGSVMPedestrianDetector, draw_detections

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_source(source):
    """
    Nếu source là số dạng '0', '1' thì dùng webcam.
    Nếu không, coi là đường dẫn video.
    """
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def build_video_writer(record_path, fps, width, height):
    folder = os.path.dirname(record_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(record_path, fourcc, fps, (width, height))

def box_area(box):
    x, y, w, h = box
    return max(0, w) * max(0, h)


def box_center(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih
    union = box_area(a) + box_area(b) - inter + 1e-6

    return inter / union


def center_distance(a, b):
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


class BoxTracker:
    """
    Tracker đơn giản để ổn định bbox qua nhiều frame.

    Công dụng:
    - Nếu một người bị mất detect tạm thời, giữ bbox cũ thêm vài frame.
    - Nếu bbox đột ngột bị thu nhỏ, hạn chế việc box co lại quá nhanh.
    - Làm mượt tọa độ bbox để video ít nhấp nháy hơn.
    """

    def __init__(
        self,
        max_missed=8,
        iou_threshold=0.08,
        smooth_alpha=0.65,
        min_area_ratio=0.65,
    ):
        self.tracks = []
        self.next_id = 1
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.smooth_alpha = smooth_alpha
        self.min_area_ratio = min_area_ratio

    def _match_track(self, det_box, used_tracks):
        best_idx = None
        best_score = -1

        for i, track in enumerate(self.tracks):
            if i in used_tracks:
                continue

            old_box = track["box"]

            iou = box_iou(old_box, det_box)
            dist = center_distance(old_box, det_box)

            ox, oy, ow, oh = old_box
            max_allowed_dist = max(80, max(ow, oh) * 0.75)

            # Chấp nhận match nếu overlap đủ hoặc tâm box không quá xa
            if iou >= self.iou_threshold or dist <= max_allowed_dist:
                score = iou - dist / 1000.0

                if score > best_score:
                    best_score = score
                    best_idx = i

        return best_idx

    def update(self, boxes, scores):
        used_tracks = set()
        used_detections = set()

        # Match detection mới với track cũ
        for det_idx, det_box in enumerate(boxes):
            track_idx = self._match_track(det_box, used_tracks)

            if track_idx is None:
                continue

            track = self.tracks[track_idx]
            old_box = track["box"]

            # Nếu box mới nhỏ bất thường so với box cũ,
            # giữ kích thước cũ nhưng cập nhật tâm theo box mới.
            old_area = box_area(old_box)
            new_area = box_area(det_box)

            if old_area > 0 and new_area < old_area * self.min_area_ratio:
                dcx, dcy = box_center(det_box)
                _, _, old_w, old_h = old_box
                det_box = (
                    int(dcx - old_w / 2),
                    int(dcy - old_h / 2),
                    int(old_w),
                    int(old_h),
                )

            # Làm mượt bbox
            alpha = self.smooth_alpha
            smoothed_box = tuple(
                int(alpha * det_box[i] + (1 - alpha) * old_box[i])
                for i in range(4)
            )

            track["box"] = smoothed_box
            track["score"] = scores[det_idx] if det_idx < len(scores) else track["score"]
            track["missed"] = 0

            used_tracks.add(track_idx)
            used_detections.add(det_idx)

        # Detection chưa match thì tạo track mới
        for det_idx, det_box in enumerate(boxes):
            if det_idx in used_detections:
                continue

            self.tracks.append({
                "id": self.next_id,
                "box": det_box,
                "score": scores[det_idx] if det_idx < len(scores) else 0.0,
                "missed": 0,
            })

            self.next_id += 1

        # Track không được update thì tăng missed
        for i, track in enumerate(self.tracks):
            if i not in used_tracks:
                track["missed"] += 1

        # Xóa track mất quá lâu
        self.tracks = [
            t for t in self.tracks
            if t["missed"] <= self.max_missed
        ]

        return self.get_boxes()

    def get_boxes(self):
        boxes = [t["box"] for t in self.tracks]
        scores = [t["score"] for t in self.tracks]
        return boxes, scores

def parse_scale_list(value):
    if value is None:
        return None

    scales = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        scales.append(float(item))

    if not scales:
        raise ValueError("Danh sach scale custom khong hop le.")

    return scales


def build_detector(args):
    if args.model == "opencv":
        return HOGSVMPedestrianDetector(
            hit_threshold=args.hit,
            max_width=args.max_width,
            roi_pct=args.roi_pct,
            nms_thresh=args.nms,
            scale=args.scale,
            group_threshold=args.group,
        )

    from custom_detector import CustomHOGSVMSlidingWindowDetector

    return CustomHOGSVMSlidingWindowDetector(
        model_path=args.custom_model,
        max_width=args.custom_max_width,
        roi_pct=args.roi_pct,
        window_step=args.custom_step,
        scales=parse_scale_list(args.custom_scales),
        decision_threshold=args.custom_threshold,
        nms_thresh=args.nms,
    )


def run(args):
    if args.source.isdigit():
        source_idx = int(args.source)
        # Ép dùng DirectShow để tránh lỗi MSMF (-1072875772) trên Windows Webcam
        cap = cv2.VideoCapture(source_idx, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được source: {args.source}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if input_fps <= 1 or input_fps != input_fps:
        input_fps = 25

    print("Đang chạy realtime HOG + SVM...")
    print(f"Source: {args.source}")
    print(f"FPS video: {input_fps}")
    print(f"Tổng số frame: {total_frames}")
    print("Phím q: thoát | s: lưu frame | p: pause/resume")

    writer = None
    paused = False
    frame_index = 0
    output = None

    detector = build_detector(args)

    last_boxes = []
    last_scores = []
    last_roi = None

    track_max_missed = args.track_max_missed
    if track_max_missed is None:
        track_max_missed = args.persist

    tracker = BoxTracker(
        max_missed=track_max_missed,
        smooth_alpha=args.smooth_alpha,
    )

    prev_time = time.time()
    fps = 0.0

    # Delay để phát gần đúng tốc độ video gốc
    delay = max(1, int(1000 / input_fps))

    while True:
        if not paused:
            ret, frame = cap.read()

            if not ret:
                print("Đã đọc hết video.")
                break

            frame_index += 1

            if frame_index % 30 == 0:
                print(f"Đang xử lý frame {frame_index}/{total_frames}")

            if frame_index % max(1, args.detect_every) == 0:
                boxes, scores, roi = detector.detect(frame)
                last_boxes, last_scores = tracker.update(boxes, scores)
                last_roi = roi
            else:
                last_boxes, last_scores = tracker.get_boxes()

            now = time.time()
            elapsed = now - prev_time
            prev_time = now

            if elapsed > 0:
                fps = 1.0 / elapsed

            output = draw_detections(
                frame,
                last_boxes,
                scores=last_scores,
                roi_box=last_roi,
                fps=fps,
            )

            if writer is None and args.record:
                h, w = output.shape[:2]
                writer = build_video_writer(args.record, input_fps, w, h)

            if writer is not None:
                writer.write(output)

            if not args.no_display:
                cv2.imshow("HOG + SVM Realtime Pedestrian Detection", output)

            if args.max_frames > 0 and frame_index >= args.max_frames:
                print(f"Reached max frames: {args.max_frames}")
                break

        if args.no_display:
            key = 255
        else:
            key = cv2.waitKey(delay) & 0xFF

        if key == ord("q"):
            print("Đã thoát bằng phím q.")
            break

        if key == ord("p"):
            paused = not paused

        if key == ord("s") and output is not None:
            save_path = os.path.join(OUTPUT_DIR, f"snapshot_{int(time.time())}.jpg")
            cv2.imwrite(save_path, output)
            print(f"Đã lưu snapshot: {save_path}")

    cap.release()

    if writer is not None:
        writer.release()
        print(f"Đã lưu video kết quả: {args.record}")

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Realtime HOG + SVM pedestrian detection on Webcam/Video"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="0 cho webcam mặc định, hoặc đường dẫn video. Ví dụ: --source 0 hoặc --source dataset/videos/walking.mp4",
    )

    parser.add_argument(
        "--record",
        default=None,
        help="Đường dẫn lưu video kết quả, ví dụ outputs/result.mp4",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Khong mo cua so OpenCV, dung khi chi muon ghi video ket qua.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Gioi han so frame can xu ly. Dat 0 de xu ly den het video.",
    )

    parser.add_argument(
        "--model",
        choices=["opencv", "custom"],
        default="opencv",
        help="Chon detector: opencv la HOG+SVM pretrained, custom la model tu train.",
    )

    parser.add_argument(
        "--custom-model",
        default=CUSTOM_MODEL_PATH,
        help="Duong dan file custom_hog_svm.pkl.",
    )

    parser.add_argument(
        "--custom-threshold",
        type=float,
        default=0.0,
        help="Nguong decision_function cho custom SVM. Tang len de giam false positive.",
    )

    parser.add_argument(
        "--custom-step",
        type=int,
        default=32,
        help="Buoc truot sliding window cho custom SVM.",
    )

    parser.add_argument(
        "--custom-scales",
        default="1.0,0.85,0.70,0.55,0.45,0.35",
        help="Danh sach scale custom, cach nhau bang dau phay.",
    )

    parser.add_argument(
        "--custom-max-width",
        type=int,
        default=640,
        help="Resize frame rieng cho custom SVM de chay nhanh hon.",
    )

    parser.add_argument(
        "--hit",
        type=float,
        default=0.0,
        help="hitThreshold của HOG SVM. Thấp hơn dễ phát hiện hơn nhưng dễ nhiễu.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.05,
        help="Tỉ lệ image pyramid trong detectMultiScale.",
    )

    parser.add_argument(
        "--group",
        type=int,
        default=2,
        help="groupThreshold để gom bbox.",
    )

    parser.add_argument(
        "--nms",
        type=float,
        default=0.35,
        help="Ngưỡng NMS.",
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=900,
        help="Resize frame về max width để chạy nhanh hơn. Đặt 0 để không resize.",
    )

    parser.add_argument(
        "--detect-every",
        type=int,
        default=2,
        help="Số frame mới detect một lần. Tăng lên để chạy nhanh hơn.",
    )

    parser.add_argument(
        "--roi-pct",
        nargs=4,
        type=float,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="ROI theo phần trăm. Ví dụ: --roi-pct 0.10 0.10 0.90 0.98",
    )
    
    parser.add_argument(
        "--persist",
        type=int,
        default=5,
        help="Giữ bbox cũ thêm vài frame nếu detector tạm thời mất dấu.",
    )
    
    parser.add_argument(
        "--track-max-missed",
        type=int,
        default=None,
        help="Số lần detect liên tiếp được phép mất một người trước khi xóa bbox.",
    )

    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.65,
        help="Hệ số làm mượt bbox. Cao hơn thì box bám detection mới nhanh hơn.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
