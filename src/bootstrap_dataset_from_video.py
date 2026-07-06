import argparse
import glob
import os
import random

import cv2

from config import NEGATIVE_DIR, POSITIVE_DIR, TEST_IMAGE_DIR, VIDEO_DIR
from detector import HOGSVMPedestrianDetector


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def ensure_dirs():
    for folder in (POSITIVE_DIR, NEGATIVE_DIR, TEST_IMAGE_DIR):
        os.makedirs(folder, exist_ok=True)


def clear_generated_files():
    patterns = [
        os.path.join(POSITIVE_DIR, "auto_positive_*.jpg"),
        os.path.join(NEGATIVE_DIR, "auto_negative_*.jpg"),
        os.path.join(TEST_IMAGE_DIR, "test_frame_*.jpg"),
    ]

    removed = 0
    for pattern in patterns:
        for path in glob.glob(pattern):
            os.remove(path)
            removed += 1

    return removed


def box_area(box):
    _, _, w, h = box
    return max(0, w) * max(0, h)


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


def expand_and_clip_box(box, width, height, expand_ratio=0.10):
    x, y, w, h = box
    dx = int(w * expand_ratio)
    dy = int(h * expand_ratio)

    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(width, x + w + dx)
    y2 = min(height, y + h + dy)

    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def save_crop(frame, box, path):
    x, y, w, h = box
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return False
    return cv2.imwrite(path, crop)


def sample_negative_box(frame_width, frame_height, person_boxes, rng):
    min_h = max(96, int(frame_height * 0.12))
    max_h = max(min_h + 1, int(frame_height * 0.45))

    for _ in range(80):
        crop_h = rng.randint(min_h, max_h)
        crop_w = max(48, crop_h // 2)

        if crop_w >= frame_width or crop_h >= frame_height:
            continue

        x = rng.randint(0, frame_width - crop_w)
        y = rng.randint(0, frame_height - crop_h)
        candidate = (x, y, crop_w, crop_h)

        if all(box_iou(candidate, person_box) < 0.03 for person_box in person_boxes):
            return candidate

    return None


def bootstrap(args):
    ensure_dirs()

    if args.clear_generated:
        removed = clear_generated_files()
        print(f"Removed generated files: {removed}")

    source = args.source
    if source is None:
        source = os.path.join(VIDEO_DIR, "walking.mp4")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")

    detector = HOGSVMPedestrianDetector(
        hit_threshold=args.hit,
        max_width=args.detector_max_width,
        nms_thresh=args.nms,
        scale=args.scale,
        group_threshold=args.group,
    )

    rng = random.Random(args.seed)

    positive_count = 0
    negative_count = 0
    test_count = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        if frame_index % args.frame_step != 0:
            continue

        h, w = frame.shape[:2]

        if test_count < args.max_test:
            test_path = os.path.join(TEST_IMAGE_DIR, f"test_frame_{test_count:03d}.jpg")
            cv2.imwrite(test_path, frame)
            test_count += 1

        boxes, scores, _ = detector.detect(frame)
        scored_boxes = sorted(
            zip(boxes, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        for box, score in scored_boxes:
            if positive_count >= args.max_positive:
                break

            _, _, bw, bh = box
            if bh < h * 0.10 or bh > h * 0.95:
                continue

            crop_box = expand_and_clip_box(box, w, h, expand_ratio=args.expand)
            out_path = os.path.join(
                POSITIVE_DIR,
                f"auto_positive_{positive_count:04d}_f{frame_index:05d}.jpg",
            )
            if save_crop(frame, crop_box, out_path):
                positive_count += 1

        for _ in range(args.negatives_per_frame):
            if negative_count >= args.max_negative:
                break

            neg_box = sample_negative_box(w, h, boxes, rng)
            if neg_box is None:
                continue

            out_path = os.path.join(
                NEGATIVE_DIR,
                f"auto_negative_{negative_count:04d}_f{frame_index:05d}.jpg",
            )
            if save_crop(frame, neg_box, out_path):
                negative_count += 1

        print(
            f"Frame {frame_index}: positive={positive_count}/{args.max_positive}, "
            f"negative={negative_count}/{args.max_negative}, test={test_count}/{args.max_test}"
        )

        if (
            positive_count >= args.max_positive
            and negative_count >= args.max_negative
            and test_count >= args.max_test
        ):
            break

    cap.release()

    print("\nBootstrap dataset done.")
    print(f"Positive images: {positive_count}")
    print(f"Negative images: {negative_count}")
    print(f"Test images: {test_count}")
    print(f"Source video: {source}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a small positive/negative dataset from a pedestrian video."
    )
    parser.add_argument("--source", default=None, help="Path to video source.")
    parser.add_argument("--max-positive", type=int, default=80)
    parser.add_argument("--max-negative", type=int, default=120)
    parser.add_argument("--max-test", type=int, default=5)
    parser.add_argument("--frame-step", type=int, default=20)
    parser.add_argument("--negatives-per-frame", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear-generated", action="store_true")
    parser.add_argument("--expand", type=float, default=0.10)
    parser.add_argument("--hit", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=1.05)
    parser.add_argument("--group", type=int, default=2)
    parser.add_argument("--nms", type=float, default=0.35)
    parser.add_argument("--detector-max-width", type=int, default=900)
    return parser.parse_args()


if __name__ == "__main__":
    bootstrap(parse_args())
