import argparse
import os
import sys
import time

import cv2

from config import POSITIVE_DIR, NEGATIVE_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def get_output_dir(label):
    if label == "positive":
        return POSITIVE_DIR
    if label == "negative":
        return NEGATIVE_DIR
    raise ValueError("label phải là positive hoặc negative")


def run(args):
    out_dir = get_output_dir(args.label)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được webcam/source: {args.source}")

    count = len([
        f for f in os.listdir(out_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"Đang thu dataset: {args.label}")
    print("Phím s: lưu frame/crop | q: thoát")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        display = frame.copy()

        h, w = frame.shape[:2]

        if args.center_crop:
            crop_w = int(w * args.crop_width_ratio)
            crop_h = int(crop_w * 2)  # tỷ lệ gần 64x128

            crop_w = min(crop_w, w)
            crop_h = min(crop_h, h)

            x1 = w // 2 - crop_w // 2
            y1 = h // 2 - crop_h // 2
            x2 = x1 + crop_w
            y2 = y1 + crop_h

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, "Crop area", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            crop = frame[y1:y2, x1:x2].copy()
        else:
            crop = frame.copy()

        cv2.putText(
            display,
            f"Label: {args.label} | Saved: {count} | s=save, q=quit",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Collect Dataset", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            filename = f"{args.label}_{count:05d}_{int(time.time())}.jpg"
            path = os.path.join(out_dir, filename)
            cv2.imwrite(path, crop)
            count += 1
            print(f"Đã lưu: {path}")

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Collect positive/negative images from webcam")

    parser.add_argument(
        "--label",
        required=True,
        choices=["positive", "negative"],
        help="Loại dữ liệu cần thu.",
    )

    parser.add_argument(
        "--source",
        type=int,
        default=0,
        help="ID webcam, mặc định là 0.",
    )

    parser.add_argument(
        "--center-crop",
        action="store_true",
        help="Chỉ lưu vùng crop ở giữa frame.",
    )

    parser.add_argument(
        "--crop-width-ratio",
        type=float,
        default=0.25,
        help="Tỉ lệ chiều rộng crop so với frame.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
