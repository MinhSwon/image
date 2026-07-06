"""
File này chỉ tạo video test kỹ thuật để kiểm tra chương trình có mở/ghi video được không.
Video này KHÔNG dùng để đánh giá phát hiện người vì không có người thật.
"""

import os
import sys
import cv2
import numpy as np

from config import VIDEO_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.makedirs(VIDEO_DIR, exist_ok=True)

out_path = os.path.join(VIDEO_DIR, "demo_background.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, 20, (640, 480))

for i in range(120):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)

    x = 50 + i * 4
    cv2.rectangle(frame, (x % 640, 180), ((x + 80) % 640, 330), (255, 255, 255), 2)

    cv2.putText(
        frame,
        "Demo technical video - use real pedestrian video for report",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    writer.write(frame)

writer.release()
print(f"Đã tạo video test kỹ thuật: {out_path}")
