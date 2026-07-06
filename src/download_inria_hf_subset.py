import argparse
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2

from config import NEGATIVE_DIR, POSITIVE_DIR


HF_REPO = "marcelarosalesj/inria-person"
HF_API_BASE = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main"
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"


def list_hf_files(subdir):
    url = f"{HF_API_BASE}/{subdir}"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)

    files = [
        item["path"]
        for item in data
        if item.get("type") == "file"
        and item.get("path", "").lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    return sorted(files)


def download_file(remote_path, local_path, retries=3):
    encoded_path = "/".join(urllib.parse.quote(part) for part in remote_path.split("/"))
    url = f"{HF_RESOLVE_BASE}/{encoded_path}"

    local_path.parent.mkdir(parents=True, exist_ok=True)

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                local_path.write_bytes(response.read())
            return True
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            wait = 1.5 * attempt
            print(f"Retry {attempt}/{retries} for {remote_path}: {error}")
            time.sleep(wait)

    print(f"Failed to download {remote_path}: {last_error}")
    return False


def clear_inria_files():
    removed = 0
    for folder, prefix in [(POSITIVE_DIR, "inria_positive_"), (NEGATIVE_DIR, "inria_negative_")]:
        for path in Path(folder).glob(f"{prefix}*"):
            if path.is_file():
                path.unlink()
                removed += 1
    return removed


def clear_auto_video_files():
    removed = 0
    for folder, prefix in [(POSITIVE_DIR, "auto_positive_"), (NEGATIVE_DIR, "auto_negative_")]:
        for path in Path(folder).glob(f"{prefix}*"):
            if path.is_file():
                path.unlink()
                removed += 1
    return removed


def sample_negative_patch(image, rng):
    height, width = image.shape[:2]

    if width < 64 or height < 128:
        return None

    min_crop_h = min(height, 128)
    max_crop_h = height

    for _ in range(100):
        crop_h = rng.randint(min_crop_h, max_crop_h)
        crop_w = int(crop_h * 0.5)

        if crop_w < 64 or crop_w > width:
            continue

        x = rng.randint(0, width - crop_w)
        y = rng.randint(0, height - crop_h)
        patch = image[y:y + crop_h, x:x + crop_w]

        if patch.size == 0:
            continue

        return patch

    return None


def import_positive(remote_files, raw_dir, rng, max_positive):
    selected = remote_files[:]
    rng.shuffle(selected)
    selected = selected[:max_positive]

    saved = 0
    for remote_path in selected:
        raw_path = raw_dir / remote_path
        if not raw_path.exists():
            ok = download_file(remote_path, raw_path)
            if not ok:
                continue

        image = cv2.imread(str(raw_path))
        if image is None:
            print(f"Skip unreadable positive: {raw_path}")
            continue

        out_path = Path(POSITIVE_DIR) / f"inria_positive_{saved:04d}.jpg"
        cv2.imwrite(str(out_path), image)
        saved += 1
        print(f"Positive {saved}/{max_positive}: {out_path.name}")

    return saved


def import_negative(remote_files, raw_dir, rng, max_negative, patches_per_image):
    selected = remote_files[:]
    rng.shuffle(selected)

    saved = 0
    for remote_path in selected:
        if saved >= max_negative:
            break

        raw_path = raw_dir / remote_path
        if not raw_path.exists():
            ok = download_file(remote_path, raw_path)
            if not ok:
                continue

        image = cv2.imread(str(raw_path))
        if image is None:
            print(f"Skip unreadable negative source: {raw_path}")
            continue

        for _ in range(patches_per_image):
            if saved >= max_negative:
                break

            patch = sample_negative_patch(image, rng)
            if patch is None:
                continue

            out_path = Path(NEGATIVE_DIR) / f"inria_negative_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), patch)
            saved += 1
            print(f"Negative {saved}/{max_negative}: {out_path.name}")

    return saved


def main(args):
    Path(POSITIVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(NEGATIVE_DIR).mkdir(parents=True, exist_ok=True)

    if args.clear_inria:
        print(f"Removed old INRIA files: {clear_inria_files()}")

    if args.clear_video_bootstrap:
        print(f"Removed video-bootstrap files: {clear_auto_video_files()}")

    raw_dir = Path(args.raw_dir)
    rng = random.Random(args.seed)

    print("Listing Hugging Face INRIA mirror files...")
    positive_files = list_hf_files("data_ped/pedestrians")
    negative_files = list_hf_files("data_ped/no_pedestrians")
    print(f"Available positive files: {len(positive_files)}")
    print(f"Available negative source files: {len(negative_files)}")

    pos_count = import_positive(positive_files, raw_dir, rng, args.max_positive)
    neg_count = import_negative(
        negative_files,
        raw_dir,
        rng,
        args.max_negative,
        args.negative_patches_per_image,
    )

    print("\nINRIA subset import done.")
    print(f"Positive images saved: {pos_count}")
    print(f"Negative patches saved: {neg_count}")
    print(f"Raw cache: {raw_dir}")
    print("Source: https://huggingface.co/datasets/marcelarosalesj/inria-person")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a lightweight INRIA Person subset from Hugging Face."
    )
    parser.add_argument("--max-positive", type=int, default=300)
    parser.add_argument("--max-negative", type=int, default=300)
    parser.add_argument("--negative-patches-per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear-inria", action="store_true")
    parser.add_argument("--clear-video-bootstrap", action="store_true")
    parser.add_argument(
        "--raw-dir",
        default=os.path.join("dataset", "external_raw", "inria_hf"),
        help="Raw image cache. Kept out of training folders.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
