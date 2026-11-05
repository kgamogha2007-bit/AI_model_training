"""Extract candidate umpire-signal frames from video into a staging folder.

Each video is labeled with a single class at the video level (e.g. a "six
signals compilation" video labeled `six`), so frames land directly under
data/scraped/<class>/ -- the same staging root scrape_images.py uses. This is
NOT a detector: it samples frames at a fixed interval and drops obviously bad
ones (blurry, redundant/near-static repeats), but you still need to manually
review and move keepers into data/train/<class> or data/val/<class>.

Sources can be local video files or YouTube URLs (downloaded via yt-dlp into
data/scraped_frames/_downloads/). Only pass in videos you have the right to
use for this -- this script never searches for or auto-selects videos itself.

sources.txt format (one per line, comma-separated):
    six,https://www.youtube.com/watch?v=XXXXXXXX
    out,C:\\videos\\out_compilation.mp4
    wide,C:\\videos\\wide_signals.mp4

Usage:
    python extract_frames.py --sources sources.txt
    python extract_frames.py --sources sources.txt --interval-sec 0.5 --blur-threshold 100
"""

import argparse
import os

import cv2
import numpy as np
import yt_dlp

import dedup

STAGING_ROOT = os.path.join("data", "scraped")
DOWNLOAD_DIR = os.path.join("data", "scraped_frames", "_downloads")

VALID_CLASSES = {"out", "six", "wide"}


def is_url(source):
    return source.startswith("http://") or source.startswith("https://")


def download_video(url, download_dir):
    os.makedirs(download_dir, exist_ok=True)
    ydl_opts = {
        "outtmpl": os.path.join(download_dir, "%(id)s.%(ext)s"),
        "format": "mp4/best",
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def laplacian_variance(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def frame_diff_score(frame_a, frame_b):
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(gray_a, gray_b)))


def extract_from_video(video_path, out_dir, interval_sec, blur_threshold, min_diff):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  could not open {video_path}, skipping")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(fps * interval_sec))

    frame_idx = 0
    saved = 0
    last_kept_gray = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval == 0:
            if laplacian_variance(frame) >= blur_threshold:
                is_redundant = False
                if last_kept_gray is not None:
                    diff = frame_diff_score(frame, last_kept_gray)
                    is_redundant = diff < min_diff

                if not is_redundant:
                    out_path = os.path.join(out_dir, f"frame_{saved:05d}.jpg")
                    cv2.imwrite(out_path, frame)
                    last_kept_gray = frame
                    saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", required=True, help="Path to a sources.txt file (see module docstring for format).")
    parser.add_argument("--interval-sec", type=float, default=0.5, help="Seconds between sampled frames.")
    parser.add_argument("--blur-threshold", type=float, default=100.0, help="Min Laplacian variance to keep a frame.")
    parser.add_argument("--min-diff", type=float, default=8.0, help="Min mean pixel diff vs last kept frame to avoid near-static skip.")
    parser.add_argument("--no-dedup", action="store_true", help="Skip the near-duplicate cleanup pass after extraction.")
    args = parser.parse_args()

    with open(args.sources, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    touched_classes = set()

    for line in lines:
        cls, source = [part.strip() for part in line.split(",", 1)]
        if cls not in VALID_CLASSES:
            print(f"skipping line, unknown class '{cls}': {line}")
            continue

        print(f"\n=== {source} (class '{cls}') ===")
        video_path = download_video(source, DOWNLOAD_DIR) if is_url(source) else source
        if not os.path.isfile(video_path):
            print(f"  file not found: {video_path}, skipping")
            continue

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(STAGING_ROOT, cls, f"frames_{video_name}")
        saved = extract_from_video(video_path, out_dir, args.interval_sec, args.blur_threshold, args.min_diff)
        print(f"  saved {saved} candidate frame(s) to {out_dir}")
        touched_classes.add(cls)

    if not args.no_dedup:
        for cls in touched_classes:
            class_dir = os.path.join(STAGING_ROOT, cls)
            test_dir_name = cls.capitalize()
            compare_against = [
                os.path.join("data", "train", cls),
                os.path.join("data", "val", cls),
                os.path.join("data", "test", test_dir_name),
            ]
            kept, removed = dedup.dedup_folder(class_dir, compare_against=compare_against)
            print(f"'{cls}': kept {kept}, removed {removed} near-duplicate frame(s)")

    print(f"\nDone. Review frames under {STAGING_ROOT}/<class>/frames_*/ and move keepers into data/train or data/val.")


if __name__ == "__main__":
    main()
