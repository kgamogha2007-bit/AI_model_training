"""Near-duplicate image removal via a simple average-hash (aHash).

Used as a shared helper by scrape_images.py and extract_frames.py, and can
also be run standalone:

    python dedup.py --dir data/scraped/six
    python dedup.py --dir data/scraped/six --compare-against data/train/six data/val/six
"""

import argparse
import os

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
HASH_SIZE = 8
DEFAULT_THRESHOLD = 5  # max Hamming distance (out of 64 bits) to count as a duplicate


def _iter_images(folder):
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                yield os.path.join(root, name)


def average_hash(path):
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((HASH_SIZE, HASH_SIZE), Image.LANCZOS)
            pixels = np.asarray(img, dtype=np.float64)
    except (UnidentifiedImageError, OSError):
        return None
    return pixels > pixels.mean()


def hamming_distance(hash_a, hash_b):
    return int(np.count_nonzero(hash_a != hash_b))


def dedup_folder(folder, compare_against=None, threshold=DEFAULT_THRESHOLD, dry_run=False):
    """Remove near-duplicate images inside `folder`, keeping the first-seen copy.

    `compare_against` is an optional list of other folders whose images are
    hashed too (but never deleted) so candidates already present in curated
    splits get dropped instead of re-added.

    Returns (kept_count, removed_count).
    """
    kept_hashes = []

    for other in compare_against or []:
        if not os.path.isdir(other):
            continue
        for path in _iter_images(other):
            h = average_hash(path)
            if h is not None:
                kept_hashes.append(h)

    removed = 0
    kept = 0
    for path in sorted(_iter_images(folder)):
        h = average_hash(path)
        if h is None:
            continue  # unreadable/corrupt download, leave it for manual review

        is_dup = any(hamming_distance(h, k) <= threshold for k in kept_hashes)
        if is_dup:
            removed += 1
            if not dry_run:
                os.remove(path)
        else:
            kept += 1
            kept_hashes.append(h)

    return kept, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Folder to dedup (searched recursively).")
    parser.add_argument(
        "--compare-against",
        nargs="*",
        default=[],
        help="Extra folders (e.g. data/train/six data/val/six) to hash for comparison without deleting from them.",
    )
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true", help="Report what would be removed without deleting.")
    args = parser.parse_args()

    kept, removed = dedup_folder(
        args.dir, compare_against=args.compare_against, threshold=args.threshold, dry_run=args.dry_run
    )
    action = "would remove" if args.dry_run else "removed"
    print(f"{args.dir}: kept {kept}, {action} {removed} near-duplicate(s)")


if __name__ == "__main__":
    main()
