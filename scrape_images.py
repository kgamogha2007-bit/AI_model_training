"""Scrape candidate umpire-signal images from Bing Image Search into a staging folder.

This does NOT write into data/train or data/val directly -- results land in
data/scraped/<class>/ for manual review. Cull the bad ones (wrong sport, memes,
stock-photo watermarks, duplicates missed by dedup) and move the keepers into
data/train/<class> or data/val/<class> yourself.

Usage:
    python scrape_images.py
    python scrape_images.py --classes six wide --max-per-query 40
"""

import argparse
import os
import re
import shutil

from icrawler.builtin import BingImageCrawler

import dedup

STAGING_ROOT = os.path.join("data", "scraped")

QUERIES = {
    "out": [
        "cricket umpire out signal",
        "cricket umpire raising finger out",
        "cricket umpire given out gesture",
    ],
    "six": [
        "cricket umpire six signal",
        "cricket umpire arms raised six",
        "umpire signaling six runs cricket",
    ],
    "wide": [
        "cricket umpire wide signal",
        "cricket umpire wide ball arms out",
        "umpire signaling wide cricket",
    ],
}


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def scrape_class(cls, queries, max_per_query):
    class_dir = os.path.join(STAGING_ROOT, cls)
    os.makedirs(class_dir, exist_ok=True)

    for query in queries:
        query_dir = os.path.join(class_dir, f"bing_{_slug(query)}")
        os.makedirs(query_dir, exist_ok=True)
        crawler = BingImageCrawler(storage={"root_dir": query_dir})
        crawler.crawl(keyword=query, max_num=max_per_query, file_idx_offset=0)

    # Flatten the per-query subfolders into the class folder so dedup and
    # later manual review only have to deal with one directory per class.
    for query in queries:
        query_dir = os.path.join(class_dir, f"bing_{_slug(query)}")
        if not os.path.isdir(query_dir):
            continue
        for name in os.listdir(query_dir):
            src = os.path.join(query_dir, name)
            dst = os.path.join(class_dir, f"{_slug(query)}_{name}")
            if os.path.isfile(src):
                shutil.move(src, dst)
        shutil.rmtree(query_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", nargs="+", default=list(QUERIES.keys()), choices=list(QUERIES.keys()))
    parser.add_argument("--max-per-query", type=int, default=60)
    parser.add_argument("--no-dedup", action="store_true", help="Skip the near-duplicate cleanup pass.")
    args = parser.parse_args()

    for cls in args.classes:
        print(f"\n=== Scraping class '{cls}' ===")
        scrape_class(cls, QUERIES[cls], args.max_per_query)

        class_dir = os.path.join(STAGING_ROOT, cls)
        if not args.no_dedup:
            test_dir_name = cls.capitalize()  # data/test uses Out/Six/Wide
            compare_against = [
                os.path.join("data", "train", cls),
                os.path.join("data", "val", cls),
                os.path.join("data", "test", test_dir_name),
            ]
            kept, removed = dedup.dedup_folder(class_dir, compare_against=compare_against)
            print(f"'{cls}': kept {kept} candidate image(s), removed {removed} near-duplicate(s)")
        else:
            count = sum(1 for _ in dedup._iter_images(class_dir))
            print(f"'{cls}': {count} candidate image(s) downloaded (dedup skipped)")

    print(f"\nDone. Review images under {STAGING_ROOT}/<class>/ and move keepers into data/train or data/val.")


if __name__ == "__main__":
    main()
