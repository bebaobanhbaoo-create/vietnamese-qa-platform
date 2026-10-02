#!/usr/bin/env python3
# MỤC ĐÍCH: Điều phối toàn bộ quá trình scraping (Bước 1).
#           Chạy 2 phase tuần tự: crawl metadata → crawl chi tiết.
#           Hỗ trợ resume (tiếp tục từ chỗ dừng) và --list-only.
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from classify import build_processed_outputs
from config import CATEGORIES
from crawler import HuflitCrawler
from storage import load_checkpoint, load_json, save_checkpoint, save_json


DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

METADATA_PATH = RAW_DIR / "articles_metadata.json"
ARTICLES_PATH = RAW_DIR / "articles.json"
CHECKPOINT_PATH = RAW_DIR / ".checkpoint.json"


def parse_args():
    parser = argparse.ArgumentParser(description="HUFLIT Portal Scraper")
    parser.add_argument("--categories", nargs="*", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.categories:
        cats = []
        for c in CATEGORIES:
            if c["id"] in args.categories:
                cats.append(c)
        if not cats:
            print("No valid category IDs:", args.categories)
            allowed_ids = []
            for c in CATEGORIES:
                allowed_ids.append(c["id"])
            print("Allowed IDs:", allowed_ids)
            sys.exit(1)
    else:
        cats = CATEGORIES

    print("=== HUFLIT Scraper start ===")
    print("Categories:", len(cats))

    crawler = HuflitCrawler()

    # Phase 1: Crawl metadata
    print("--- Phase 1: Crawl metadata ---")
    all_metas = [] if args.no_resume else load_json(METADATA_PATH)
    seen_ids = set()
    for a in all_metas:
        seen_ids.add(a["id"])

    for cat in cats:
        for meta in crawler.crawl_category(cat["id"], cat["name"]):
            if meta["id"] not in seen_ids:
                all_metas.append(meta)
                seen_ids.add(meta["id"])

    save_json(all_metas, METADATA_PATH)
    print("Total metadata:", len(all_metas))

    if args.list_only:
        print("--list-only: done after phase 1.")
        return

    # Phase 2: Crawl article detail
    print("--- Phase 2: Crawl article detail ---")
    done_ids = set() if args.no_resume else load_checkpoint(CHECKPOINT_PATH)
    results = [] if args.no_resume else load_json(ARTICLES_PATH)

    todo = []
    for meta in all_metas:
        if meta["id"] not in done_ids:
            todo.append(meta)
    print("Need detail crawl:", len(todo), "(already done:", len(done_ids), ")")

    start_time = time.time()
    for index, meta in enumerate(todo, 1):
        if index % 10 == 0 or index == 1:
            elapsed = time.time() - start_time
            rate = index / elapsed if elapsed > 0 else 0
            remaining = (len(todo) - index) / rate if rate > 0 else 0
            print(f"Progress: {index}/{len(todo)} | {rate:.1f} items/s | ~{remaining:.0f}s remaining")

        detail = crawler.crawl_detail(meta)
        if detail:
            results.append(detail)
        else:
            fallback = dict(meta)
            fallback["content"] = ""
            fallback["attachments"] = []
            fallback["attachment_count"] = 0
            fallback["fetch_error"] = True
            results.append(fallback)

        done_ids.add(meta["id"])

        if index % 25 == 0:
            save_json(results, ARTICLES_PATH)
            save_checkpoint(done_ids, CHECKPOINT_PATH)

    save_json(results, ARTICLES_PATH)
    save_checkpoint(done_ids, CHECKPOINT_PATH)

    build_processed_outputs(ARTICLES_PATH, PROCESSED_DIR)

    errors = 0
    for a in results:
        if a.get("fetch_error"):
            errors += 1
    print(f"=== DONE === {len(results)} articles | {errors} errors | {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
