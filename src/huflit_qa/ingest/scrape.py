"""Step 1: crawl article metadata, then article details, with resume support."""

import logging
import time

from huflit_qa.config import Settings
from huflit_qa.ingest.classify import build_processed_outputs
from huflit_qa.ingest.crawler import HuflitCrawler
from huflit_qa.ingest.sources import CATEGORIES
from huflit_qa.ingest.storage import load_checkpoint, load_json, save_checkpoint, save_json

logger = logging.getLogger(__name__)


def select_categories(category_ids: list[int] | None) -> list[dict]:
    if not category_ids:
        return CATEGORIES
    selected = [c for c in CATEGORIES if c["id"] in category_ids]
    if not selected:
        allowed = [c["id"] for c in CATEGORIES]
        raise ValueError(f"No valid category IDs in {category_ids}; allowed: {allowed}")
    return selected


def scrape(
    settings: Settings,
    category_ids: list[int] | None = None,
    resume: bool = True,
    list_only: bool = False,
) -> None:
    metadata_path = settings.raw_dir / "articles_metadata.json"
    articles_path = settings.raw_dir / "articles.json"
    checkpoint_path = settings.raw_dir / ".checkpoint.json"

    cats = select_categories(category_ids)
    logger.info("Scraping %d categories", len(cats))
    crawler = HuflitCrawler()

    # Phase 1: metadata from list pages
    all_metas = load_json(metadata_path) if resume else []
    seen_ids = {a["id"] for a in all_metas}
    for cat in cats:
        for meta in crawler.crawl_category(cat["id"], cat["name"]):
            if meta["id"] not in seen_ids:
                all_metas.append(meta)
                seen_ids.add(meta["id"])
    save_json(all_metas, metadata_path)
    logger.info("Total metadata: %d", len(all_metas))

    if list_only:
        return

    # Phase 2: article detail pages
    done_ids = load_checkpoint(checkpoint_path) if resume else set()
    results = load_json(articles_path) if resume else []
    todo = [m for m in all_metas if m["id"] not in done_ids]
    logger.info("Detail pages to crawl: %d (already done: %d)", len(todo), len(done_ids))

    start_time = time.time()
    for index, meta in enumerate(todo, 1):
        if index % 10 == 0 or index == 1:
            elapsed = time.time() - start_time
            rate = index / elapsed if elapsed > 0 else 0
            remaining = (len(todo) - index) / rate if rate > 0 else 0
            logger.info("Progress %d/%d | %.1f items/s | ~%.0fs remaining", index, len(todo), rate, remaining)

        detail = crawler.crawl_detail(meta)
        if detail is None:
            detail = {**meta, "content": "", "attachments": [], "attachment_count": 0, "fetch_error": True}
        results.append(detail)
        done_ids.add(meta["id"])

        if index % 25 == 0:
            save_json(results, articles_path)
            save_checkpoint(done_ids, checkpoint_path)

    save_json(results, articles_path)
    save_checkpoint(done_ids, checkpoint_path)
    build_processed_outputs(articles_path, settings.processed_dir)

    errors = sum(1 for a in results if a.get("fetch_error"))
    logger.info("Done: %d articles | %d errors | %.1fs", len(results), errors, time.time() - start_time)
