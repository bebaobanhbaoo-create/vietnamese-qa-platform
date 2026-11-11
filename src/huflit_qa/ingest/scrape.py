"""Step 1: crawl article metadata, then article details, with resume support."""

import logging
import time
from datetime import date
from pathlib import Path

from huflit_qa.config import Settings
from huflit_qa.ingest.classify import build_processed_outputs
from huflit_qa.ingest.crawler import HuflitCrawler
from huflit_qa.ingest.sources import CATEGORIES
from huflit_qa.ingest.storage import load_json, save_json

logger = logging.getLogger(__name__)


def select_categories(category_ids: list[int] | None) -> list[dict]:
    if not category_ids:
        return CATEGORIES
    selected = [c for c in CATEGORIES if c["id"] in category_ids]
    if not selected:
        allowed = [c["id"] for c in CATEGORIES]
        raise ValueError(f"No valid category IDs in {category_ids}; allowed: {allowed}")
    return selected


def plan_details(metas: list[dict], stored: list[dict]) -> list[dict]:
    """Articles whose detail page must be (re)fetched: new ones, failed ones, and ones whose
    'ngày cập nhật' on the list page differs from the stored copy."""
    by_id = {a["id"]: a for a in stored}
    todo = []
    for meta in metas:
        old = by_id.get(meta["id"])
        if (
            old is None
            or old.get("fetch_error")
            or (meta.get("date_updated") and meta.get("date_updated") != old.get("date_updated"))
        ):
            todo.append(meta)
    return todo


def merge_articles(stored: list[dict], fetched: list[dict]) -> list[dict]:
    """Replace stored articles by id with freshly fetched ones; keep order, append new ids."""
    fresh = {a["id"]: a for a in fetched}
    merged = [fresh.pop(a["id"], a) for a in stored]
    return merged + list(fresh.values())


def backfill_from_processed(stored: list[dict], processed_path: Path) -> list[dict]:
    """Add articles that only exist in an older processed file (a legacy crawl kept regulations in
    by_doc_type/quy_che.json but not in raw/articles.json); without this, rebuilding the processed
    files from raw would drop them."""
    known = {a["id"] for a in stored}
    legacy = [doc for doc in load_json(processed_path) if doc["id"] not in known]
    if legacy:
        logger.info("Backfilled %d articles from %s", len(legacy), processed_path.name)
    return stored + legacy


def mark_presence(articles: list[dict], seen_ids: set[int], crawled_categories: set[int], today: str) -> int:
    """Record whether each article is still listed on the portal. Articles that disappear are kept
    (soft delete) with ``on_portal: false`` and the last date they were seen. Returns how many are gone."""
    gone = 0
    for article in articles:
        if article.get("category_id") not in crawled_categories:
            continue
        if article["id"] in seen_ids:
            article["on_portal"], article["last_seen"] = True, today
        else:
            article["on_portal"] = False
            article.setdefault("last_seen", None)
            gone += 1
    return gone


def scrape(
    settings: Settings,
    category_ids: list[int] | None = None,
    resume: bool = True,
    list_only: bool = False,
) -> dict:
    """Crawl list pages, then the detail pages that are new or changed. Returns a summary.

    With ``resume`` the existing ``articles.json`` is the source of truth for what is done; the
    old checkpoint file is no longer needed (re-running without it used to duplicate articles).
    """
    metadata_path = settings.raw_dir / "articles_metadata.json"
    articles_path = settings.raw_dir / "articles.json"

    cats = select_categories(category_ids)
    logger.info("Scraping %d categories", len(cats))
    crawler = HuflitCrawler()

    # Phase 1: metadata from list pages; refresh dates of known articles.
    stored_metas = {m["id"]: m for m in (load_json(metadata_path) if resume else [])}
    seen_ids: set[int] = set()
    for cat in cats:
        for meta in crawler.crawl_category(cat["id"], cat["name"]):
            stored_metas[meta["id"]] = {**stored_metas.get(meta["id"], {}), **meta}
            seen_ids.add(meta["id"])
    listed = len(seen_ids)
    all_metas = list(stored_metas.values())
    save_json(all_metas, metadata_path)
    logger.info("Listed %d articles, %d known in total", listed, len(all_metas))
    if list_only:
        return {"listed": listed, "known": len(all_metas)}

    # Phase 2: detail pages of new or updated articles.
    stored = load_json(articles_path) if resume else []
    if resume:
        stored = backfill_from_processed(stored, settings.by_doc_type_dir / "quy_che.json")
    todo = plan_details(all_metas, stored)
    logger.info("Detail pages to crawl: %d (stored: %d)", len(todo), len(stored))

    start_time = time.time()
    fetched = []
    for index, meta in enumerate(todo, 1):
        if index % 10 == 0 or index == 1:
            elapsed = time.time() - start_time
            rate = index / elapsed if elapsed > 0 else 0
            remaining = (len(todo) - index) / rate if rate > 0 else 0
            logger.info("Progress %d/%d | %.1f items/s | ~%.0fs remaining", index, len(todo), rate, remaining)
        detail = crawler.crawl_detail(meta)
        if detail is None:
            detail = {**meta, "content": "", "attachments": [], "attachment_count": 0, "fetch_error": True}
        fetched.append(detail)
        if index % 25 == 0:
            save_json(merge_articles(stored, fetched), articles_path)

    results = merge_articles(stored, fetched)
    crawled = {c["id"] for c in cats}
    gone = mark_presence(results, seen_ids, crawled, date.today().isoformat())
    save_json(results, articles_path)
    build_processed_outputs(articles_path, settings.processed_dir)

    errors = sum(1 for a in fetched if a.get("fetch_error"))
    stored_ids = {a["id"] for a in stored}
    summary = {
        "listed": listed,
        "fetched": len(fetched),
        "new": sum(1 for a in fetched if a["id"] not in stored_ids),
        "updated": sum(1 for a in fetched if a["id"] in stored_ids),
        "fetch_errors": errors,
        "not_on_portal": gone,
        "total_articles": len(results),
    }
    logger.info("Scrape done in %.1fs: %s", time.time() - start_time, summary)
    return summary
