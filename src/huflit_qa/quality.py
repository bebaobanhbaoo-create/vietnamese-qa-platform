"""Data quality checks between pipeline steps.

``snapshot`` summarises the corpus at one point in time; ``check`` compares a new snapshot with the
previous one and returns the problems found. Errors stop the weekly pipeline before a bad crawl
reaches the index (for example a parser change that silently drops every regulation); warnings
are reported but do not block.
"""

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from huflit_qa.config import Settings

RE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REQUIRED_ARTICLE_FIELDS = ("id", "title", "url", "category_id")


@dataclass
class Thresholds:
    # A doc type may shrink by at most this share between runs (pages are rarely deleted).
    max_drop: float = 0.05
    max_fetch_error_rate: float = 0.10
    max_empty_record_rate: float = 0.25
    # Warn when nothing new was posted for this many days.
    stale_after_days: int = 60


@dataclass
class Report:
    snapshot: dict
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok}


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def snapshot(settings: Settings) -> dict:
    """Counts and ids that later checks compare against."""
    articles = _load(settings.raw_dir / "articles.json")
    quy_che = _load(settings.quy_che_flat_path)
    thong_bao = _load(settings.thong_bao_flat_path)
    records = quy_che + thong_bao
    dates = sorted(a["date_posted"] for a in articles + quy_che if a.get("date_posted"))
    return {
        "articles": len(articles),
        "article_ids": sorted({a["id"] for a in articles}),
        "duplicate_article_ids": sorted(k for k, n in Counter(a["id"] for a in articles).items() if n > 1),
        "fetch_errors": sum(1 for a in articles if a.get("fetch_error")),
        "records_by_type": {"quy_che": len(quy_che), "thong_bao": len(thong_bao)},
        "records_by_status": dict(Counter(r.get("content_status") for r in records)),
        "missing_fields": sum(
            1 for a in articles if any(a.get(f) in (None, "") for f in REQUIRED_ARTICLE_FIELDS)
        ),
        "bad_dates": sum(1 for a in articles if a.get("date_posted") and not RE_DATE.match(a["date_posted"])),
        "latest_post": dates[-1] if dates else None,
        "not_on_portal": sum(1 for a in articles if a.get("on_portal") is False),
    }


def check(new: dict, previous: dict | None = None, thresholds: Thresholds | None = None,
          today: date | None = None) -> Report:  # fmt: skip
    t = thresholds or Thresholds()
    report = Report(snapshot={k: v for k, v in new.items() if k != "article_ids"})

    if new["duplicate_article_ids"]:
        report.errors.append(f"duplicate article ids: {new['duplicate_article_ids'][:10]}")
    if new["missing_fields"]:
        report.errors.append(f"{new['missing_fields']} articles miss a required field")
    if new["bad_dates"]:
        report.errors.append(f"{new['bad_dates']} articles have a malformed date_posted")
    if new["articles"] and new["fetch_errors"] / new["articles"] > t.max_fetch_error_rate:
        report.errors.append(
            f"fetch errors {new['fetch_errors']}/{new['articles']} above {t.max_fetch_error_rate:.0%}"
        )

    total_records = sum(new["records_by_type"].values())
    empty = new["records_by_status"].get("empty", 0)
    if total_records and empty / total_records > t.max_empty_record_rate:
        report.errors.append(f"{empty}/{total_records} records have no content")
    if total_records == 0:
        report.errors.append("no records")

    if previous:
        for doc_type, before in previous["records_by_type"].items():
            after = new["records_by_type"].get(doc_type, 0)
            if before and (before - after) / before > t.max_drop:
                report.errors.append(f"{doc_type} records dropped from {before} to {after}")
        lost = sorted(set(previous.get("article_ids", [])) - set(new["article_ids"]))
        if lost:
            report.warnings.append(f"{len(lost)} articles disappeared: {lost[:10]}")
        added = len(set(new["article_ids"]) - set(previous.get("article_ids", [])))
        report.snapshot["new_articles"] = added

    if new.get("not_on_portal"):
        report.warnings.append(f"{new['not_on_portal']} articles are no longer listed on the portal (kept)")
    if new["latest_post"]:
        age = ((today or date.today()) - date.fromisoformat(new["latest_post"])).days
        if age > t.stale_after_days:
            report.warnings.append(f"newest article is {age} days old ({new['latest_post']})")
    return report
