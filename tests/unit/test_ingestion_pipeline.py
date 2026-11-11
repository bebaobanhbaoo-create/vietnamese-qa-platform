import json
from datetime import date

import pytest

from huflit_qa.config import Settings
from huflit_qa.ingest.download import download_attachments, download_url, folder_name
from huflit_qa.ingest.scrape import merge_articles, plan_details
from huflit_qa.orchestration import DataQualityError, run_step
from huflit_qa.quality import Thresholds, check

# --- scraper resume ---------------------------------------------------------------------------


def test_plan_details_fetches_new_failed_and_updated_articles():
    stored = [
        {"id": 1, "date_updated": "2026-01-01"},
        {"id": 2, "date_updated": "2026-01-01", "fetch_error": True},
        {"id": 3, "date_updated": "2026-01-01"},
    ]
    metas = [
        {"id": 1, "date_updated": "2026-01-01"},  # unchanged
        {"id": 2, "date_updated": "2026-01-01"},  # failed last time
        {"id": 3, "date_updated": "2026-02-01"},  # updated on the portal
        {"id": 4, "date_updated": None},  # new
    ]
    assert [m["id"] for m in plan_details(metas, stored)] == [2, 3, 4]


def test_merge_replaces_by_id_without_duplicates():
    stored = [{"id": 1, "v": "old"}, {"id": 2, "v": "old"}]
    merged = merge_articles(stored, [{"id": 2, "v": "new"}, {"id": 3, "v": "new"}])
    assert merged == [{"id": 1, "v": "old"}, {"id": 2, "v": "new"}, {"id": 3, "v": "new"}]


# --- attachment download ----------------------------------------------------------------------


def test_download_urls_for_share_links():
    drive = "https://drive.google.com/file/d/1KiZy32VBtmJcY4j/view?usp=drive_link"
    assert (
        download_url(drive, "google_drive")
        == "https://drive.google.com/uc?export=download&id=1KiZy32VBtmJcY4j"
    )
    sp = download_url("https://x.sharepoint.com/:b:/g/personal/dt/EWHc?e=Iabc", "sharepoint")
    assert sp is not None and "download=1" in sp and "e=Iabc" in sp
    assert download_url("https://forms.office.com/r/abc", "ms_form") is None
    assert folder_name(3209, 'Quy định "chuẩn" đầu ra: 2024') == "3209_Quy_định_chuẩn_đầu_ra_2024"


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


class FakeSession:
    headers: dict = {}  # noqa: RUF012

    def __init__(self, bodies):
        self.bodies = bodies

    def get(self, url, timeout, allow_redirects):
        return FakeResponse(self.bodies[url])


def test_download_saves_files_skips_existing_and_rejects_login_pages(tmp_path):
    base = tmp_path / "quy_ch"
    existing = base / "10_Old"
    existing.mkdir(parents=True)
    (existing / "10_1__da_co.txt").write_text("x", encoding="utf-8")
    docs = [
        {"id": 10, "title": "Old", "attachments": [
            {"type": "google_drive", "url": "https://drive.google.com/file/d/A/view", "text": "a"},
        ]},
        {"id": 20, "title": "Quy chế mới", "attachments": [
            {"type": "google_drive", "url": "https://drive.google.com/file/d/B/view", "text": "Quy chế 2026"},
            {"type": "sharepoint", "url": "https://x.sharepoint.com/:b:/g/p/C?e=1", "text": "Phụ lục"},
            {"type": "ms_form", "url": "https://forms.office.com/r/x", "text": "Khảo sát"},
        ]},
    ]  # fmt: skip
    source = tmp_path / "quy_che.json"
    source.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    session = FakeSession(
        {
            "https://drive.google.com/uc?export=download&id=B": b"%PDF-1.7 ...",
            "https://x.sharepoint.com/:b:/g/p/C?e=1&download=1": b"<!DOCTYPE html><html>Sign in</html>",
        }
    )
    stats = download_attachments(source, base, session)  # type: ignore[arg-type]
    assert stats.skipped_existing == 1 and stats.unsupported == 1
    assert stats.downloaded == ["20_1__quy_che_2026.pdf"]
    assert (base / "20_Quy_chế_mới" / "20_1__quy_che_2026.pdf").exists()
    assert stats.failed == [{"doc_id": 20, "index": 2, "reason": "got html, not a file"}]
    assert not list(base.rglob("*.bin"))


# --- quality gate -----------------------------------------------------------------------------


def _snapshot(**overrides):
    snap = {
        "articles": 100,
        "article_ids": list(range(100)),
        "duplicate_article_ids": [],
        "fetch_errors": 0,
        "records_by_type": {"quy_che": 24, "thong_bao": 100},
        "records_by_status": {"ok": 120, "empty": 4},
        "missing_fields": 0,
        "bad_dates": 0,
        "latest_post": "2026-11-01",
    }
    return {**snap, **overrides}


def test_quality_passes_and_counts_new_articles():
    report = check(_snapshot(article_ids=list(range(103))), _snapshot(), today=date(2026, 11, 10))
    assert report.ok and report.snapshot["new_articles"] == 3 and not report.warnings


def test_quality_blocks_data_loss_and_bad_records():
    new = _snapshot(
        records_by_type={"quy_che": 0, "thong_bao": 100}, duplicate_article_ids=[7], fetch_errors=20
    )
    report = check(new, _snapshot(), today=date(2026, 11, 10))
    assert not report.ok
    assert any("quy_che records dropped from 24 to 0" in e for e in report.errors)
    assert any("duplicate" in e for e in report.errors) and any("fetch errors" in e for e in report.errors)


def test_quality_warns_on_stale_corpus_and_lost_ids():
    report = check(_snapshot(article_ids=list(range(99))), _snapshot(), Thresholds(), today=date(2027, 3, 1))
    assert report.ok
    assert any("days old" in w for w in report.warnings) and any("disappeared" in w for w in report.warnings)


# --- orchestration ----------------------------------------------------------------------------


def _corpus(root, quy_che_records):
    by_type = root / "data" / "processed" / "by_doc_type"
    by_type.mkdir(parents=True)
    (root / "data" / "raw").mkdir(parents=True)
    articles = [
        {"id": i, "title": "t", "url": "u", "category_id": 1, "date_posted": "2026-11-01"} for i in range(5)
    ]
    (root / "data" / "raw" / "articles.json").write_text(json.dumps(articles), encoding="utf-8")
    quy_che = [{"id": f"q{i}", "content_status": "ok"} for i in range(quy_che_records)]
    (by_type / "quy_che_flat.json").write_text(json.dumps(quy_che), encoding="utf-8")
    (by_type / "thong_bao_flat.json").write_text(
        json.dumps([{"id": "t", "content_status": "ok"}]), encoding="utf-8"
    )


def test_run_steps_persist_summaries_and_restore_after_a_bad_crawl(tmp_path):
    _corpus(tmp_path, quy_che_records=10)
    settings = Settings(root_dir=tmp_path)
    run_id = "scheduled__2026-11-09T19:00:00+00:00"

    run_step(settings, "backup", run_id)
    assert run_step(settings, "snapshot", run_id)["records_by_type"]["quy_che"] == 10

    # A crawl that loses every regulation must be stopped and rolled back.
    settings.quy_che_flat_path.write_text("[]", encoding="utf-8")
    with pytest.raises(DataQualityError, match="quy_che records dropped"):
        run_step(settings, "quality_gate", run_id)
    run_step(settings, "restore", run_id)
    assert len(json.loads(settings.quy_che_flat_path.read_text(encoding="utf-8"))) == 10

    report = run_step(settings, "report", run_id)
    saved = json.loads((tmp_path / report["report"]).read_text(encoding="utf-8"))
    assert saved["backup"]["backup"].endswith("scheduled__2026-11-09T19-00-00+00-00")
    assert saved["restore"]["restored"] == ["raw", "processed"]

    with pytest.raises(ValueError, match="Unknown step"):
        run_step(settings, "deploy_everything", run_id)


def test_backfill_and_soft_delete(tmp_path):
    from huflit_qa.ingest.scrape import backfill_from_processed, mark_presence

    processed = tmp_path / "quy_che.json"
    processed.write_text(json.dumps([{"id": 3208, "category_id": 1016}, {"id": 3278, "category_id": 1016}]))
    articles = backfill_from_processed([{"id": 3278, "category_id": 1016}], processed)
    assert sorted(a["id"] for a in articles) == [3208, 3278]

    articles.append({"id": 9, "category_id": 1})  # category not crawled this run: left alone
    gone = mark_presence(articles, seen_ids={3278}, crawled_categories={1016}, today="2026-11-11")
    by_id = {a["id"]: a for a in articles}
    assert gone == 1
    assert by_id[3208]["on_portal"] is False and by_id[3208]["last_seen"] is None
    assert by_id[3278] == {"id": 3278, "category_id": 1016, "on_portal": True, "last_seen": "2026-11-11"}
    assert "on_portal" not in by_id[9]
