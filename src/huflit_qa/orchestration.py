"""Steps of the weekly ingestion pipeline, independent of the scheduler.

The Airflow DAG (deploy/airflow/dags/huflit_weekly_ingest.py) only wires these together, so every step is
testable without Airflow. Steps take the settings and return small JSON-serialisable summaries
(passed between tasks as XComs).

    backup -> snapshot -> scrape -> download -> extract_pdf -> process -> quality_gate
           -> build_index -> eval_regression -> publish -> report      (restore on failure)
"""

import json
import logging
import os
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from huflit_qa.config import Settings
from huflit_qa.quality import check, snapshot

logger = logging.getLogger(__name__)

BACKED_UP = ("raw", "processed", "index")


class DataQualityError(RuntimeError):
    pass


class RegressionError(RuntimeError):
    pass


def backup_dir(settings: Settings, run_id: str) -> Path:
    return settings.data_dir / ".backups" / run_id.replace(":", "-")


def step_backup(settings: Settings, run_id: str) -> dict:
    """Copy raw/, processed/ and index/ aside so a failed run can be rolled back."""
    target = backup_dir(settings, run_id)
    for name in BACKED_UP:
        src = settings.data_dir / name
        if src.exists():
            shutil.copytree(src, target / name, dirs_exist_ok=True)
    return {"backup": str(target)}


def step_restore(settings: Settings, run_id: str) -> dict:
    source = backup_dir(settings, run_id)
    restored = []
    for name in BACKED_UP:
        if (source / name).exists():
            shutil.rmtree(settings.data_dir / name, ignore_errors=True)
            shutil.copytree(source / name, settings.data_dir / name)
            restored.append(name)
    logger.warning("Restored %s from %s", restored, source)
    return {"restored": restored}


def step_cleanup_backups(settings: Settings, keep: int = 4) -> dict:
    root = settings.data_dir / ".backups"
    runs = sorted(root.iterdir()) if root.exists() else []
    for old in runs[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    return {"kept": min(len(runs), keep)}


def step_snapshot(settings: Settings) -> dict:
    return snapshot(settings)


def step_scrape(settings: Settings) -> dict:
    from huflit_qa.ingest.scrape import scrape

    return scrape(settings, resume=True)


def step_download(settings: Settings) -> dict:
    from huflit_qa.ingest.download import download_attachments

    stats = download_attachments(settings.quy_che_path, settings.quy_che_attachments_dir)
    return {**asdict(stats), "downloaded": len(stats.downloaded), "failed": len(stats.failed),
            "failures": stats.failed[:20]}  # fmt: skip


def step_extract_pdf(settings: Settings) -> dict:
    from huflit_qa.ingest.pdf import extract_missing_txt

    return {"txt_written": extract_missing_txt(settings.attachments_dir)}


def step_process(settings: Settings) -> dict:
    from huflit_qa.pipeline import process

    process(settings)
    return {
        "quy_che": len(json.loads(settings.quy_che_flat_path.read_text(encoding="utf-8"))),
        "thong_bao": len(json.loads(settings.thong_bao_flat_path.read_text(encoding="utf-8"))),
    }


def step_quality_gate(settings: Settings, previous: dict) -> dict:
    report = check(snapshot(settings), previous)
    for warning in report.warnings:
        logger.warning("quality: %s", warning)
    if not report.ok:
        raise DataQualityError("; ".join(report.errors))
    return report.to_dict()


def step_build_index(settings: Settings) -> dict:
    from huflit_qa.pipeline import build_index

    vectors = build_index(settings)  # Qdrant sync counts: embedded / payload_updated / deleted
    manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    return {
        "num_chunks": manifest["num_chunks"],
        "source_digest": manifest["source_digest"],
        "vectors": vectors,
    }


def step_eval_regression(settings: Settings, baseline_path: Path, max_drop: float = 0.05) -> dict:
    """Fail if hybrid nDCG@10 on the labelled queries fell more than ``max_drop`` below the baseline."""
    from huflit_qa.evaluation.retrieval_eval import run_evaluation

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["results"]["Hybrid"]["nDCG@10"]
    report = run_evaluation(settings, ("hybrid",))
    current = report["results"]["Hybrid"]["nDCG@10"]
    result = {"baseline_ndcg10": baseline, "ndcg10": current, "num_queries": report["num_queries"]}
    if baseline - current > max_drop:
        raise RegressionError(f"hybrid nDCG@10 {current:.3f} is more than {max_drop} below {baseline:.3f}")
    return result


def step_publish(settings: Settings, repo_id: str) -> dict:
    """Upload the refreshed corpus to the private dataset, if a Hugging Face token is configured."""
    if not os.environ.get("HF_TOKEN"):
        return {"published": False, "reason": "HF_TOKEN not set"}
    from huflit_qa.hub import upload_data

    card = settings.root_dir / "docs" / "DATA_CARD.md"
    return {"published": True, "url": upload_data(settings.data_dir, repo_id, private=True, card=card)}


def step_report(settings: Settings, run_id: str, results: dict) -> dict:
    out = settings.data_dir / "reports" / f"ingest_{run_id.replace(':', '-')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "finished_at": datetime.now(UTC).isoformat(timespec="seconds"), **results}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"report": str(out)}


# Entry point used by the scheduler: one step per process, results kept on disk per run.

DATASET_REPO = "bebaobanhbaoo/huflit-regulations"
STEPS = (
    "backup", "snapshot", "scrape", "download", "extract_pdf", "process", "quality_gate",
    "build_index", "eval_regression", "publish", "report", "restore", "cleanup",
)  # fmt: skip


def run_dir(settings: Settings, run_id: str) -> Path:
    return settings.data_dir / ".runs" / run_id.replace(":", "-")


def run_step(settings: Settings, step: str, run_id: str) -> dict:
    """Run one step and store its summary as data/.runs/<run_id>/<step>.json."""
    folder = run_dir(settings, run_id)
    folder.mkdir(parents=True, exist_ok=True)

    def previous(name: str) -> dict:
        path = folder / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    actions = {
        "backup": lambda: step_backup(settings, run_id),
        "snapshot": lambda: step_snapshot(settings),
        "scrape": lambda: step_scrape(settings),
        "download": lambda: step_download(settings),
        "extract_pdf": lambda: step_extract_pdf(settings),
        "process": lambda: step_process(settings),
        "quality_gate": lambda: step_quality_gate(settings, previous("snapshot")),
        "build_index": lambda: step_build_index(settings),
        "eval_regression": lambda: step_eval_regression(
            settings, settings.eval_dir / "results" / "baseline.json"
        ),
        "publish": lambda: step_publish(settings, DATASET_REPO),
        "report": lambda: step_report(
            settings, run_id, {name: previous(name) for name in STEPS if name not in ("snapshot", "report")}
        ),
        "restore": lambda: step_restore(settings, run_id),
        "cleanup": lambda: step_cleanup_backups(settings),
    }
    if step not in actions:
        raise ValueError(f"Unknown step {step!r}; expected one of {STEPS}")
    result = actions[step]()
    (folder / f"{step}.json").write_text(
        json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return result
