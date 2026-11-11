"""
### HUFLIT weekly ingestion

Every Monday at 02:00 (Asia/Ho_Chi_Minh): crawl the student portal for new or updated announcements
and regulations, download regulation attachments, rebuild the records and the index, and publish
the refreshed corpus, unless a check fails.

`backup → snapshot → scrape → download → extract_pdf → process → quality_gate → build_index →
eval_regression → publish → report`, with `restore` rolling `data/` back if any step after the
backup fails, and `cleanup` keeping the last four backups.

* **quality_gate** stops the run on duplicate ids, malformed records, too many fetch errors or empty
  texts, or when a document type shrinks by more than 5% (e.g. a parser change dropping regulations).
* **eval_regression** stops it if hybrid nDCG@10 on the labelled queries falls more than 0.05 below
  the committed baseline.

Each task runs `huflit-qa run-step <step>` in the project's own virtualenv (its pinned ML stack
would conflict with Airflow's dependencies) and writes its summary to `data/.runs/<run_id>/`.
"""

import os
from datetime import timedelta
from itertools import pairwise

import pendulum
from airflow.sdk import dag, task

HUFLIT_BIN = os.environ.get("HUFLIT_BIN", "huflit-qa")


def _cmd(step: str) -> str:
    # The last stdout line (a JSON summary) becomes the task's XCom.
    return f"{HUFLIT_BIN} run-step {step} --run-id '{{{{ run_id }}}}'"


@dag(
    dag_id="huflit_weekly_ingest",
    schedule="0 2 * * 1",
    start_date=pendulum.datetime(2026, 11, 1, tz="Asia/Ho_Chi_Minh"),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=3),
    default_args={"owner": "huflit-qa", "retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["huflit", "ingestion"],
    doc_md=__doc__,
)
def huflit_weekly_ingest():
    @task.bash
    def backup() -> str:
        return _cmd("backup")

    @task.bash
    def snapshot() -> str:
        return _cmd("snapshot")

    # The portal is sometimes slow; the crawler already retries each request, this retries the step.
    @task.bash(retries=3, retry_delay=timedelta(minutes=20), execution_timeout=timedelta(hours=1))
    def scrape() -> str:
        return _cmd("scrape")

    @task.bash(retries=2)
    def download() -> str:
        return _cmd("download")

    @task.bash
    def extract_pdf() -> str:
        return _cmd("extract_pdf")

    @task.bash
    def process() -> str:
        return _cmd("process")

    @task.bash(retries=0)
    def quality_gate() -> str:
        return _cmd("quality_gate")

    @task.bash(execution_timeout=timedelta(minutes=45))
    def build_index() -> str:
        return _cmd("build_index")

    @task.bash(retries=0)
    def eval_regression() -> str:
        return _cmd("eval_regression")

    @task.bash
    def publish() -> str:
        return _cmd("publish")

    @task.bash
    def report() -> str:
        return _cmd("report")

    @task.bash(trigger_rule="one_failed", retries=0)
    def restore() -> str:
        return _cmd("restore")

    @task.bash(trigger_rule="all_done")
    def cleanup() -> str:
        return _cmd("cleanup")

    pipeline = [backup(), snapshot(), scrape(), download(), extract_pdf(), process(), quality_gate(),
                build_index(), eval_regression(), publish(), report()]  # fmt: skip
    for upstream, downstream in pairwise(pipeline):
        upstream >> downstream

    rollback = restore()
    # Any failure from scrape to eval_regression rolls data/ back; a failed upload does not.
    pipeline[2:-2] >> rollback
    [pipeline[-1], rollback] >> cleanup()


huflit_weekly_ingest()
