"""DAG integrity: the file imports, the schedule and the wiring are what the docs say.

Runs in the CI job that installs Airflow; skipped elsewhere (Airflow does not run on Windows).
"""

import os
from pathlib import Path

import pytest

# Airflow 3 reads this instead of the removed DagBag(include_examples=...) argument.
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "false")
pytest.importorskip("airflow")

import itertools

from airflow.models.dagbag import DagBag

DAGS = Path(__file__).resolve().parents[2] / "deploy" / "airflow" / "dags"


@pytest.fixture(scope="module")
def dag():
    bag = DagBag(dag_folder=str(DAGS))
    assert bag.import_errors == {}
    return bag.get_dag("huflit_weekly_ingest")


def test_schedule_and_limits(dag):
    assert dag is not None
    assert dag.timetable.summary == "0 2 * * 1"
    assert dag.catchup is False and dag.max_active_runs == 1


def test_steps_run_in_order(dag):
    order = ["backup", "snapshot", "scrape", "download", "extract_pdf", "process", "quality_gate",
             "build_index", "eval_regression", "publish", "report"]  # fmt: skip
    for upstream, downstream in itertools.pairwise(order):
        assert downstream in dag.get_task(upstream).downstream_task_ids


def test_restore_runs_only_on_failure_and_not_after_publish(dag):
    restore = dag.get_task("restore")
    assert str(restore.trigger_rule) in ("one_failed", "TriggerRule.ONE_FAILED")
    assert "publish" not in restore.upstream_task_ids
    assert {"scrape", "quality_gate", "eval_regression"} <= restore.upstream_task_ids
    assert str(dag.get_task("cleanup").trigger_rule) in ("all_done", "TriggerRule.ALL_DONE")


def test_tasks_call_the_project_cli(dag):
    command = dag.get_task("quality_gate").python_callable()
    assert command.endswith("run-step quality_gate --run-id '{{ run_id }}'")
