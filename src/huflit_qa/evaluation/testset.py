"""Retrieval test set stored as JSONL, one query per line.

Fields: ``id``, ``query``, ``relevant_chunk_ids``, ``labelled`` (judged by a human), ``split``
(dev/test), ``style`` (keyword/question) and ``source`` (who wrote it). Labelled queries with no
relevant chunk have no answer in the corpus: they are kept for answer-refusal tests but skipped by
retrieval metrics, as are queries not labelled yet.
"""

import hashlib
import json
from pathlib import Path

DEV_RATIO = 0.4


def assign_split(query_id: str, dev_ratio: float = DEV_RATIO) -> str:
    """Deterministic split from a hash of the id, so adding queries never reshuffles old ones."""
    bucket = int(hashlib.sha1(query_id.encode("utf-8")).hexdigest(), 16) % 100
    return "dev" if bucket < dev_ratio * 100 else "test"


def load_testset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_testset(queries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def select(queries: list[dict], split: str | None = None) -> list[dict]:
    """Labelled queries with at least one relevant chunk, from a split (``None`` = all)."""
    return [
        q
        for q in queries
        if q.get("labelled") and q["relevant_chunk_ids"] and (split is None or q["split"] == split)
    ]
