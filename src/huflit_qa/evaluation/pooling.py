"""Candidate pooling for relevance annotation.

Labelling only what the current system returns would bias the test set towards it, so candidates are
pooled from the top results of several retrievers (TREC-style). Labels come back from the annotation
page (eval/annotation/annotate.html) as JSON and are merged into the test set.
"""

import json
from collections.abc import Callable
from pathlib import Path

Searcher = Callable[[str, int], list[str]]


def build_pool(
    queries: list[dict],
    searchers: dict[str, Searcher],
    chunk_map: dict[str, dict],
    depth: int = 10,
) -> list[dict]:
    """For each query, the union of the top-``depth`` chunks of every searcher plus existing labels."""
    pool = []
    for q in queries:
        ranks: dict[str, dict[str, int]] = {}
        for name, search in searchers.items():
            for rank, chunk_id in enumerate(search(q["query"], depth), 1):
                ranks.setdefault(chunk_id, {})[name] = rank
        for chunk_id in q["relevant_chunk_ids"]:
            ranks.setdefault(chunk_id, {})

        candidates = []
        for chunk_id, by_system in ranks.items():
            chunk = chunk_map[chunk_id]
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "doc_title": chunk["doc_title"],
                    "chunk_title": chunk.get("chunk_title"),
                    "date_posted": chunk.get("date_posted"),
                    "source_url": chunk["source_url"],
                    "text": chunk["text"],
                    "ranks": by_system,
                    "labelled_relevant": chunk_id in q["relevant_chunk_ids"],
                }
            )
        # Best rank first; chunks found only through existing labels go last.
        candidates.sort(key=lambda c: min(c["ranks"].values(), default=depth + 1))
        pool.append({"id": q["id"], "query": q["query"], "candidates": candidates})
    return pool


def merge_labels(queries: list[dict], labels: list[dict]) -> int:
    """Apply exported labels ``[{id, relevant_chunk_ids}]`` in place. Returns the number updated."""
    by_id = {q["id"]: q for q in queries}
    updated = 0
    for label in labels:
        query = by_id.get(label["id"])
        if query is None:
            raise KeyError(f"Unknown query id in labels: {label['id']}")
        query["relevant_chunk_ids"] = sorted(set(label["relevant_chunk_ids"]))
        query["labelled"] = True
        updated += 1
    return updated


def save_pool(pool: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")
