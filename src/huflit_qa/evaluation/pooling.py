"""Candidate pooling for relevance annotation.

Labelling only what the current system returns would bias the test set towards it, so candidates are
pooled from the top results of several retrievers (TREC-style). Labels come back from the annotation
page (eval/annotation/annotate.html) as JSON and are merged into the test set.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from huflit_qa.evaluation.projection import build_projection, project_ranking

if TYPE_CHECKING:
    from huflit_qa.retrieval.hybrid import HybridRetriever

Searcher = Callable[[str, int], list[str]]


def unit_rankings(
    retriever: "HybridRetriever", queries: list[dict], units: list[dict], depth: int
) -> dict[str, dict[str, list[str]]]:
    """mode -> query text -> label units covered by the top-``depth`` chunks, in rank order.

    Chunks are projected onto label units because units are what annotators judge.
    """
    projection = build_projection(list(retriever.chunk_map.values()), units)
    out: dict[str, dict[str, list[str]]] = {"bm25": {}, "dense": {}, "hybrid": {}}
    for q in queries:
        text = q["query"]
        out["bm25"][text] = [cid for cid, _ in retriever.search_bm25(text, depth)]
        out["dense"][text] = [cid for cid, _ in retriever.search_dense(text, depth)]
        out["hybrid"][text] = [r["chunk_id"] for r in retriever.retrieve(text, top_k=depth)]
    return {
        mode: {text: project_ranking(ids, projection) for text, ids in by_query.items()}
        for mode, by_query in out.items()
    }


def precomputed_searcher(by_query: dict[str, list[str]]) -> Searcher:
    def search(query: str, n: int) -> list[str]:
        return by_query[query][:n]

    return search


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
