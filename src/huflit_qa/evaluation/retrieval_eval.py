"""Retrieval evaluation for BM25, dense and hybrid search."""

import time
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from huflit_qa.config import Settings
from huflit_qa.evaluation.metrics import hit_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k
from huflit_qa.evaluation.testset import load_testset, select
from huflit_qa.retrieval.hybrid import HybridRetriever

MODES = ("bm25", "dense", "hybrid")
LABELS = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}
DEFAULT_KS = (5, 10, 20)
# MRR is cut at 10 so numbers stay comparable with the original baseline.
MRR_CUTOFF = 10


def score_run(all_retrieved: list[list[str]], queries: list[dict], ks: Sequence[int]) -> dict[str, float]:
    """Average metrics over queries; all_retrieved[i] is the ranked chunk_id list for queries[i]."""
    per_metric: dict[str, list[float]] = {}

    def add(name: str, value: float) -> None:
        per_metric.setdefault(name, []).append(value)

    for retrieved, query in zip(all_retrieved, queries, strict=True):
        relevant = set(query["relevant_chunk_ids"])
        for k in ks:
            add(f"P@{k}", precision_at_k(retrieved, relevant, k))
            add(f"R@{k}", recall_at_k(retrieved, relevant, k))
            add(f"Hit@{k}", hit_at_k(retrieved, relevant, k))
            add(f"nDCG@{k}", ndcg_at_k(retrieved, relevant, k))
        add(f"MRR@{MRR_CUTOFF}", mrr(retrieved[:MRR_CUTOFF], relevant))

    return {name: float(np.mean(values)) for name, values in per_metric.items()}


def run_evaluation(
    settings: Settings,
    modes: Iterable[str],
    ks: Sequence[int] = DEFAULT_KS,
    split: str | None = None,
) -> dict:
    """Evaluate the requested modes on a split of the test set and return a JSON-serialisable report."""
    queries = select(load_testset(settings.testset_path), split)
    depth = max(ks)
    retriever = HybridRetriever(settings)

    searchers: dict[str, Callable[[str], list[str]]] = {
        "bm25": lambda q: [cid for cid, _ in retriever.search_bm25(q, depth)],
        "dense": lambda q: [cid for cid, _ in retriever.search_dense(q, depth)],
        "hybrid": lambda q: [r["chunk_id"] for r in retriever.retrieve(q, top_k=depth)],
    }

    results = {}
    for mode in modes:
        all_ids, latencies = [], []
        for q in queries:
            start = time.perf_counter()
            all_ids.append(searchers[mode](q["query"]))
            latencies.append((time.perf_counter() - start) * 1000)
        metrics = score_run(all_ids, queries, ks)
        metrics["latency_ms_p50"] = float(np.percentile(latencies, 50))
        metrics["latency_ms_p95"] = float(np.percentile(latencies, 95))
        results[LABELS[mode]] = metrics

    return {
        "split": split or "all",
        "num_queries": len(queries),
        "ks": list(ks),
        "embedding_model": settings.embedding_model,
        "results": results,
    }


def format_report(report: dict, columns: Sequence[str] | None = None) -> str:
    """Markdown table of the main metrics (latency in ms)."""
    columns = columns or ["nDCG@10", "R@10", "R@20", "Hit@10", "P@10", f"MRR@{MRR_CUTOFF}", "latency_ms_p95"]
    header = ["Retriever", *[c.replace("latency_ms_p95", "p95 ms") for c in columns]]
    lines = [
        f"Split: {report['split']} · {report['num_queries']} queries · {report['embedding_model']}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for label, metrics in report["results"].items():
        cells = [f"{metrics[c]:.1f}" if c.startswith("latency") else f"{metrics[c]:.3f}" for c in columns]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
