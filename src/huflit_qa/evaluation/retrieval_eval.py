"""Retrieval evaluation for BM25, dense and hybrid search."""

import time
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from huflit_qa.config import Settings
from huflit_qa.evaluation.metrics import (
    hit_from_hits,
    mrr_from_hits,
    ndcg_from_hits,
    precision_from_hits,
    recall_from_hits,
)
from huflit_qa.evaluation.projection import build_projection, load_units
from huflit_qa.evaluation.testset import load_testset, select
from huflit_qa.retrieval.hybrid import HybridRetriever

MODES = ("bm25", "dense", "hybrid")
LABELS = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}
DEFAULT_KS = (5, 10, 20)
# MRR is cut at 10 so numbers stay comparable with the original baseline.
MRR_CUTOFF = 10


def per_query_scores(
    all_retrieved: list[list[str]],
    queries: list[dict],
    ks: Sequence[int],
    projection: dict[str, list[str]] | None = None,
) -> dict[str, list[float]]:
    """Metric name -> one value per query; all_retrieved[i] is the ranked chunk_id list for queries[i].

    ``projection`` maps chunk ids to the label units they cover; without it, chunk ids are
    assumed to be label units.
    """
    per_metric: dict[str, list[float]] = {}

    def add(name: str, value: float) -> None:
        per_metric.setdefault(name, []).append(value)

    for retrieved, query in zip(all_retrieved, queries, strict=True):
        relevant = set(query["relevant_chunk_ids"])
        if projection is None:
            hits = [{cid} & relevant for cid in retrieved]
        else:
            hits = [set(projection.get(cid, ())) & relevant for cid in retrieved]
        for k in ks:
            add(f"P@{k}", precision_from_hits(hits, k))
            add(f"R@{k}", recall_from_hits(hits, len(relevant), k))
            add(f"Hit@{k}", hit_from_hits(hits, k))
            add(f"nDCG@{k}", ndcg_from_hits(hits, len(relevant), k))
        add(f"MRR@{MRR_CUTOFF}", mrr_from_hits(hits[:MRR_CUTOFF]))
    return per_metric


def score_run(
    all_retrieved: list[list[str]],
    queries: list[dict],
    ks: Sequence[int],
    projection: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    """Metrics averaged over queries."""
    per_metric = per_query_scores(all_retrieved, queries, ks, projection)
    return {name: float(np.mean(values)) for name, values in per_metric.items()}


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Mean difference b - a over the same queries, with a 95% bootstrap CI and the share of
    resamples where b does not beat a (a one-sided p-value estimate)."""
    diff = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_resamples, len(diff)))].mean(axis=1)
    return {
        "diff": float(diff.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p_not_better": float((means <= 0).mean()),
    }


def run_evaluation(
    settings: Settings,
    modes: Iterable[str],
    ks: Sequence[int] = DEFAULT_KS,
    split: str | None = None,
    retriever: HybridRetriever | None = None,
) -> dict:
    """Evaluate the requested modes on a split of the test set and return a JSON-serialisable report."""
    queries = select(load_testset(settings.testset_path), split)
    depth = max(ks)
    retriever = retriever or HybridRetriever(settings)
    projection = build_projection(list(retriever.chunk_map.values()), load_units(settings.label_units_path))
    words = {cid: len(c["text"].split()) for cid, c in retriever.chunk_map.items()}

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
        per_query = per_query_scores(all_ids, queries, ks, projection)
        metrics: dict = {name: float(np.mean(values)) for name, values in per_query.items()}
        metrics["per_query_nDCG@10"] = dict(
            zip([q["id"] for q in queries], per_query["nDCG@10"], strict=True)
        )
        metrics["words_per_chunk@10"] = float(np.mean([words[c] for ids in all_ids for c in ids[:10]]))
        metrics["latency_ms_p50"] = float(np.percentile(latencies, 50))
        metrics["latency_ms_p95"] = float(np.percentile(latencies, 95))
        results[LABELS[mode]] = metrics

    return {
        "split": split or "all",
        "num_queries": len(queries),
        "ks": list(ks),
        "index": retriever.manifest,
        "results": results,
    }


def format_report(report: dict, columns: Sequence[str] | None = None) -> str:
    """Markdown table of the main metrics (latency in ms)."""
    columns = columns or [
        "nDCG@10",
        "R@10",
        "R@20",
        "Hit@10",
        f"MRR@{MRR_CUTOFF}",
        "words_per_chunk@10",
        "latency_ms_p95",
    ]
    renames = {"latency_ms_p95": "p95 ms", "words_per_chunk@10": "words/chunk"}
    header = ["Retriever", *[renames.get(c, c) for c in columns]]
    idx = report["index"]
    lines = [
        f"Split: {report['split']} · {report['num_queries']} queries · chunking {idx['chunking']['strategy']}"
        f" · BM25 {idx['bm25']['tokenizer']} · {idx['embedding']['model']}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for label, metrics in report["results"].items():
        cells = [
            f"{metrics[c]:.1f}" if c.startswith(("latency", "words")) else f"{metrics[c]:.3f}"
            for c in columns
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
