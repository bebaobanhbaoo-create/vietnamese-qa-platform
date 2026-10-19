"""Retrieval evaluation: P@k, R@k and MRR for BM25, dense and hybrid search."""

import json
from collections.abc import Callable, Iterable

from huflit_qa.config import Settings
from huflit_qa.evaluation.metrics import mrr as mrr_fn
from huflit_qa.evaluation.metrics import precision_at_k, recall_at_k
from huflit_qa.retrieval.hybrid import HybridRetriever

MODES = ("bm25", "dense", "hybrid")
LABELS = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}


def evaluate_results(all_retrieved_ids: list[list[str]], queries: list[dict], k: int) -> dict[str, float]:
    """Average precision@k, recall@k and MRR over queries.

    all_retrieved_ids[i] is the ranked chunk_id list for queries[i];
    each query dict has a 'relevant_chunk_ids' list.
    """
    p_scores, r_scores, mrr_scores = [], [], []
    for retrieved, query in zip(all_retrieved_ids, queries, strict=True):
        relevant = set(query["relevant_chunk_ids"])
        p_scores.append(precision_at_k(retrieved, relevant, k))
        r_scores.append(recall_at_k(retrieved, relevant, k))
        mrr_scores.append(mrr_fn(retrieved, relevant))

    n = len(queries)
    return {
        "precision": sum(p_scores) / n,
        "recall": sum(r_scores) / n,
        "mrr": sum(mrr_scores) / n,
    }


def load_queries(settings: Settings) -> list[dict]:
    return json.loads(settings.test_queries_path.read_text(encoding="utf-8"))


def run_evaluation(settings: Settings, modes: Iterable[str], k: int | None = None) -> dict:
    """Evaluate the requested modes and return a JSON-serialisable report."""
    k = k or settings.retrieval.top_k
    queries = load_queries(settings)
    retriever = HybridRetriever(settings)

    searchers: dict[str, Callable[[str], list[str]]] = {
        "bm25": lambda q: [cid for cid, _ in retriever.search_bm25(q, k)],
        "dense": lambda q: [cid for cid, _ in retriever.search_dense(q, k)],
        "hybrid": lambda q: [r["chunk_id"] for r in retriever.retrieve(q, top_k=k)],
    }

    results = {}
    for mode in modes:
        all_ids = [searchers[mode](q["query"]) for q in queries]
        results[LABELS[mode]] = evaluate_results(all_ids, queries, k)

    return {
        "k": k,
        "num_queries": len(queries),
        "embedding_model": settings.embedding_model,
        "results": results,
    }


def format_report(report: dict) -> str:
    k = report["k"]
    lines = [f"Evaluation on {report['num_queries']} queries (k={k})", "-" * 40]
    for label, res in report["results"].items():
        lines.append(
            f"  {label:<7} P@{k}={res['precision']:.3f}  R@{k}={res['recall']:.3f}  MRR={res['mrr']:.3f}"
        )
    return "\n".join(lines)
