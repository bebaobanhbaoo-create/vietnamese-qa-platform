"""Ranking metrics with binary relevance: Precision@k, Recall@k, Hit@k, nDCG@k, MRR."""

import math
from collections.abc import Collection, Sequence


def precision_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    if k == 0:
        return 0.0
    hits = sum(1 for cid in retrieved[:k] if cid in relevant)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of relevant items found in top-k retrieved."""
    if not relevant:
        return 0.0
    hits = sum(1 for cid in retrieved[:k] if cid in relevant)
    return hits / len(relevant)


def hit_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1.0 if at least one relevant item is in the top-k, else 0.0."""
    return 1.0 if any(cid in relevant for cid in retrieved[:k]) else 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Normalised DCG with binary gains; the ideal ranking puts all relevant items first."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 2) for rank, cid in enumerate(retrieved[:k]) if cid in relevant)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal


def mrr(retrieved: Sequence[str], relevant: Collection[str]) -> float:
    """Reciprocal rank of the first relevant result (0.0 if none)."""
    for rank, cid in enumerate(retrieved, 1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0
