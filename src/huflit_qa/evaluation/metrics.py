"""Ranking metrics with binary relevance: Precision@k, Recall@k, Hit@k, nDCG@k, MRR.

Metrics are computed from *hits*: for each retrieved item, the set of relevant label units it
covers. When retrieved items are the label units themselves, a hit is ``{item}`` or ``set()`` and
the metrics reduce to their textbook definitions. When items are chunks of another chunking (see
``projection.py``), one chunk can cover several relevant units; recall then counts units, while
precision, nDCG and MRR count a chunk as relevant if it covers any not-yet-seen relevant unit.
"""

import math
from collections.abc import Collection, Sequence

Hits = Sequence[set[str]]


def to_hits(retrieved: Sequence[str], relevant: Collection[str]) -> list[set[str]]:
    return [{cid} if cid in relevant else set() for cid in retrieved]


def _new_hit_flags(hits: Hits, k: int) -> list[bool]:
    """Whether each of the top-k items adds a relevant unit not covered by an earlier item."""
    seen: set[str] = set()
    flags = []
    for h in hits[:k]:
        flags.append(bool(h - seen))
        seen |= h
    return flags


def precision_from_hits(hits: Hits, k: int) -> float:
    return sum(1 for h in hits[:k] if h) / k if k else 0.0


def recall_from_hits(hits: Hits, n_relevant: int, k: int) -> float:
    if not n_relevant:
        return 0.0
    return len(set().union(*hits[:k])) / n_relevant


def hit_from_hits(hits: Hits, k: int) -> float:
    return 1.0 if any(hits[:k]) else 0.0


def ndcg_from_hits(hits: Hits, n_relevant: int, k: int) -> float:
    if not n_relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 2) for rank, new in enumerate(_new_hit_flags(hits, k)) if new)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(n_relevant, k)))
    return dcg / ideal


def mrr_from_hits(hits: Hits) -> float:
    for rank, h in enumerate(hits, 1):
        if h:
            return 1.0 / rank
    return 0.0


# Item-level wrappers (retrieved ids are the label units themselves).


def precision_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    return precision_from_hits(to_hits(retrieved, relevant), k)


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of relevant items found in top-k retrieved."""
    return recall_from_hits(to_hits(retrieved, relevant), len(relevant), k)


def hit_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1.0 if at least one relevant item is in the top-k, else 0.0."""
    return hit_from_hits(to_hits(retrieved, relevant), k)


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Normalised DCG with binary gains; the ideal ranking puts all relevant items first."""
    return ndcg_from_hits(to_hits(retrieved, relevant), len(relevant), k)


def mrr(retrieved: Sequence[str], relevant: Collection[str]) -> float:
    """Reciprocal rank of the first relevant result (0.0 if none)."""
    return mrr_from_hits(to_hits(retrieved, relevant))
