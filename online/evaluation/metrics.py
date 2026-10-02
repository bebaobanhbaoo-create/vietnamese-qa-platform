"""Evaluation metrics: Precision@k, Recall@k, MRR."""


def precision_at_k(retrieved, relevant, k):
    """Fraction of top-k retrieved items that are relevant."""
    if k == 0:
        return 0.0
    hits = 0
    for cid in retrieved[:k]:
        if cid in relevant:
            hits = hits + 1
    return hits / k


def recall_at_k(retrieved, relevant, k):
    """Fraction of relevant items found in top-k retrieved."""
    if not relevant:
        return 0.0
    hits = 0
    for cid in retrieved[:k]:
        if cid in relevant:
            hits = hits + 1
    return hits / len(relevant)


def mrr(retrieved, relevant):
    """Mean Reciprocal Rank — 1/rank of first relevant result."""
    rank = 1
    for cid in retrieved:
        if cid in relevant:
            return 1.0 / rank
        rank = rank + 1
    return 0.0
