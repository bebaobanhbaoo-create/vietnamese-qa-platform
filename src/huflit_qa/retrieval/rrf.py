"""Reciprocal Rank Fusion for combining BM25 and dense results."""

from collections.abc import Sequence


def reciprocal_rank_fusion(
    results_list: Sequence[Sequence[tuple[str, float]]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists: score(d) = sum_i w_i / (k + rank_i(d)), ranks starting at 1.

    Only ranks matter, so BM25 and cosine scores need no normalisation. ``weights`` (default all 1)
    let a stronger retriever count more. Returns [(chunk_id, score), ...] sorted desc; ties keep
    first-seen order.
    """
    weights = weights or [1.0] * len(results_list)
    scores: dict[str, float] = {}
    for results, weight in zip(results_list, weights, strict=True):
        for rank, (chunk_id, _score) in enumerate(results, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
