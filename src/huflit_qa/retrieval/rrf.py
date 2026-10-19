"""Reciprocal Rank Fusion for combining BM25 and dense results."""


def _get_score(pair):
    """Lấy score từ tuple (chunk_id, score)."""
    return pair[1]


def reciprocal_rank_fusion(results_list, k=60):
    """Fuse multiple ranked result lists using RRF.

    results_list: list of [(chunk_id, score), ...], each sorted desc.
    k: RRF constant (default 60).
    Returns: [(chunk_id, rrf_score), ...] sorted desc.
    """
    rrf_scores = {}
    for results in results_list:
        rank = 0
        for chunk_id, _score in results:
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] = rrf_scores[chunk_id] + 1.0 / (k + rank + 1)
            rank = rank + 1

    pairs = list(rrf_scores.items())
    pairs.sort(key=_get_score, reverse=True)
    return pairs
