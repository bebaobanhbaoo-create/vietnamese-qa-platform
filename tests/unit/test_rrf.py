from huflit_qa.retrieval.rrf import reciprocal_rank_fusion


def test_item_ranked_high_in_both_lists_wins():
    bm25 = [("a", 9.0), ("b", 5.0), ("c", 1.0)]
    dense = [("b", 0.9), ("a", 0.8), ("d", 0.1)]
    fused = reciprocal_rank_fusion([bm25, dense], k=60)
    ids = [cid for cid, _ in fused]
    assert set(ids[:2]) == {"a", "b"}
    assert ids[-1] in {"c", "d"}


def test_scores_use_rank_not_raw_score():
    fused = dict(reciprocal_rank_fusion([[("a", 1000.0), ("b", 0.001)]], k=60))
    assert fused["a"] == 1 / 61
    assert fused["b"] == 1 / 62


def test_empty_input():
    assert reciprocal_rank_fusion([[], []]) == []
