import math

import pytest

from huflit_qa.evaluation.metrics import hit_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k


class TestRetrievalMetrics:
    """Test suite for retrieval evaluation metrics."""

    def test_precision_at_k_perfect(self):
        """Test P@K with perfect retrieval."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = [1, 2, 3, 4, 5]
        assert precision_at_k(retrieved, relevant, 5) == 1.0

    def test_precision_at_k_partial(self):
        """Test P@K with partial matches."""
        retrieved = [1, 2, 6, 7, 8]
        relevant = [1, 2, 3, 4, 5]
        assert precision_at_k(retrieved, relevant, 5) == 0.4

    def test_recall_at_k_perfect(self):
        """Test R@K with perfect recall."""
        retrieved = [1, 2, 3, 4, 5, 6, 7]
        relevant = [1, 2, 3]
        assert recall_at_k(retrieved, relevant, 5) == 1.0

    def test_recall_at_k_partial(self):
        """Test R@K with partial recall."""
        retrieved = [1, 6, 7, 8, 9]
        relevant = [1, 2, 3, 4, 5]
        assert recall_at_k(retrieved, relevant, 5) == 0.2

    def test_mrr_first_position(self):
        """Test MRR when relevant doc is first."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = [1]
        assert mrr(retrieved, relevant) == 1.0

    def test_mrr_third_position(self):
        """Test MRR when relevant doc is third."""
        retrieved = [6, 7, 1, 8, 9]
        relevant = [1]
        assert abs(mrr(retrieved, relevant) - 1 / 3) < 0.001

    def test_mrr_not_found(self):
        """Test MRR when no relevant doc found."""
        retrieved = [6, 7, 8, 9, 10]
        relevant = [1, 2, 3]
        assert mrr(retrieved, relevant) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestRankQualityMetrics:
    def test_hit_at_k(self):
        assert hit_at_k(["x", "y", "a"], {"a"}, 3) == 1.0
        assert hit_at_k(["x", "y", "a"], {"a"}, 2) == 0.0

    def test_ndcg_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == 1.0

    def test_ndcg_penalises_lower_rank(self):
        # one relevant item at rank 2: (1/log2(3)) / (1/log2(2))
        assert abs(ndcg_at_k(["x", "a"], {"a"}, 2) - 1 / math.log2(3)) < 1e-9

    def test_ndcg_no_relevant(self):
        assert ndcg_at_k(["a"], set(), 5) == 0.0
