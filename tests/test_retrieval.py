import pytest
from online.evaluation.metrics import precision_at_k, recall_at_k, mrr

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
        assert abs(mrr(retrieved, relevant) - 1/3) < 0.001
    
    def test_mrr_not_found(self):
        """Test MRR when no relevant doc found."""
        retrieved = [6, 7, 8, 9, 10]
        relevant = [1, 2, 3]
        assert mrr(retrieved, relevant) == 0.0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
