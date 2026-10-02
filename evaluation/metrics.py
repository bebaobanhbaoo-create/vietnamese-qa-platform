from typing import List, Dict
import numpy as np

def precision_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """Calculate Precision@K."""
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc in retrieved_k if doc in relevant_set)
    return hits / k

def recall_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """Calculate Recall@K."""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    hits = len(retrieved_k & relevant_set)
    return hits / len(relevant_set)

def mrr(retrieved: List[int], relevant: List[int]) -> float:
    """Calculate Mean Reciprocal Rank."""
    relevant_set = set(relevant)
    for rank, doc in enumerate(retrieved, 1):
        if doc in relevant_set:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval(queries_results: List[Dict]) -> Dict[str, float]:
    """Evaluate retrieval performance across multiple queries."""
    metrics = {'P@5': [], 'P@10': [], 'R@5': [], 'R@10': [], 'MRR': []}
    
    for result in queries_results:
        retrieved = result['retrieved']
        relevant = result['relevant']
        
        metrics['P@5'].append(precision_at_k(retrieved, relevant, 5))
        metrics['P@10'].append(precision_at_k(retrieved, relevant, 10))
        metrics['R@5'].append(recall_at_k(retrieved, relevant, 5))
        metrics['R@10'].append(recall_at_k(retrieved, relevant, 10))
        metrics['MRR'].append(mrr(retrieved, relevant))
    
    return {k: np.mean(v) for k, v in metrics.items()}
