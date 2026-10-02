"""Tính điểm đánh giá cho IR pipeline."""
from online.evaluation.metrics import precision_at_k, recall_at_k, mrr as mrr_fn


def evaluate_results(all_retrieved_ids, queries, k):
    """Tính precision, recall, MRR cho kết quả retrieval.

    all_retrieved_ids: list các list chunk_id, mỗi list ứng với một query
    queries: list các dict có key 'query', 'relevant_chunk_ids'
    k: số kết quả lấy ra để đánh giá
    Returns: dict với 'precision', 'recall', 'mrr'
    """
    p_scores = []
    r_scores = []
    mrr_scores = []

    for i in range(len(queries)):
        retrieved = all_retrieved_ids[i]
        relevant = set(queries[i]['relevant_chunk_ids'])
        p_scores.append(precision_at_k(retrieved, relevant, k))
        r_scores.append(recall_at_k(retrieved, relevant, k))
        mrr_scores.append(mrr_fn(retrieved, relevant))

    n = len(queries)
    return {
        'precision': sum(p_scores) / n,
        'recall': sum(r_scores) / n,
        'mrr': sum(mrr_scores) / n,
    }


def print_results(label, results, k):
    p = round(results['precision'], 3)
    r = round(results['recall'], 3)
    m = round(results['mrr'], 3)
    print(f"  {label}  P@{k}={p}  R@{k}={r}  MRR={m}")
