"""Bước 5: Đánh giá hiệu năng retrieval.

Yêu cầu: index đã được build (chạy run_index.py trước).

Usage:
    python online/run_evaluate.py
    python online/run_evaluate.py --mode bm25
    python online/run_evaluate.py --mode dense
    python online/run_evaluate.py --mode hybrid
    python online/run_evaluate.py --mode all
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # root

from config import TOP_K_RESULTS, EMBEDDING_MODEL
from online.evaluation.evaluate import evaluate_results, print_results

BASE_DIR = Path(__file__).parent.parent
BM25_PATH = BASE_DIR / "data/index/bm25.pkl"
FAISS_PATH = BASE_DIR / "data/index/faiss.bin"
META_PATH = BASE_DIR / "data/index/faiss_meta.json"
TEST_QUERIES_PATH = Path(__file__).parent / "evaluation/test_queries.json"


def search_bm25_all(bm25, bm25_ids, queries, k):
    """Chạy BM25 cho tất cả queries. Trả về list các list chunk_id."""
    from offline.ir_build.bm25 import search_bm25
    all_ids = []
    for q in queries:
        query = q['query'].lower().strip()
        hits = search_bm25(bm25, bm25_ids, query, k)
        ids = []
        for chunk_id, score in hits:
            ids.append(chunk_id)
        all_ids.append(ids)
    return all_ids


def search_dense_all(faiss_index, faiss_meta, model, queries, k):
    """Chạy Dense search cho tất cả queries. Trả về list các list chunk_id."""
    from offline.ir_build.dense import search_dense
    all_ids = []
    for q in queries:
        query = q['query'].lower().strip()
        hits = search_dense(faiss_index, faiss_meta, model, query, k)
        ids = []
        for chunk_id, score in hits:
            ids.append(chunk_id)
        all_ids.append(ids)
    return all_ids


def search_hybrid_all(retriever, queries, k):
    """Chạy Hybrid search cho tất cả queries. Trả về list các list chunk_id."""
    all_ids = []
    for q in queries:
        hits = retriever.retrieve(q['query'], top_k=k)
        ids = []
        for r in hits:
            ids.append(r['chunk_id'])
        all_ids.append(ids)
    return all_ids


def main():
    parser = argparse.ArgumentParser(description="Đánh giá IR pipeline")
    parser.add_argument(
        "--mode",
        choices=["bm25", "dense", "hybrid", "all"],
        default="hybrid",
    )
    parser.add_argument("--output", type=Path, help="Lưu kết quả ra file JSON")
    args = parser.parse_args()

    queries = json.loads(TEST_QUERIES_PATH.read_text(encoding='utf-8'))
    k = TOP_K_RESULTS
    print(f"\nEvaluation trên {len(queries)} queries (k={k}), mode={args.mode}\n")

    results = {}

    if args.mode in ('bm25', 'all'):
        from offline.ir_build.bm25 import load_bm25
        bm25, bm25_ids = load_bm25(BM25_PATH)
        all_ids = search_bm25_all(bm25, bm25_ids, queries, k)
        results['BM25'] = evaluate_results(all_ids, queries, k)

    if args.mode in ('dense', 'all'):
        from offline.ir_build.dense import load_dense
        from sentence_transformers import SentenceTransformer
        faiss_index, faiss_meta = load_dense(FAISS_PATH, META_PATH)
        try:
            model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        except Exception:
            model = SentenceTransformer(EMBEDDING_MODEL)
        all_ids = search_dense_all(faiss_index, faiss_meta, model, queries, k)
        results['Dense'] = evaluate_results(all_ids, queries, k)

    if args.mode in ('hybrid', 'all'):
        from online.retrieval.hybrid import HybridRetriever
        retriever = HybridRetriever()
        all_ids = search_hybrid_all(retriever, queries, k)
        results['Hybrid'] = evaluate_results(all_ids, queries, k)

    print(f"  {'-' * 38}")
    for label, res in results.items():
        print_results(label, res, k)

    if args.output:
        report = {
            'k': k,
            'num_queries': len(queries),
            'embedding_model': EMBEDDING_MODEL,
            'results': results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\nĐã lưu → {args.output}")


if __name__ == "__main__":
    main()
