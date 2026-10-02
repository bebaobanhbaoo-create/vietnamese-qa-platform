"""Bước 4: Truy vấn hybrid retrieval.

Yêu cầu: index đã được build (chạy run_index.py trước).

Usage:
    python online/run_query.py "điều kiện xét tốt nghiệp"
    python online/run_query.py "câu hỏi" --top-k 5
"""
import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))  # root

from online.retrieval.hybrid import HybridRetriever
from config import TOP_K_RESULTS


def main():
    parser = argparse.ArgumentParser(description="Truy vấn HUFLIT IR")
    parser.add_argument("query", type=str, help="Câu hỏi cần tìm kiếm")
    parser.add_argument("--top-k", type=int, default=TOP_K_RESULTS)
    args = parser.parse_args()

    retriever = HybridRetriever()
    results = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\nQuery: {args.query}")
    print(f"Top {len(results)} kết quả:\n")
    for i, r in enumerate(results, 1):
        title = r['chunk_title'] or r['doc_title']
        print(f"[{i}] {title}")
        print(f"    Score: {r['score']:.4f}")
        print(f"    URL: {r['source_url']}")
        print(f"    {r['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
