"""Bước 3: Chunk dữ liệu + build BM25 và FAISS index.

Yêu cầu: quy_che_flat.json và thong_bao_flat.json đã tồn tại (chạy run_process.py trước).

Usage:
    python offline/run_index.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EMBEDDING_MODEL
from offline.ir_build.chunker import chunk_all
from offline.ir_build.bm25 import build_bm25, save_bm25
from offline.ir_build.dense import build_dense, save_dense

BASE_DIR = Path(__file__).parent.parent
QUY_CHE_FLAT = BASE_DIR / "data/processed/by_doc_type/quy_che_flat.json"
THONG_BAO_FLAT = BASE_DIR / "data/processed/by_doc_type/thong_bao_flat.json"
CHUNKS_PATH = BASE_DIR / "data/processed/chunks.json"
BM25_PATH = BASE_DIR / "data/index/bm25.pkl"
FAISS_PATH = BASE_DIR / "data/index/faiss.bin"
META_PATH = BASE_DIR / "data/index/faiss_meta.json"


def main():
    print("=== Bước 3: Chunk + Build Index ===")

    print("1/3 Chunking...")
    chunks = chunk_all(QUY_CHE_FLAT, THONG_BAO_FLAT)
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"  {len(chunks)} chunks → {CHUNKS_PATH}")

    print("2/3 Building BM25 index...")
    bm25, chunk_ids = build_bm25(chunks)
    save_bm25(bm25, chunk_ids, BM25_PATH)
    print(f"  BM25 → {BM25_PATH}")

    print("3/3 Building FAISS index (có thể mất vài phút)...")
    faiss_index, meta = build_dense(chunks, EMBEDDING_MODEL)
    save_dense(faiss_index, meta, FAISS_PATH, META_PATH)
    print(f"  FAISS → {FAISS_PATH}")

    print("=== Index hoàn tất ===")


if __name__ == "__main__":
    main()
