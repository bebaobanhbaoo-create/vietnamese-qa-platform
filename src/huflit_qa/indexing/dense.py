"""Dense index: exact inner-product search over L2-normalised embeddings (FAISS IndexFlatIP).

Exact search is the right choice at this scale (~10^3 vectors, < 1 ms per query); approximate
indexes (IVF, HNSW) only pay off from ~10^5 vectors and would cost recall here.
"""

import json
from pathlib import Path

import faiss
import numpy as np


def build_dense(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_dense(index: faiss.Index, chunk_ids: list[str], index_path: Path, ids_path: Path) -> None:
    """Row i of the FAISS index is ``chunk_ids[i]``; chunk text and metadata live in chunks.json."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(chunk_ids), encoding="utf-8")


def load_dense(index_path: Path, ids_path: Path) -> tuple[faiss.Index, list[str]]:
    return faiss.read_index(str(index_path)), json.loads(ids_path.read_text(encoding="utf-8"))


def search_dense(
    index: faiss.Index, chunk_ids: list[str], query_vector: np.ndarray, top_k: int
) -> list[tuple[str, float]]:
    """Return [(chunk_id, score), ...] sorted by score desc."""
    scores, rows = index.search(query_vector.reshape(1, -1), top_k)
    return [(chunk_ids[i], float(s)) for s, i in zip(scores[0], rows[0], strict=True) if i >= 0]
