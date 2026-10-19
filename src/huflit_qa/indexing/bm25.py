"""BM25 index: build, save, load, search."""

import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


def build_bm25(texts: list[str], tokenize: Callable[[str], list[str]]) -> BM25Okapi:
    return BM25Okapi([tokenize(t) for t in texts])


def save_bm25(bm25: BM25Okapi, chunk_ids: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)


def load_bm25(path: Path) -> tuple[BM25Okapi, list[str]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunk_ids"]


def search_bm25(
    bm25: BM25Okapi, chunk_ids: list[str], query_tokens: list[str], top_k: int
) -> list[tuple[str, float]]:
    """Return [(chunk_id, score), ...] sorted by score desc (ties keep index order)."""
    scores = bm25.get_scores(query_tokens)
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(chunk_ids[i], float(scores[i])) for i in order]
