"""BM25 index: build, save, load, search."""

import pickle

from rank_bm25 import BM25Okapi


def build_bm25(chunks):
    """Build BM25 index from chunks.
    Returns (bm25_index, chunk_ids).
    """
    chunk_ids = []
    tokenized = []
    for c in chunks:
        chunk_ids.append(c["chunk_id"])
        tokenized.append(c["text"].lower().split())
    bm25 = BM25Okapi(tokenized)
    return bm25, chunk_ids


def save_bm25(bm25, chunk_ids, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)


def load_bm25(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunk_ids"]


def _get_score(score_index_pair):
    """Lấy score (phần tử đầu) từ tuple (score, index)."""
    return score_index_pair[0]


def search_bm25(bm25, chunk_ids, query, top_k):
    """Return [(chunk_id, score), ...] sorted by score desc."""
    tokens = query.lower().split()
    raw_scores = bm25.get_scores(tokens)

    score_index_list = []
    for i in range(len(raw_scores)):
        score_index_list.append((raw_scores[i], i))
    score_index_list.sort(key=_get_score, reverse=True)

    results = []
    for j in range(min(top_k, len(score_index_list))):
        score, i = score_index_list[j]
        results.append((chunk_ids[i], float(score)))
    return results
