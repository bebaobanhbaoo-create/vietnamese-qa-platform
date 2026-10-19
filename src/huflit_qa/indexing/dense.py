"""Dense index: build (FAISS), save, load, search."""

import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_encoder(model_name: str) -> SentenceTransformer:
    """Load from the local HF cache first so repeated runs work offline."""
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def build_dense(chunks, model_name):
    """Encode all chunk texts, build FAISS IndexFlatIP.
    Returns (faiss_index, meta_list).
    meta_list[i] = metadata dict for FAISS vector at position i.
    """
    model = load_encoder(model_name)

    texts = []
    for c in chunks:
        texts.append(c["text"])

    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    meta = []
    for c in chunks:
        meta.append(
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "doc_title": c["doc_title"],
                "chunk_title": c.get("chunk_title"),
                "source_url": c["source_url"],
                "date_posted": c.get("date_posted"),
            }
        )
    return index, meta


def save_dense(index, meta, index_path, meta_path):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_dense(index_path, meta_path):
    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, meta


def search_dense(index, meta, model, query, top_k):
    """Return [(chunk_id, score), ...] sorted by score desc."""
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0], strict=True):
        if idx >= 0:
            results.append((meta[idx]["chunk_id"], float(score)))
    return results
