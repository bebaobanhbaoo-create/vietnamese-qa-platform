"""Hybrid retriever: BM25 + FAISS fused with Reciprocal Rank Fusion."""

import json

from huflit_qa.config import Settings
from huflit_qa.indexing.bm25 import load_bm25, search_bm25
from huflit_qa.indexing.dense import load_dense, load_encoder, search_dense
from huflit_qa.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bm25, self.bm25_ids = load_bm25(settings.bm25_path)
        self.faiss_index, self.faiss_meta = load_dense(settings.faiss_path, settings.faiss_meta_path)
        self.model = load_encoder(settings.embedding_model)

        chunks = json.loads(settings.chunks_path.read_text(encoding="utf-8"))
        self.chunk_map = {c["chunk_id"]: c for c in chunks}

    def search_bm25(self, query: str, top_k: int) -> list[tuple[str, float]]:
        return search_bm25(self.bm25, self.bm25_ids, query.lower().strip(), top_k)

    def search_dense(self, query: str, top_k: int) -> list[tuple[str, float]]:
        return search_dense(self.faiss_index, self.faiss_meta, self.model, query.lower().strip(), top_k)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return the top-k chunks for ``query`` with their metadata and RRF score."""
        cfg = self.settings.retrieval
        top_k = top_k or cfg.top_k
        fused = reciprocal_rank_fusion(
            [self.search_bm25(query, cfg.top_k_bm25), self.search_dense(query, cfg.top_k_dense)],
            k=cfg.rrf_k,
        )

        output = []
        for chunk_id, score in fused[:top_k]:
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            output.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "doc_title": chunk["doc_title"],
                    "chunk_title": chunk.get("chunk_title"),
                    "source_url": chunk["source_url"],
                    "date_posted": chunk.get("date_posted"),
                    "score": score,
                }
            )
        return output
