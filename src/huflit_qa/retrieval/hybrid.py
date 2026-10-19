"""Hybrid retriever: BM25 + dense search fused with Reciprocal Rank Fusion, optionally reranked."""

import json
from pathlib import Path

import numpy as np

from huflit_qa.config import EmbeddingSettings, RerankSettings, RetrievalSettings, Settings
from huflit_qa.indexing.bm25 import load_bm25
from huflit_qa.indexing.encoders import Encoder
from huflit_qa.indexing.manifest import load_manifest
from huflit_qa.indexing.stores import FaissStore, QdrantStore, VectorStore
from huflit_qa.indexing.tokenize import get_tokenizer
from huflit_qa.processing.chunker import index_text
from huflit_qa.retrieval.filters import SearchFilters
from huflit_qa.retrieval.reranker import CrossEncoderReranker
from huflit_qa.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    """Searches an index the way its manifest says it was built (tokenizer, model, query casing).

    Retrieval, fusion and reranking settings are query-time and come from ``settings``.
    """

    def __init__(
        self,
        settings: Settings,
        encoder: Encoder | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.cfg: RetrievalSettings = settings.retrieval
        self.rerank_cfg: RerankSettings = settings.rerank
        self.manifest = load_manifest(settings.manifest_path)
        self.embedding = EmbeddingSettings(
            **{k: v for k, v in self.manifest["embedding"].items() if k != "dim"}
        )
        self.context_prefix = bool(self.manifest["chunking"].get("context_prefix"))
        self.tokenize = get_tokenizer(self.manifest["bm25"]["tokenizer"])

        chunks = json.loads(Path(settings.chunks_path).read_text(encoding="utf-8"))
        self.chunk_map = {c["chunk_id"]: c for c in chunks}

        self.bm25, self.bm25_ids = load_bm25(settings.bm25_path)
        store_type = self.manifest.get("index", {}).get("vector_store", "faiss")
        self.store: VectorStore = (
            QdrantStore(settings.qdrant_path)
            if store_type == "qdrant"
            else FaissStore(settings.faiss_path, settings.faiss_ids_path, self.chunk_map)
        )

        if encoder is not None and encoder.model_name != self.embedding.model:
            raise ValueError(f"Index was built with {self.embedding.model}, encoder is {encoder.model_name}")
        self.encoder = encoder or Encoder(
            self.embedding.model, self.embedding.max_seq_length, self.embedding.batch_size
        )
        if reranker is None and self.rerank_cfg.model:
            reranker = CrossEncoderReranker(self.rerank_cfg.model, self.rerank_cfg.batch_size)
        self.reranker = reranker

    def search_bm25(
        self, query: str, top_k: int, filters: SearchFilters | None = None
    ) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(self.tokenize(query.strip()))
        if filters is not None and not filters.is_empty():
            mask = np.array([filters.matches(self.chunk_map[cid]) for cid in self.bm25_ids])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.bm25_ids[i], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def search_dense(
        self, query: str, top_k: int, filters: SearchFilters | None = None
    ) -> list[tuple[str, float]]:
        q = query.strip()
        if self.embedding.lowercase_query:
            q = q.lower()
        return self.store.search(self.encoder.encode_queries([q])[0], top_k, filters)

    def search_hybrid(
        self, query: str, top_k: int, filters: SearchFilters | None = None
    ) -> list[tuple[str, float]]:
        fused = reciprocal_rank_fusion(
            [
                self.search_bm25(query, self.cfg.top_k_bm25, filters),
                self.search_dense(query, self.cfg.top_k_dense, filters),
            ],
            k=self.cfg.rrf_k,
            weights=[self.cfg.bm25_weight, self.cfg.dense_weight],
        )
        return fused[:top_k]

    def rerank(self, query: str, candidates: list[tuple[str, float]], top_k: int) -> list[tuple[str, float]]:
        if self.reranker is None:
            raise ValueError("No reranker configured (set rerank.model)")
        texts = [index_text(self.chunk_map[cid], self.context_prefix) for cid, _ in candidates]
        scores = self.reranker.score(query, texts)
        ranked = sorted(zip([cid for cid, _ in candidates], scores, strict=True), key=lambda p: -p[1])
        return ranked[:top_k]

    def search(self, query: str, top_k: int, filters: SearchFilters | None = None) -> list[tuple[str, float]]:
        """Full pipeline: hybrid search, then cross-encoder reranking if configured."""
        if self.reranker is None:
            return self.search_hybrid(query, top_k, filters)
        candidates = self.search_hybrid(query, max(self.rerank_cfg.depth, top_k), filters)
        return self.rerank(query, candidates, top_k)

    def retrieve(
        self, query: str, top_k: int | None = None, filters: SearchFilters | None = None
    ) -> list[dict]:
        """Top-k chunks for ``query`` with their metadata and score."""
        results = []
        for chunk_id, score in self.search(query, top_k or self.cfg.top_k, filters):
            chunk = self.chunk_map[chunk_id]
            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "doc_title": chunk["doc_title"],
                    "chunk_title": chunk.get("chunk_title"),
                    "source_url": chunk["source_url"],
                    "date_posted": chunk.get("date_posted"),
                    "issued_date": chunk.get("issued_date"),
                    "issuer_unit": chunk.get("issuer_unit"),
                    "newest_in_group": chunk.get("newest_in_group"),
                    "score": score,
                }
            )
        return results
