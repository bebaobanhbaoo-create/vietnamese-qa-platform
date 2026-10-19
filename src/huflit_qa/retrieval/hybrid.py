"""Hybrid retriever: BM25 + FAISS fused with Reciprocal Rank Fusion."""

import json
from pathlib import Path

from huflit_qa.config import EmbeddingSettings, RetrievalSettings, Settings
from huflit_qa.indexing.bm25 import load_bm25, search_bm25
from huflit_qa.indexing.dense import load_dense, search_dense
from huflit_qa.indexing.encoders import Encoder
from huflit_qa.indexing.manifest import load_manifest
from huflit_qa.indexing.tokenize import get_tokenizer
from huflit_qa.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    """Searches an index the way its manifest says it was built (tokenizer, model, query casing)."""

    def __init__(self, settings: Settings, encoder: Encoder | None = None):
        self.cfg: RetrievalSettings = settings.retrieval
        self.manifest = load_manifest(settings.manifest_path)
        self.embedding = EmbeddingSettings(
            **{k: v for k, v in self.manifest["embedding"].items() if k != "dim"}
        )
        self.tokenize = get_tokenizer(self.manifest["bm25"]["tokenizer"])

        self.bm25, self.bm25_ids = load_bm25(settings.bm25_path)
        self.faiss_index, self.faiss_ids = load_dense(settings.faiss_path, settings.faiss_ids_path)
        if encoder is not None and encoder.model_name != self.embedding.model:
            raise ValueError(f"Index was built with {self.embedding.model}, encoder is {encoder.model_name}")
        self.encoder = encoder or Encoder(
            self.embedding.model, self.embedding.max_seq_length, self.embedding.batch_size
        )

        chunks = json.loads(Path(settings.chunks_path).read_text(encoding="utf-8"))
        self.chunk_map = {c["chunk_id"]: c for c in chunks}

    def search_bm25(self, query: str, top_k: int) -> list[tuple[str, float]]:
        return search_bm25(self.bm25, self.bm25_ids, self.tokenize(query.strip()), top_k)

    def search_dense(self, query: str, top_k: int) -> list[tuple[str, float]]:
        q = query.strip()
        if self.embedding.lowercase_query:
            q = q.lower()
        return search_dense(self.faiss_index, self.faiss_ids, self.encoder.encode_queries([q])[0], top_k)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return the top-k chunks for ``query`` with their metadata and RRF score."""
        top_k = top_k or self.cfg.top_k
        fused = reciprocal_rank_fusion(
            [self.search_bm25(query, self.cfg.top_k_bm25), self.search_dense(query, self.cfg.top_k_dense)],
            k=self.cfg.rrf_k,
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
