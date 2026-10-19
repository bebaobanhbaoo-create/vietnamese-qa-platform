"""Vector stores behind one interface: FAISS (exact, rebuilt) and Qdrant (payload filters, incremental).

Both use cosine similarity on L2-normalised embeddings, so at this corpus size they return the same
neighbours; Qdrant is used for metadata filtering and for updating the index when the crawler adds
documents, not for speed.
"""

import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Protocol

import numpy as np

from huflit_qa.retrieval.filters import SearchFilters, chunk_year

logger = logging.getLogger(__name__)

PAYLOAD_FIELDS = ("doc_type", "issuer_unit", "doc_group", "issued_date", "date_posted", "newest_in_group")


class VectorStore(Protocol):
    def search(
        self, vector: np.ndarray, top_k: int, filters: SearchFilters | None = None
    ) -> list[tuple[str, float]]: ...


def text_hash(text: str, model: str) -> str:
    """Changes only when the embedding has to be recomputed."""
    return hashlib.sha1(f"{model}\n{text}".encode()).hexdigest()


def payload_for(chunk: dict) -> dict:
    payload = {f: chunk.get(f) for f in PAYLOAD_FIELDS}
    payload.update(chunk_id=chunk["chunk_id"], year=chunk_year(chunk))
    return payload


class FaissStore:
    def __init__(self, index_path: Path, ids_path: Path, chunk_map: dict[str, dict]):
        from huflit_qa.indexing.dense import load_dense

        self.index, self.ids = load_dense(index_path, ids_path)
        self.chunk_map = chunk_map

    @staticmethod
    def build(vectors: np.ndarray, chunk_ids: list[str], index_path: Path, ids_path: Path) -> None:
        from huflit_qa.indexing.dense import build_dense, save_dense

        save_dense(build_dense(vectors), chunk_ids, index_path, ids_path)

    def search(
        self, vector: np.ndarray, top_k: int, filters: SearchFilters | None = None
    ) -> list[tuple[str, float]]:
        from huflit_qa.indexing.dense import search_dense

        if filters is None or filters.is_empty():
            return search_dense(self.index, self.ids, vector, top_k)
        # Exact search over everything, then filter: cheap at ~10^3 vectors.
        hits = search_dense(self.index, self.ids, vector, len(self.ids))
        return [h for h in hits if filters.matches(self.chunk_map[h[0]])][:top_k]


class QdrantStore:
    COLLECTION = "chunks"

    def __init__(self, path: Path):
        from qdrant_client import QdrantClient

        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))  # embedded mode, no server

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"huflit-qa/{chunk_id}"))

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.COLLECTION):
            return
        self.client.create_collection(
            self.COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )

    def existing_points(self) -> dict[str, dict]:
        """chunk_id -> stored payload (including ``text_hash``)."""
        if not self.client.collection_exists(self.COLLECTION):
            return {}
        stored: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(self.COLLECTION, with_payload=True, limit=512, offset=offset)
            for p in points:
                payload = p.payload or {}
                stored[payload["chunk_id"]] = payload
            if offset is None:
                return stored

    def sync(self, chunks: list[dict], texts: list[str], encoder) -> dict[str, int]:
        """Make the collection match ``chunks``: embed new or changed text, rewrite payloads whose
        metadata changed, delete chunks that no longer exist."""
        from qdrant_client import models

        existing = self.existing_points()
        payloads = [
            {**payload_for(c), "text_hash": text_hash(t, encoder.model_name)}
            for c, t in zip(chunks, texts, strict=True)
        ]
        to_embed, to_repayload = [], []
        for i, chunk in enumerate(chunks):
            old = existing.get(chunk["chunk_id"])
            if old is None or old.get("text_hash") != payloads[i]["text_hash"]:
                to_embed.append(i)
            elif old != payloads[i]:
                to_repayload.append(i)
        removed = set(existing) - {c["chunk_id"] for c in chunks}

        self._ensure_collection(encoder.dim)
        if to_embed:
            vectors = encoder.encode_passages([texts[i] for i in to_embed])
            points = [
                models.PointStruct(
                    id=self.point_id(chunks[i]["chunk_id"]), vector=vectors[row].tolist(), payload=payloads[i]
                )
                for row, i in enumerate(to_embed)
            ]
            for start in range(0, len(points), 256):
                self.client.upsert(self.COLLECTION, points=points[start : start + 256])
        for i in to_repayload:
            self.client.overwrite_payload(
                self.COLLECTION, payload=payloads[i], points=[self.point_id(chunks[i]["chunk_id"])]
            )
        if removed:
            self.client.delete(
                self.COLLECTION,
                points_selector=models.PointIdsList(points=[self.point_id(c) for c in removed]),
            )
        stats = {
            "embedded": len(to_embed),
            "payload_updated": len(to_repayload),
            "deleted": len(removed),
            "unchanged": len(chunks) - len(to_embed) - len(to_repayload),
        }
        logger.info("Qdrant sync: %s", json.dumps(stats))
        return stats

    @staticmethod
    def to_qdrant_filter(filters: SearchFilters | None):
        from qdrant_client import models

        if filters is None or filters.is_empty():
            return None
        must: list = []
        must_not: list = []
        if filters.doc_type:
            must.append(
                models.FieldCondition(key="doc_type", match=models.MatchValue(value=filters.doc_type))
            )
        if filters.issuer_unit:
            must.append(
                models.FieldCondition(key="issuer_unit", match=models.MatchValue(value=filters.issuer_unit))
            )
        if filters.year_from:
            must.append(models.FieldCondition(key="year", range=models.Range(gte=filters.year_from)))
        if filters.newest_only:
            must_not.append(
                models.Filter(
                    must=[
                        models.FieldCondition(key="doc_type", match=models.MatchValue(value="quy_che")),
                        models.FieldCondition(key="newest_in_group", match=models.MatchValue(value=False)),
                    ]
                )
            )
        return models.Filter(must=must or None, must_not=must_not or None)

    def search(
        self, vector: np.ndarray, top_k: int, filters: SearchFilters | None = None
    ) -> list[tuple[str, float]]:
        response = self.client.query_points(
            self.COLLECTION,
            query=vector.tolist(),
            limit=top_k,
            query_filter=self.to_qdrant_filter(filters),
            with_payload=["chunk_id"],
        )
        return [((p.payload or {})["chunk_id"], float(p.score)) for p in response.points]
