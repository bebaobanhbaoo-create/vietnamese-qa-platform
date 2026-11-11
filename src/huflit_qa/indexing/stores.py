"""Dense vectors live in Qdrant: embedded (a folder next to the index) or a Qdrant server.

Cosine similarity on L2-normalised embeddings. At ~10^3 vectors Qdrant returns exactly the
neighbours a brute-force search would (checked against FAISS in eval/EXPERIMENTS_2C.md); it is
used because it filters inside the vector search and updates incrementally, so the weekly crawl
only embeds new or changed chunks.
"""

import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from huflit_qa.retrieval.filters import SearchFilters, chunk_year

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from huflit_qa.config import Settings

logger = logging.getLogger(__name__)

PAYLOAD_FIELDS = ("doc_type", "issuer_unit", "doc_group", "issued_date", "date_posted", "newest_in_group")
BATCH = 256
TIE_MARGIN = 8  # extra hits fetched so ties at the cut-off can be ordered


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


# Embedded Qdrant locks its folder, so a process keeps one client per location.
_CLIENTS: dict[str, "QdrantClient"] = {}


def _client(path: Path | None, url: str | None, api_key: str | None) -> "QdrantClient":
    from qdrant_client import QdrantClient

    key = url or str(Path(path or ".").resolve())
    if key not in _CLIENTS:
        if url:
            _CLIENTS[key] = QdrantClient(url=url, api_key=api_key, timeout=30)
        else:
            assert path is not None
            path.mkdir(parents=True, exist_ok=True)
            _CLIENTS[key] = QdrantClient(path=str(path))
    return _CLIENTS[key]


class QdrantStore:
    COLLECTION = "chunks"

    def __init__(self, path: Path | None = None, url: str | None = None, api_key: str | None = None):
        if (path is None) == (url is None):
            raise ValueError("Give either an embedded Qdrant folder or a server URL")
        self.location = url or str(path)
        self.is_server = url is not None
        self.tie_order: dict[str, int] | None = None  # chunk_id -> rank among exact ties
        self._key = url or str(Path(path or ".").resolve())
        self.client = _client(path, url, api_key)

    @classmethod
    def from_settings(cls, settings: "Settings") -> "QdrantStore":
        """The server when ``index.qdrant_url`` is set, otherwise the folder shipped with the index."""
        idx = settings.index
        if idx.qdrant_url:
            return cls(url=idx.qdrant_url, api_key=idx.qdrant_api_key)
        return cls(path=settings.qdrant_path)

    def close(self) -> None:
        _CLIENTS.pop(self._key, None)
        self.client.close()

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"huflit-qa/{chunk_id}"))

    def wait_ready(self, timeout_s: float = 60.0) -> None:
        """A server started by docker compose may still be booting."""
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                self.client.get_collections()
                return
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1)

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.COLLECTION):
            return
        self.client.create_collection(
            self.COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )
        if not self.is_server:  # embedded mode has no payload indexes
            return
        # Payload indexes let the server filter inside the HNSW search.
        for field, schema in (
            ("doc_type", models.PayloadSchemaType.KEYWORD),
            ("issuer_unit", models.PayloadSchemaType.KEYWORD),
            ("year", models.PayloadSchemaType.INTEGER),
            ("newest_in_group", models.PayloadSchemaType.BOOL),
        ):
            self.client.create_payload_index(self.COLLECTION, field, field_schema=schema)

    def count(self) -> int:
        if not self.client.collection_exists(self.COLLECTION):
            return 0
        return self.client.count(self.COLLECTION, exact=True).count

    def _scroll(self, with_vectors: bool = False):
        if not self.client.collection_exists(self.COLLECTION):
            return
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.COLLECTION, with_payload=True, with_vectors=with_vectors, limit=512, offset=offset
            )
            yield from points
            if offset is None:
                return

    def existing_points(self) -> dict[str, dict]:
        """chunk_id -> stored payload (including ``text_hash``)."""
        return {(p.payload or {})["chunk_id"]: p.payload or {} for p in self._scroll()}

    def vectors(self, chunk_ids: list[str]) -> np.ndarray:
        """Stored vectors in the order of ``chunk_ids`` (used to export the browser demo)."""
        by_id = {(p.payload or {})["chunk_id"]: p.vector for p in self._scroll(with_vectors=True)}
        missing = [cid for cid in chunk_ids if cid not in by_id]
        if missing:
            raise KeyError(f"{len(missing)} chunks have no vector in Qdrant, e.g. {missing[:3]}")
        return np.asarray([by_id[cid] for cid in chunk_ids], dtype="float32")

    def _apply(self, payloads: dict[str, dict], vectors_for, dim: int) -> dict[str, int]:
        """Make the collection hold exactly ``payloads`` (chunk_id -> payload with ``text_hash``).

        ``vectors_for(chunk_ids)`` is called only for chunks that are new or whose text changed;
        metadata-only changes rewrite the payload and keep the vector.
        """
        from qdrant_client import models

        existing = self.existing_points()
        to_write, to_repayload = [], []
        for cid, payload in payloads.items():
            old = existing.get(cid)
            if old is None or old.get("text_hash") != payload["text_hash"]:
                to_write.append(cid)
            elif old != payload:
                to_repayload.append(cid)
        removed = set(existing) - set(payloads)

        self._ensure_collection(dim)
        for start in range(0, len(to_write), BATCH):
            batch = to_write[start : start + BATCH]
            vectors = vectors_for(batch)
            self.client.upsert(
                self.COLLECTION,
                points=[
                    models.PointStruct(
                        id=self.point_id(cid), vector=list(map(float, vec)), payload=payloads[cid]
                    )
                    for cid, vec in zip(batch, vectors, strict=True)
                ],
            )
        for cid in to_repayload:
            self.client.overwrite_payload(self.COLLECTION, payload=payloads[cid], points=[self.point_id(cid)])
        if removed:
            self.client.delete(
                self.COLLECTION,
                points_selector=models.PointIdsList(points=[self.point_id(c) for c in removed]),
            )
        stats = {
            "embedded": len(to_write),
            "payload_updated": len(to_repayload),
            "deleted": len(removed),
            "unchanged": len(payloads) - len(to_write) - len(to_repayload),
        }
        logger.info("Qdrant %s: %s", self.location, json.dumps(stats))
        return stats

    def sync(self, chunks: list[dict], texts: list[str], encoder) -> dict[str, int]:
        """Embed new or changed chunks, rewrite changed metadata, delete chunks that are gone."""
        payloads = {
            c["chunk_id"]: {**payload_for(c), "text_hash": text_hash(t, encoder.model_name)}
            for c, t in zip(chunks, texts, strict=True)
        }
        text_of = {c["chunk_id"]: t for c, t in zip(chunks, texts, strict=True)}
        return self._apply(
            payloads, lambda ids: encoder.encode_passages([text_of[i] for i in ids]), encoder.dim
        )

    def copy_from(self, source: "QdrantStore") -> dict[str, int]:
        """Mirror another collection without re-embedding (seeds a server from the shipped index)."""
        points = {(p.payload or {})["chunk_id"]: p for p in source._scroll(with_vectors=True)}
        if not points:
            raise ValueError(f"{source.location} has no vectors: build the index first")
        dim = len(next(iter(points.values())).vector)
        return self._apply(
            {cid: p.payload or {} for cid, p in points.items()},
            lambda ids: [points[i].vector for i in ids],
            dim,
        )

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
            limit=top_k + TIE_MARGIN,
            query_filter=self.to_qdrant_filter(filters),
            with_payload=["chunk_id"],
        )
        hits = [((p.payload or {})["chunk_id"], float(p.score)) for p in response.points]
        # Identical passages (one per version of a regulation) tie exactly; `tie_order` decides,
        # so the ranking does not depend on the backend (embedded or server).
        order = self.tie_order or {}
        hits.sort(key=lambda h: (-h[1], order.get(h[0], len(order)), h[0]))
        return hits[:top_k]
