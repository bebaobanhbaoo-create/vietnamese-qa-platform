import os

import numpy as np
import pytest

from huflit_qa.processing.metadata import mark_newest_in_group, parse_issued_date
from huflit_qa.retrieval.filters import SearchFilters
from huflit_qa.retrieval.rrf import reciprocal_rank_fusion


def _chunk(cid, doc_type="thong_bao", issuer="Phòng Đào tạo", posted="2025-03-01", group="1", issued=None):
    return {
        "chunk_id": cid,
        "doc_type": doc_type,
        "issuer_unit": issuer,
        "date_posted": posted,
        "doc_group": group,
        "issued_date": issued,
    }


def test_weighted_rrf_lets_stronger_list_win():
    bm25 = [("a", 1.0), ("b", 0.5)]
    dense = [("b", 0.9), ("a", 0.1)]
    assert reciprocal_rank_fusion([bm25, dense])[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert reciprocal_rank_fusion([bm25, dense], weights=[0.5, 1.0])[0][0] == "b"


def test_issued_date_parsing():
    assert parse_issued_date("Ban hành kèm theo Quyết định số 12 ngày 5 tháng 9 năm 2022") == "2022-09-05"
    assert parse_issued_date("ngày 40 tháng 13 năm 2022") is None
    assert parse_issued_date(None) is None


def test_newest_in_group_and_filters():
    chunks = [
        _chunk("old", doc_type="quy_che", group="3209", issued="2017-06-29", posted="2025-07-16"),
        _chunk("new", doc_type="quy_che", group="3209", issued="2025-05-22", posted="2025-07-16"),
        _chunk("tb", posted="2023-01-01", issuer="Phòng CTSV"),
    ]
    mark_newest_in_group(chunks)
    assert [c["chunk_id"] for c in chunks if c.get("newest_in_group")] == ["new"]

    newest = SearchFilters(newest_only=True)
    assert [c["chunk_id"] for c in chunks if newest.matches(c)] == ["new", "tb"]
    assert not SearchFilters(year_from=2024).matches(chunks[2])
    assert SearchFilters(issuer_unit="Phòng CTSV", doc_type="thong_bao").matches(chunks[2])
    assert SearchFilters().is_empty()


class FakeEncoder:
    """Deterministic 8-d unit vectors from a hash of the text."""

    model_name = "fake"
    dim = 8

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=self.dim).astype("float32")
        return v / np.linalg.norm(v)

    def encode_passages(self, texts, show_progress=False):
        self.calls += len(texts)
        return np.stack([self._vec(t) for t in texts])


def _brute_force(vectors, chunks, query, top_k, filters):
    """Exact cosine search with the Python filter: what Qdrant must return."""
    scores = vectors @ query / np.linalg.norm(query)
    order = [i for i in np.argsort(-scores, kind="stable") if filters is None or filters.matches(chunks[i])]
    return [chunks[i]["chunk_id"] for i in order[:top_k]]


def test_qdrant_sync_is_incremental_and_filters_match_exact_search(tmp_path):
    pytest.importorskip("qdrant_client")
    from huflit_qa.indexing.stores import QdrantStore

    chunks = [
        _chunk(f"c{i}", doc_type="quy_che" if i % 2 else "thong_bao", posted=f"{2020 + i % 6}-01-01")
        for i in range(30)
    ]
    for i, c in enumerate(chunks):
        if c["doc_type"] == "quy_che":
            c["newest_in_group"] = i % 4 == 1
    texts = [f"văn bản số {i}" for i in range(30)]
    enc = FakeEncoder()

    store = QdrantStore(tmp_path / "q")
    assert store.sync(chunks, texts, enc)["embedded"] == 30
    assert store.sync(chunks, texts, enc) == {
        "embedded": 0,
        "payload_updated": 0,
        "deleted": 0,
        "unchanged": 30,
    }
    chunks[0]["issuer_unit"] = "Phòng CTSV"
    assert store.sync(chunks, texts, enc)["payload_updated"] == 1
    assert enc.calls == 30  # metadata changes never re-embed
    assert store.sync(chunks[:-1], texts[:-1], enc)["deleted"] == 1

    kept = chunks[:-1]
    vectors = enc.encode_passages(texts[:-1])
    query = vectors[3] + 0.3 * vectors[8]
    for filters in (
        None,
        SearchFilters(doc_type="quy_che", year_from=2023),
        SearchFilters(newest_only=True),
        SearchFilters(issuer_unit="Phòng CTSV"),
    ):
        expected = _brute_force(vectors, kept, query, 5, filters)
        assert [cid for cid, _ in store.search(query, 5, filters)] == expected

    # The browser demo export reads the stored vectors back in chunk order.
    ids = [c["chunk_id"] for c in kept]
    np.testing.assert_allclose(store.vectors(ids), vectors, atol=1e-6)

    # Seeding another collection (a server in production) copies vectors without re-embedding,
    # and a second copy writes nothing.
    mirror = QdrantStore(path=tmp_path / "mirror")
    assert mirror.copy_from(store)["embedded"] == 29
    assert mirror.copy_from(store) == {"embedded": 0, "payload_updated": 0, "deleted": 0, "unchanged": 29}
    assert mirror.search(query, 5) == store.search(query, 5)
    assert enc.calls == 30 + 29  # only the explicit encode above
    mirror.close()
    store.close()


@pytest.mark.skipif(not os.environ.get("HUFLIT_TEST_QDRANT_URL"), reason="needs a Qdrant server (CI service)")
def test_server_mirrors_embedded_store(tmp_path):
    """What `huflit-qa push-vectors` does in docker compose, against a real Qdrant server."""
    pytest.importorskip("qdrant_client")
    from huflit_qa.indexing.stores import QdrantStore

    chunks = [
        _chunk(f"c{i}", doc_type="quy_che" if i % 2 else "thong_bao", posted=f"{2020 + i % 6}-01-01")
        for i in range(40)
    ]
    for i, c in enumerate(chunks):
        if c["doc_type"] == "quy_che":
            c["newest_in_group"] = i % 4 == 1
    texts = [f"đoạn {i}" for i in range(40)]
    enc = FakeEncoder()
    local = QdrantStore(path=tmp_path / "q")
    local.sync(chunks, texts, enc)

    server = QdrantStore(url=os.environ["HUFLIT_TEST_QDRANT_URL"])
    server.wait_ready(timeout_s=30)
    if server.client.collection_exists(QdrantStore.COLLECTION):
        server.client.delete_collection(QdrantStore.COLLECTION)
    try:
        assert server.copy_from(local)["embedded"] == 40
        assert server.copy_from(local)["unchanged"] == 40
        local.sync(chunks[:-2], texts[:-2], enc)
        assert server.copy_from(local)["deleted"] == 2
        assert server.count() == 38

        query = enc.encode_passages(["truy vấn"])[0]
        for filters in (
            None,
            SearchFilters(doc_type="quy_che", year_from=2023),
            SearchFilters(newest_only=True),
        ):
            assert server.search(query, 10, filters) == pytest.approx(local.search(query, 10, filters))
    finally:
        server.client.delete_collection(QdrantStore.COLLECTION)
        server.close()
        local.close()


def test_exact_ties_prefer_the_newest_version():
    from huflit_qa.retrieval.hybrid import newest_first

    chunks = [
        {"chunk_id": "old", "newest_in_group": False, "issued_date": "2021-03-18"},
        {"chunk_id": "new", "newest_in_group": True, "issued_date": "2021-12-30"},
        {"chunk_id": "tb_a", "date_posted": "2021-09-29"},
        {"chunk_id": "tb_b", "date_posted": "2022-04-18"},
    ]
    rank = newest_first(chunks)
    assert sorted(rank, key=rank.get) == ["new", "tb_b", "tb_a", "old"]


def test_store_needs_exactly_one_location(tmp_path):
    pytest.importorskip("qdrant_client")
    from huflit_qa.indexing.stores import QdrantStore

    with pytest.raises(ValueError):
        QdrantStore()
    with pytest.raises(ValueError):
        QdrantStore(path=tmp_path, url="http://localhost:6333")
