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


def test_qdrant_sync_is_incremental_and_filters_match_faiss(tmp_path):
    pytest.importorskip("qdrant_client")
    pytest.importorskip("faiss")
    from huflit_qa.indexing.stores import FaissStore, QdrantStore

    chunks = [
        _chunk(f"c{i}", doc_type="quy_che" if i % 2 else "thong_bao", posted=f"{2020 + i % 6}-01-01")
        for i in range(30)
    ]
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

    vectors = enc.encode_passages(texts[:-1])
    FaissStore.build(vectors, [c["chunk_id"] for c in chunks[:-1]], tmp_path / "f.bin", tmp_path / "ids.json")
    faiss_store = FaissStore(tmp_path / "f.bin", tmp_path / "ids.json", {c["chunk_id"]: c for c in chunks})

    query = vectors[3]
    for filters in (None, SearchFilters(doc_type="quy_che", year_from=2023)):
        a = [cid for cid, _ in faiss_store.search(query, 5, filters)]
        b = [cid for cid, _ in store.search(query, 5, filters)]
        assert a == b
    store.close()
