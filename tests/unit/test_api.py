import json
import threading

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from huflit_qa.api.app import Services, create_app
from huflit_qa.config import GenerationSettings, RerankSettings
from huflit_qa.generation.llm import FakeLLM, LLMUnavailableError
from huflit_qa.generation.rag import RAGPipeline

CHUNK = {
    "chunk_id": "3208_4_d10",
    "doc_title": "Quy chế đào tạo",
    "chunk_title": "Điều 10. Đăng ký học phần",
    "text": "Sinh viên được đăng ký tối đa 25 tín chỉ trong một học kỳ chính.",
    "source_url": "https://portal.huflit.edu.vn/News/Detail/3208/x",
    "date_posted": "2024-11-25",
    "issued_date": "2021-12-30",
    "issuer_unit": "Chung",
    "newest_in_group": True,
    "score": 0.03,
}


class FakeRetriever:
    manifest = {"num_chunks": 1, "chunking": {"strategy": "v1"}, "embedding": {"model": "fake"}}  # noqa: RUF012
    rerank_cfg = RerankSettings()
    reranker = None

    def __init__(self, chunks):
        self.chunks = chunks
        self.chunk_map = {c["chunk_id"]: c for c in chunks}
        self.last_filters = None

    def retrieve(self, query, top_k=None, filters=None):
        self.last_filters = filters
        return self.chunks[:top_k]


class DownLLM(FakeLLM):
    def available(self):
        return False, "Ollama unreachable"

    def complete(self, messages):
        raise LLMUnavailableError("Ollama unreachable")


def make_client(chunks=(CHUNK,), llm=None):
    retriever = FakeRetriever(list(chunks))
    llm = llm or FakeLLM(answer="Tối đa 25 tín chỉ [1].")
    services = Services(retriever, RAGPipeline(retriever, llm, GenerationSettings()), llm, threading.Lock())
    return TestClient(create_app(services=services)), retriever


def test_health_and_ready():
    client, _ = make_client()
    with client:
        assert client.get("/health").json()["status"] == "ok"
        ready = client.get("/ready").json()
        assert ready["ready"] and ready["index"]["num_chunks"] == 1 and ready["llm_model"] == "fake"


def test_search_passes_filters_and_returns_sources():
    client, retriever = make_client()
    with client:
        resp = client.post(
            "/v1/search", json={"query": "tín chỉ", "top_k": 3, "filters": {"newest_only": True}}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["chunk_id"] == "3208_4_d10"
    assert retriever.last_filters.newest_only is True
    assert resp.headers["x-request-id"]


def test_search_validates_input():
    client, _ = make_client()
    with client:
        assert client.post("/v1/search", json={"query": "x"}).status_code == 422
        assert client.post("/v1/search", json={"query": "tín chỉ", "top_k": 99}).status_code == 422


def test_ask_returns_answer_with_checked_citations():
    client, _ = make_client()
    with client:
        body = client.post("/v1/ask", json={"question": "Tối đa bao nhiêu tín chỉ?"}).json()
    assert body["answer"] == "Tối đa 25 tín chỉ [1]."
    assert body["citations"] == {"cited": [1], "invalid": [], "refused": False, "grounded": True}
    assert body["sources"][0]["issued_date"] == "2021-12-30"


def test_ask_returns_503_when_llm_is_down():
    client, _ = make_client(llm=DownLLM())
    with client:
        assert client.get("/ready").json()["ready"] is False
        resp = client.post("/v1/ask", json={"question": "Tối đa bao nhiêu tín chỉ?"})
    assert resp.status_code == 503 and "Ollama" in resp.json()["detail"]


def test_ask_stream_sends_sources_tokens_and_citation_check():
    client, _ = make_client()
    with client:
        resp = client.post("/v1/ask/stream", json={"question": "Tối đa bao nhiêu tín chỉ?"})
    events = []
    for block in resp.text.strip().split("\n\n"):
        event, data = block.split("\n", 1)
        events.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    assert events[0][0] == "sources" and events[0][1][0]["chunk_id"] == "3208_4_d10"
    assert "".join(d for e, d in events if e == "token") == "Tối đa 25 tín chỉ [1]."
    assert events[-1] == ("done", {"cited": [1], "invalid": [], "refused": False, "grounded": True})


def test_index_page_and_metrics():
    client, _ = make_client()
    with client:
        assert "Hỏi đáp quy chế HUFLIT" in client.get("/").text
        client.get("/health")
        assert "huflit_requests_total" in client.get("/metrics").text


def test_search_only_mode_reports_ready_and_refuses_answers():
    from huflit_qa.generation.llm import DisabledLLM

    client, _ = make_client(llm=DisabledLLM())
    with client:
        ready = client.get("/ready").json()
        assert ready["ready"] is True and ready["mode"] == "search" and ready["llm_available"] is False
        assert client.post("/v1/search", json={"query": "tín chỉ"}).status_code == 200
        resp = client.post("/v1/ask", json={"question": "Tối đa bao nhiêu tín chỉ?"})
    assert resp.status_code == 503 and "search only" in resp.json()["detail"]
