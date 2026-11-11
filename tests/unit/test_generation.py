import json
import re
from typing import ClassVar

import httpx
import pytest

from huflit_qa.config import GenerationSettings
from huflit_qa.evaluation.answer_eval import score_answers
from huflit_qa.generation.citations import check_citations, is_refusal
from huflit_qa.generation.llm import FakeLLM, LLMUnavailableError, OllamaLLM
from huflit_qa.generation.prompts import REFUSAL, build_messages
from huflit_qa.generation.rag import RAGPipeline, fit_budget


def _chunk(cid, text="Sinh viên được đăng ký tối đa 25 tín chỉ.", **extra):
    return {
        "chunk_id": cid,
        "doc_title": "Quy chế đào tạo",
        "chunk_title": "Điều 10. Đăng ký học phần",
        "text": text,
        "source_url": f"https://portal/{cid}",
        "date_posted": "2024-11-25",
        "issued_date": "2021-12-30",
        "issuer_unit": "Chung",
        **extra,
    }


class FakeRetriever:
    manifest: ClassVar[dict] = {"embedding": {"model": "fake"}}
    reranker = None

    def __init__(self, chunks):
        self.chunks = chunks
        self.chunk_map = {c["chunk_id"]: c for c in chunks}

    def retrieve(self, query, top_k=None, filters=None):
        return self.chunks[:top_k]


def test_citation_parsing():
    check = check_citations("Tối đa 25 tín chỉ [1]. Học kỳ phụ [2, 3] và [1-2]. Sai [7].", n_sources=3)
    assert check.cited == [1, 2, 3]
    assert check.invalid == [7]
    assert not check.grounded
    assert check_citations("Tối đa 25 tín chỉ [2].", 3).grounded
    assert not check_citations("Tối đa 25 tín chỉ.", 3).grounded


def test_refusal_is_grounded():
    check = check_citations(REFUSAL, 3)
    assert check.refused and check.grounded
    assert is_refusal("  không tìm thấy thông tin này trong các văn bản hiện có ")


def test_prompt_numbers_sources_with_dates():
    messages = build_messages("Tối đa bao nhiêu tín chỉ?", [_chunk("a"), _chunk("b")])
    user = messages[1]["content"]
    assert messages[0]["role"] == "system" and REFUSAL in messages[0]["content"]
    assert (
        "[1] Quy chế đào tạo — Điều 10. Đăng ký học phần (ban hành 2021-12-30; đăng 2024-11-25; Chung)"
        in user
    )
    assert "[2]" in user and user.endswith("Câu hỏi: Tối đa bao nhiêu tín chỉ?")


def test_fit_budget_keeps_rank_order_and_first_chunk():
    chunks = [_chunk("a", "x " * 900), _chunk("b", "y " * 700), _chunk("c", "z " * 500)]
    assert [c["chunk_id"] for c in fit_budget(chunks, 1500)] == ["a", "c"]
    assert [c["chunk_id"] for c in fit_budget([_chunk("big", "w " * 3000)], 1500)] == ["big"]


def test_pipeline_answers_with_cited_sources():
    llm = FakeLLM(answer="Tối đa 25 tín chỉ [2].")
    pipeline = RAGPipeline(FakeRetriever([_chunk("a"), _chunk("b")]), llm, GenerationSettings(top_k=5))
    result = pipeline.answer("Tối đa bao nhiêu tín chỉ?")
    assert [s["chunk_id"] for s in result.cited_sources] == ["b"]
    assert result.to_dict()["cited_chunk_ids"] == ["b"]
    assert len(llm.calls) == 1


def test_pipeline_refuses_without_calling_llm_when_nothing_retrieved():
    llm = FakeLLM()
    result = RAGPipeline(FakeRetriever([]), llm, GenerationSettings()).answer("?")
    assert result.answer == REFUSAL and result.citations.refused
    assert llm.calls == []


def test_streaming_yields_full_answer():
    pipeline = RAGPipeline(FakeRetriever([_chunk("a")]), FakeLLM(answer="Có [1] nhé."), GenerationSettings())
    sources, tokens = pipeline.stream("?")
    assert "".join(tokens) == "Có [1] nhé." and len(sources) == 1


def test_ollama_client(monkeypatch):
    def fake_post(url, json, timeout):
        assert url.endswith("/api/chat") and json["model"] == "qwen" and json["stream"] is False
        body = {
            "message": {"content": "Trả lời [1]."},
            "prompt_eval_count": 120,
            "eval_count": 8,
            "total_duration": 2_000_000_000,
        }
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    completion = OllamaLLM("qwen").complete([{"role": "user", "content": "?"}])
    assert completion.text == "Trả lời [1]." and completion.prompt_tokens == 120
    assert completion.duration_ms == pytest.approx(2000)


def test_ollama_errors_are_actionable(monkeypatch):
    def missing_model(url, json, timeout):
        return httpx.Response(404, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", missing_model)
    with pytest.raises(LLMUnavailableError, match="ollama pull qwen"):
        OllamaLLM("qwen").complete([])

    def down(url, json, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", down)
    with pytest.raises(LLMUnavailableError, match=re.escape("https://ollama.com")):
        OllamaLLM("qwen").complete([])


def test_score_answers():
    records = [
        {
            "relevant": True,
            "refused": False,
            "grounded": True,
            "context_hit": True,
            "supported": True,
            "cited_relevant": 1.0,
            "generation_ms": 100.0,
        },
        {
            "relevant": True,
            "refused": True,
            "grounded": True,
            "context_hit": False,
            "supported": False,
            "cited_relevant": None,
            "generation_ms": 50.0,
        },
        {
            "relevant": False,
            "refused": True,
            "grounded": True,
            "context_hit": False,
            "supported": False,
            "cited_relevant": None,
            "generation_ms": 40.0,
        },
    ]
    m = score_answers(records)
    assert m["supported"] == 0.5 and m["false_refusal"] == 0.5 and m["correct_refusal"] == 1.0
    assert m["citation_precision"] == 1.0 and m["grounded"] == 1.0
    json.dumps(m)  # serialisable


def test_ollama_retries_once_on_server_error(monkeypatch):
    responses = iter(
        [
            httpx.Response(500, text="llama-server terminated", request=httpx.Request("POST", "http://x")),
            httpx.Response(
                200, json={"message": {"content": "OK [1]."}}, request=httpx.Request("POST", "http://x")
            ),
        ]
    )
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: next(responses))
    assert OllamaLLM("qwen").complete([]).text == "OK [1]."

    failing = lambda url, json, timeout: httpx.Response(500, text="boom", request=httpx.Request("POST", url))  # noqa: E731
    monkeypatch.setattr(httpx, "post", failing)
    with pytest.raises(LLMUnavailableError, match="Ollama error 500"):
        OllamaLLM("qwen").complete([])
