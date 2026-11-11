"""Retrieval-augmented answering: retrieve -> build a cited prompt -> generate -> check citations."""

import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass

from huflit_qa.config import GenerationSettings
from huflit_qa.generation.citations import CitationCheck, check_citations
from huflit_qa.generation.llm import LLM
from huflit_qa.generation.prompts import REFUSAL, build_messages
from huflit_qa.retrieval.filters import SearchFilters
from huflit_qa.retrieval.hybrid import HybridRetriever


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[dict]  # what the model was shown, numbered from 1
    citations: CitationCheck
    retrieval_ms: float
    generation_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def cited_sources(self) -> list[dict]:
        return [self.sources[n - 1] for n in self.citations.cited]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["citations"]["grounded"] = self.citations.grounded
        data["cited_chunk_ids"] = [s["chunk_id"] for s in self.cited_sources]
        return data


def fit_budget(chunks: list[dict], max_words: int) -> list[dict]:
    """Keep chunks in rank order while the total stays under ``max_words`` (always keep the first)."""
    kept: list[dict] = []
    total = 0
    for chunk in chunks:
        words = len(chunk["text"].split())
        if kept and total + words > max_words:
            continue
        kept.append(chunk)
        total += words
    return kept


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm: LLM, cfg: GenerationSettings):
        self.retriever = retriever
        self.llm = llm
        self.cfg = cfg

    def _retrieve(self, question: str, filters: SearchFilters | None) -> tuple[list[dict], float]:
        start = time.perf_counter()
        chunks = self.retriever.retrieve(question, top_k=self.cfg.top_k, filters=filters)
        return fit_budget(chunks, self.cfg.max_context_words), (time.perf_counter() - start) * 1000

    def answer(self, question: str, filters: SearchFilters | None = None) -> RAGAnswer:
        sources, retrieval_ms = self._retrieve(question, filters)
        if not sources:
            return RAGAnswer(question, REFUSAL, [], check_citations(REFUSAL, 0), retrieval_ms, 0.0)

        start = time.perf_counter()
        completion = self.llm.complete(build_messages(question, sources))
        generation_ms = (time.perf_counter() - start) * 1000
        text = completion.text.strip()
        return RAGAnswer(
            question=question,
            answer=text,
            sources=sources,
            citations=check_citations(text, len(sources)),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
        )

    def stream(self, question: str, filters: SearchFilters | None = None) -> tuple[list[dict], Iterator[str]]:
        """Sources first (so a UI can show them immediately), then the answer as it is generated."""
        sources, _ = self._retrieve(question, filters)
        if not sources:
            return [], iter([REFUSAL])
        return sources, self.llm.stream(build_messages(question, sources))
