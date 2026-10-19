"""Answer-level evaluation that needs no judge model.

Citations make answers checkable against the relevance labels:
- grounded: the answer refuses, or cites at least one shown source and no non-existent one;
- citation precision: share of cited sources that cover a relevant label unit;
- supported: the answer cites at least one relevant source (answerable queries);
- context hit: some shown source is relevant, i.e. the best a generator could do with this retrieval;
- refusals: wrong on answerable queries, right on queries labelled as having no answer.

Faithfulness (is every sentence backed by the cited text?) still needs reading the answers; the run
is saved as JSONL for that manual review.
"""

import json
from pathlib import Path

import numpy as np

from huflit_qa.config import Settings
from huflit_qa.evaluation.projection import build_projection, load_units
from huflit_qa.evaluation.testset import load_testset
from huflit_qa.generation.rag import RAGPipeline


def select_for_answers(queries: list[dict], split: str | None) -> list[dict]:
    """Labelled queries of a split, including those labelled as having no answer."""
    return [q for q in queries if q.get("labelled") and (split is None or q["split"] == split)]


def score_answers(records: list[dict]) -> dict:
    answerable = [r for r in records if r["relevant"]]
    unanswerable = [r for r in records if not r["relevant"]]
    cited = [r for r in answerable if r["cited_relevant"] is not None]

    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    gen_ms = [r["generation_ms"] for r in records if r["generation_ms"]]
    return {
        "num_queries": len(records),
        "num_answerable": len(answerable),
        "num_unanswerable": len(unanswerable),
        "grounded": mean([r["grounded"] for r in records]),
        "context_hit": mean([r["context_hit"] for r in answerable]),
        "supported": mean([r["supported"] for r in answerable]),
        "citation_precision": mean([r["cited_relevant"] for r in cited]),
        "false_refusal": mean([r["refused"] for r in answerable]),
        "correct_refusal": mean([r["refused"] for r in unanswerable]),
        "generation_ms_p50": float(np.percentile(gen_ms, 50)) if gen_ms else None,
        "generation_ms_p95": float(np.percentile(gen_ms, 95)) if gen_ms else None,
    }


def run_answer_eval(
    settings: Settings,
    pipeline: RAGPipeline,
    split: str | None = None,
    limit: int | None = None,
    answers_path: Path | None = None,
) -> dict:
    queries = select_for_answers(load_testset(settings.testset_path), split)[:limit]
    projection = build_projection(
        list(pipeline.retriever.chunk_map.values()), load_units(settings.label_units_path)
    )

    records = []
    for q in queries:
        result = pipeline.answer(q["query"])
        relevant = set(q["relevant_chunk_ids"])

        def is_relevant(chunk_id: str, relevant: set[str] = relevant) -> bool:
            return bool(set(projection.get(chunk_id, ())) & relevant)

        shown = [s["chunk_id"] for s in result.sources]
        cited = [s["chunk_id"] for s in result.cited_sources]
        records.append(
            {
                "id": q["id"],
                "query": q["query"],
                "answer": result.answer,
                "relevant": bool(relevant),
                "refused": result.citations.refused,
                "grounded": result.citations.grounded,
                "invalid_citations": result.citations.invalid,
                "shown": shown,
                "cited": cited,
                "context_hit": any(is_relevant(c) for c in shown),
                "supported": any(is_relevant(c) for c in cited),
                "cited_relevant": (sum(is_relevant(c) for c in cited) / len(cited)) if cited else None,
                "retrieval_ms": result.retrieval_ms,
                "generation_ms": result.generation_ms,
                "completion_tokens": result.completion_tokens,
            }
        )

    if answers_path:
        answers_path.parent.mkdir(parents=True, exist_ok=True)
        with open(answers_path, "w", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "split": split or "all",
        "model": pipeline.llm.model,
        "retriever": pipeline.retriever.manifest,
        "rerank": pipeline.retriever.rerank_cfg.model if pipeline.retriever.reranker else None,
        "metrics": score_answers(records),
    }


def format_answer_report(report: dict) -> str:
    m = report["metrics"]

    def pct(x: float | None) -> str:
        return "—" if x is None else f"{100 * x:.1f}%"

    lines = [
        f"Model {report['model']} · split {report['split']} · {m['num_queries']} queries "
        f"({m['num_answerable']} answerable, {m['num_unanswerable']} without answer)",
        f"  grounded            {pct(m['grounded'])}",
        f"  context hit         {pct(m['context_hit'])}   (some shown source is relevant)",
        f"  supported           {pct(m['supported'])}   (cites a relevant source)",
        f"  citation precision  {pct(m['citation_precision'])}",
        f"  false refusal       {pct(m['false_refusal'])}",
        f"  correct refusal     {pct(m['correct_refusal'])}",
    ]
    if m["generation_ms_p50"] is not None:
        lines.append(f"  generation p50/p95  {m['generation_ms_p50']:.0f} / {m['generation_ms_p95']:.0f} ms")
    return "\n".join(lines)
