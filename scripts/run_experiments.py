"""Phase 2B experiment grid: chunking x BM25 tokenizer x embedding model.

Each configuration is built into data/experiments/<name>/ and evaluated on the **dev** split only;
the chosen configuration is then evaluated once on test. Results go to eval/results/.

    python scripts/run_experiments.py                 # whole grid
    python scripts/run_experiments.py --only e5_v2ctx # one configuration
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from huflit_qa.config import Bm25Settings, ChunkingSettings, EmbeddingSettings, Settings
from huflit_qa.evaluation.retrieval_eval import format_report, run_evaluation
from huflit_qa.indexing.encoders import Encoder
from huflit_qa.log import setup_logging
from huflit_qa.pipeline import build_index
from huflit_qa.retrieval.hybrid import HybridRetriever
from huflit_qa.retrieval.reranker import CrossEncoderReranker

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "intfloat/multilingual-e5-base"
BKAI = "bkai-foundation-models/vietnamese-bi-encoder"
BGE_M3 = "BAAI/bge-m3"

V1 = {"strategy": "v1"}
V2 = {"strategy": "v2", "max_words": 200}
V2_CTX = {"strategy": "v2", "max_words": 200, "context_prefix": True}

# name -> (chunking, bm25 tokenizer, embedding model, lowercase queries)
GRID: dict[str, tuple[dict, str, str, bool]] = {
    # Baseline as shipped.
    "baseline": (V1, "whitespace", MINILM, True),
    # BM25 tokenisation (dense unchanged).
    "bm25_words": (V1, "words", MINILM, True),
    "bm25_pyvi": (V1, "pyvi", MINILM, True),
    # Chunking, with the original model.
    "minilm_v2": (V2, "pyvi", MINILM, False),
    "minilm_v2ctx": (V2_CTX, "pyvi", MINILM, False),
    # Embedding models on both chunkings.
    "minilm_v1": (V1, "pyvi", MINILM, False),
    "e5_v1": (V1, "pyvi", E5, False),
    "e5_v2ctx": (V2_CTX, "pyvi", E5, False),
    "bkai_v1": (V1, "pyvi", BKAI, False),
    "bkai_v2ctx": (V2_CTX, "pyvi", BKAI, False),
    "bgem3_v1": (V1, "pyvi", BGE_M3, False),
    "bgem3_v2ctx": (V2_CTX, "pyvi", BGE_M3, False),
    # Chunk size, with the strongest long-context model.
    "bgem3_v2ctx_w120": ({**V2_CTX, "max_words": 120}, "pyvi", BGE_M3, False),
    "bgem3_v2ctx_w300": ({**V2_CTX, "max_words": 300}, "pyvi", BGE_M3, False),
}


def settings_for(name: str) -> Settings:
    chunking, tokenizer, model, lowercase = GRID[name]
    return Settings(
        index_path=Path("data/experiments") / name,
        chunking=ChunkingSettings(**chunking),
        bm25=Bm25Settings(tokenizer=tokenizer),  # type: ignore[arg-type]
        embedding=EmbeddingSettings(model=model, lowercase_query=lowercase),
    )


def summarize(reports: dict, baseline: str = "baseline") -> str:
    """One row per configuration: nDCG@10 per retriever and the Hybrid gain over the baseline."""
    from huflit_qa.evaluation.retrieval_eval import paired_bootstrap

    base = reports.get(baseline, {}).get("results", {}).get("Hybrid", {}).get("per_query_nDCG@10")
    lines = [
        "| Config | BM25 | Dense | Hybrid | +rerank | Δ best vs baseline [95% CI] | R@20 | words/chunk |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, report in reports.items():
        r = report["results"]
        best = r.get("Hybrid+rerank") or r["Hybrid"]
        delta = ""
        if base and name != baseline:
            qids = list(base)
            cmp = paired_bootstrap([base[q] for q in qids], [best["per_query_nDCG@10"][q] for q in qids])
            delta = f"{cmp['diff']:+.3f} [{cmp['ci_low']:+.3f}, {cmp['ci_high']:+.3f}]"
        reranked = f"{r['Hybrid+rerank']['nDCG@10']:.3f}" if "Hybrid+rerank" in r else ""
        lines.append(
            f"| {name} | {r['BM25']['nDCG@10']:.3f} | {r['Dense']['nDCG@10']:.3f} | "
            f"{r['Hybrid']['nDCG@10']:.3f} | {reranked} | {delta} | {r['Hybrid']['R@20']:.3f} | "
            f"{r['Hybrid']['words_per_chunk@10']:.0f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="Configuration names to run")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild indexes that already exist")
    parser.add_argument("--summary", type=Path, help="Only print the summary of a saved results file")
    parser.add_argument("--rerank", help="Cross-encoder to rerank hybrid results with (adds '+<name>' rows)")
    args = parser.parse_args()
    setup_logging()
    if args.summary:
        print(summarize(json.loads(args.summary.read_text(encoding="utf-8"))))
        return

    names = args.only or list(GRID)
    reranker = CrossEncoderReranker(args.rerank) if args.rerank else None
    suffix = f"+{args.rerank.split('/')[-1]}" if args.rerank else ""
    encoders: dict[str, Encoder] = {}
    reports = {}
    # Group by model so each encoder is loaded once.
    for name in sorted(names, key=lambda n: GRID[n][2]):
        settings = settings_for(name)
        emb = settings.embedding
        if emb.model not in encoders:
            encoders.clear()  # keep one model in GPU memory at a time
            encoders[emb.model] = Encoder(emb.model, emb.max_seq_length, emb.batch_size)
        encoder = encoders[emb.model]
        if args.rebuild or not settings.manifest_path.exists():
            build_index(settings, encoder)
        report = run_evaluation(
            settings,
            ("bm25", "dense", "hybrid", "rerank"),
            split=args.split,
            retriever=HybridRetriever(settings, encoder, reranker),
        )
        reports[name + suffix] = report
        print(f"\n## {name + suffix}\n\n{format_report(report)}\n", flush=True)

    out = Path("eval/results") / f"{date.today()}_2b_{args.split}.json"
    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    merged = {**previous, **reports}
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    logging.getLogger(__name__).info("Saved -> %s", out)
    print(summarize(merged))


if __name__ == "__main__":
    main()
