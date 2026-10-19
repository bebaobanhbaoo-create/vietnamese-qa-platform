# HUFLIT Regulation QA

Hybrid retrieval over the academic regulations and announcements published on the HUFLIT student portal (Ho Chi Minh City University of Foreign Languages – Information Technology). Ask a question in Vietnamese, get the most relevant regulation articles back with source links.

> Status: work in progress. The retrieval pipeline below works end to end; answer generation, an API and a demo are planned (see [docs/UPGRADE_PLAN.md](docs/UPGRADE_PLAN.md)).

## Pipeline

```
portal.huflit.edu.vn ──► scrape ──► process ──► chunk ──► BM25 + FAISS ──► RRF ──► top-k chunks
   (HTML + PDF)          ingest/    processing/  processing/  indexing/      retrieval/
```

| Step | Command | Output |
|---|---|---|
| 1. Scrape announcements, regulations and attachment links | `huflit-qa scrape` | `data/raw/`, `data/processed/by_doc_type/` |
| 1b. Normalise downloaded attachments, extract PDF text | `huflit-qa normalize-attachments`, `huflit-qa extract-pdf` | `data/attachments/**.txt` |
| 2. Flatten into records (one per web page or attachment) | `huflit-qa process` | `data/processed/by_doc_type/*_flat.json` |
| 3. Chunk by article/clause, build BM25 and FAISS indexes | `huflit-qa index` | `data/index/` (chunks, BM25, FAISS, `manifest.json`) |
| 4. Query | `huflit-qa query "..."` | ranked chunks |
| 5. Evaluate | `huflit-qa eval` | nDCG, Recall, Hit, MRR, latency |

Code lives in `src/huflit_qa/`:

```
ingest/       crawler, HTML parser, classification, attachment and PDF handling
processing/   flat records, hand-corrected text sync, Điều/Khoản-aware chunkers (v1, v2)
indexing/     BM25 (rank_bm25, optional pyvi word segmentation), dense encoders, FAISS or Qdrant (incremental), manifest
retrieval/    hybrid retriever, weighted Reciprocal Rank Fusion, cross-encoder reranker, metadata filters
evaluation/   P@k, R@k, MRR and the evaluation runner
config.py     settings (pydantic-settings, overridable via HUFLIT_* env vars or .env)
cli.py        `huflit-qa` command
```

## Data

1,209 chunks from 516 documents (715 regulation chunks, 494 announcement chunks), posted between 07/2020 and 03/2026. Source documents are public pages and attachments from the HUFLIT student portal; all rights remain with the university.

## Results

50 hand-labelled queries, k = 10 ([details](eval/BASELINE.md), [methodology](docs/EVALUATION.md)):

| Retriever | nDCG@10 | R@10 | Hit@10 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.485 | 0.640 | 0.800 | 0.499 |
| Dense (multilingual MiniLM) | 0.490 | 0.608 | 0.780 | 0.525 |
| **Hybrid (RRF)** | **0.621** | **0.723** | **0.840** | **0.630** |

The test set is being expanded to ~160 queries with pooled relevance judgements and a dev/test split.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"

huflit-qa query "điều kiện xét tốt nghiệp" -k 5
huflit-qa query "chuẩn đầu ra ngoại ngữ" --doc-type quy_che --newest-only --rerank BAAI/bge-reranker-v2-m3
huflit-qa eval
```

The repository ships with a prebuilt index; run `huflit-qa index` to rebuild it. Settings are documented in [.env.example](.env.example).

## Development

```bash
make check      # ruff lint + format check, mypy, pytest
make format     # auto-fix lint and formatting
```

## License

MIT — see [LICENSE](LICENSE).
