# Evaluation

## Test set

`eval/testset.jsonl`, one query per line:

| Field | Meaning |
|---|---|
| `id` | `q001`, `q002`, … (never reused) |
| `query` | Question in Vietnamese, as a student would type it |
| `relevant_chunk_ids` | Chunks that answer the question (from `data/processed/chunks.json`) |
| `labelled` | `true` once a human has judged the pooled candidates |
| `split` | `dev` (tuning) or `test` (reporting only), from a hash of the id |
| `style` | `keyword` (search-box style) or `question` (natural question) |
| `source` | `v1` (original 50 queries) or `draft-v2` (drafted for this round) |

Queries labelled with no relevant chunk have no answer in the corpus. Retrieval metrics skip them; they are kept to test answer refusal later. Unlabelled queries are skipped too.

**Splits.** About 40% of ids hash to `dev`, the rest to `test`. Adding queries never moves existing ones. Tune parameters (fusion weights, chunk size, model choice) on `dev`; report `test`. While only the 50 v1 queries are labelled, reports use all of them.

## Labelling

Labelling only what the current system returns would bias the test set towards it, so candidates are **pooled**: the top 10 of BM25, dense and hybrid search, plus any existing labels. A new retriever can be added to the pool later.

```bash
huflit-qa pool                                  # -> eval/annotation/pool.json
# open eval/annotation/annotate.html in a browser, load pool.json, label, export labels.json
huflit-qa merge-labels labels.json              # -> eval/testset.jsonl
```

The annotation page keeps progress in the browser, so labelling can stop and resume. Keys: `J`/`K` move, `Space` toggles relevant, `E` expands text, `Enter` finishes the query, `N` means no answer in the corpus.

**Guidelines**

- A chunk is **relevant** if a student reading it would get the answer, or a necessary part of it. Mentioning the topic is not enough.
- Several chunks can be relevant, for example the same rule in several versions of a regulation, or an announcement plus the regulation it applies. Mark all of them.
- For time-bound questions ("lịch thi học kỳ 1 năm 2025-2026"), only the announcement for that period is relevant.
- If nothing in the pool answers the question, press `N`, even if you suspect the answer exists elsewhere. Add a note in the commit instead.

## Comparing chunkings fairly

Labels point at **label units**: the chunks of the original (v1) chunking, frozen in `eval/label_units.json`. Re-chunking would otherwise invalidate every label. A chunk from any other chunking is projected onto the units it covers: same source record, and at least 50% of the unit's word 3-grams inside the chunk or at least 50% of the chunk's 3-grams from the unit ([projection.py](../src/huflit_qa/evaluation/projection.py)). Every configuration is therefore scored against the same relevant set.

- Recall counts relevant units covered by the top-k chunks.
- nDCG, precision and MRR count a chunk as a hit if it covers a relevant unit not covered higher up.
- With the v1 chunking the projection is the identity, so the numbers equal plain chunk-level scoring.
- Long chunks can cover several units, so the report also shows **words per retrieved chunk**: a gain bought with much longer chunks costs prompt space later.

## Significance

The dev split has 30 queries, so differences of a few points can be noise. `scripts/run_experiments.py` reports each configuration's Hybrid nDCG@10 gain over the baseline with a 95% **paired bootstrap** confidence interval (10,000 resamples over queries). A gain whose interval includes 0 is not treated as established.

## Metrics

Binary relevance, computed by `huflit-qa eval` ([metrics.py](../src/huflit_qa/evaluation/metrics.py)):

| Metric | What it says |
|---|---|
| nDCG@10 | Ranking quality of the first page; rewards relevant chunks near the top. **Primary metric.** |
| R@10, R@20 | Share of relevant chunks retrieved; R@20 bounds what a reranker on top-20 can reach |
| Hit@10 | Share of queries with at least one relevant chunk in the top 10 (what a RAG prompt needs) |
| P@10 | Precision of the first page (low by construction: most queries have 1–5 relevant chunks) |
| MRR@10 | Reciprocal rank of the first relevant chunk |
| p50 / p95 ms | Per-query search latency after models are loaded (query encoding + search + fusion) |

```bash
huflit-qa eval                        # all labelled queries
huflit-qa eval --split dev            # while tuning
huflit-qa eval --split test --output eval/results/<date>_<change>.json
```

## Current results

See [eval/BASELINE.md](../eval/BASELINE.md).
