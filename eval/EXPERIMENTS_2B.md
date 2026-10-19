# Phase 2B: chunking, BM25 tokenisation, embedding models

> **Interim.** Run on the 30 dev queries of the original (v1) labels. These labels turned out to be incomplete (see below), so the table is a sanity check, not a decision. The decision is taken after re-labelling on the pooled candidates.

Reproduce: `python scripts/run_experiments.py` (results in `eval/results/2026-10-19_2b_dev.json`).

## Configurations

| Name | Chunking | BM25 tokenizer | Embedding model |
|---|---|---|---|
| baseline | v1 | whitespace | paraphrase-multilingual-MiniLM-L12-v2 (128 tokens) |
| bm25_words / bm25_pyvi | v1 | words / pyvi | MiniLM |
| minilm_v1, minilm_v2, minilm_v2ctx | v1 / v2 / v2 + context | pyvi | MiniLM |
| e5_* | v1 / v2 + context | pyvi | intfloat/multilingual-e5-base (512) |
| bkai_* | v1 / v2 + context | pyvi | bkai-foundation-models/vietnamese-bi-encoder (256, pyvi input) |
| bgem3_* | v1 / v2 + context (120, 200, 300 words) | pyvi | BAAI/bge-m3 (512 used) |

## Results (dev, 30 queries, v1 labels)

nDCG@10 per retriever; Δ is Hybrid vs baseline with a 95% paired-bootstrap CI.

| Config | BM25 | Dense | Hybrid | Δ Hybrid [95% CI] | Hybrid R@20 | Hybrid Hit@10 | words/chunk |
|---|---|---|---|---|---|---|---|
| baseline | 0.513 | 0.480 | **0.605** | | 0.792 | 0.833 | 206 |
| bm25_words | 0.467 | 0.480 | 0.596 | −0.009 [−0.026, +0.006] | 0.792 | 0.800 | 192 |
| bm25_pyvi | 0.467 | 0.480 | 0.574 | −0.031 [−0.072, +0.013] | 0.764 | 0.800 | 207 |
| minilm_v2 | 0.484 | 0.398 | 0.474 | −0.132 [−0.234, −0.032] | 0.751 | 0.767 | 150 |
| minilm_v2ctx | 0.503 | 0.345 | 0.471 | −0.134 [−0.265, −0.005] | 0.788 | 0.833 | 151 |
| e5_v2ctx | 0.503 | 0.470 | 0.519 | −0.087 [−0.218, +0.042] | 0.863 | 0.867 | 149 |
| bkai_v2ctx | 0.503 | 0.419 | 0.510 | −0.095 [−0.223, +0.033] | 0.813 | 0.933 | 156 |
| bgem3_v1 | 0.467 | **0.585** | 0.559 | −0.046 [−0.151, +0.052] | 0.794 | 0.900 | 259 |
| bgem3_v2ctx | 0.503 | 0.537 | 0.544 | −0.061 [−0.190, +0.065] | 0.824 | 0.900 | 154 |
| bgem3_v2ctx_w300 | 0.496 | 0.554 | 0.531 | −0.074 [−0.195, +0.053] | **0.863** | **0.967** | 216 |

## Observations

1. **The v1 labels are incomplete.** Stronger models lose nDCG on queries where they return relevant but unlabelled chunks. Examples: for *"quy định điểm số xét học bổng"*, e5 ranks the scholarship regulation's eligibility article first, scored 0; for *"lịch thi học kỳ"*, it ranks an exam-schedule announcement first, scored 0. Pooling from the new systems adds **1,413 candidates** (+48%) that the original pool never showed an annotator.
2. **bge-m3 is the strongest single retriever** even on these labels: dense nDCG@10 0.585 vs 0.480 for MiniLM, Hit@10 0.933 vs 0.800.
3. **MiniLM cannot use v2 chunks.** It truncates at 128 tokens; v2 chunks are about 150 words, and the context prefix uses part of that window. Dense nDCG drops from 0.480 to 0.345. Chunking and model must be chosen together.
4. **Fusing with BM25 hurts bge-m3** (hybrid 0.559 < dense 0.585). Equal-weight RRF with a much weaker BM25 drags the ranking down; fusion weights (2.14) need tuning once labels are fixed.
5. **pyvi word segmentation does not help BM25 here** (0.467 vs 0.513 for whitespace on dev). This may also be label bias, since the original labels were probably produced with whitespace BM25. Re-check after re-labelling.
6. Bigger v2 chunks (300 words) give the best recall (R@20 0.863, Hit@10 0.967) at the cost of 40% more words per retrieved chunk than 200 words.

## Next

1. Re-label all 159 queries on `eval/annotation/pool.json` (depth 7 from baseline, e5_v2ctx, bkai_v2ctx, bgem3_v2ctx: 3,724 candidates).
2. Re-run the grid on the new dev labels; tune RRF weights on dev.
3. Pick one configuration and report it once on test.
