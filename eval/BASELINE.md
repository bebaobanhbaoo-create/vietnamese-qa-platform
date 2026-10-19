# Retrieval baseline

Measured on 2026-10-19 with `python online/run_evaluate.py --mode all --output eval/results/baseline.json`.

| Setting | Value |
|---|---|
| Corpus | 1,209 chunks (715 regulations, 494 announcements) from 516 documents |
| Test set | 50 queries, 129 labelled relevant chunks (`online/evaluation/test_queries.json`) |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` (384-d) |
| Dense index | FAISS `IndexFlatIP` on L2-normalised vectors |
| BM25 | `rank_bm25.BM25Okapi`, lowercase + whitespace tokenisation |
| Fusion | Reciprocal Rank Fusion, k = 60, top-20 from each retriever |
| Chunking | `CHUNK_SIZE=400`, `CHUNK_OVERLAP=0`, `MIN_CHUNK_SIZE=100` (characters) |

| Retriever | P@10 | R@10 | MRR |
|---|---|---|---|
| BM25 | 0.152 | 0.640 | 0.499 |
| Dense | 0.146 | 0.608 | 0.525 |
| **Hybrid (RRF)** | **0.182** | **0.723** | **0.630** |

## Notes

- **Reproducibility fix.** `config.py` had `CHUNK_OVERLAP = 80`, but the committed index was built with overlap 0. Rebuilding with 80 changed the text of 693/1,209 chunks (character-level overlap splices the tail of the previous segment, often mid-word) and lowered Hybrid to R@10 0.669 / MRR 0.533. Setting it to 0 reproduces the committed `chunks.json` exactly and the numbers above.
- **Known weaknesses to address in phase 2:** MiniLM truncates at 128 tokens while the median chunk is 102 words (p90 461, max 4,128), so over half the chunks are only partially embedded; some chunks exceed `CHUNK_SIZE` by far (max 19,440 characters); BM25 does not use Vietnamese word segmentation.
- The test set is small (50 queries) and was labelled before this baseline; differences of a few points should not be over-interpreted until it is expanded and split into dev/test.
