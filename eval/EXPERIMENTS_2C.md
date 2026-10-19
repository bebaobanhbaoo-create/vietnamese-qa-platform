# Phase 2C: reranking, metadata filters, Qdrant

> **Interim**, like [2B](EXPERIMENTS_2B.md): measured on the 30 dev queries of the original labels, which miss relevant chunks that newer systems find. Re-run after re-labelling.

Reproduce: `python scripts/run_experiments.py --only baseline bgem3_v1 bgem3_v2ctx bgem3_v2ctx_w300 --rerank <model>`.

## Cross-encoder reranking

The cross-encoder rescores the top 20 fused candidates (so R@20 of hybrid search bounds it). nDCG@10 / MRR@10 on dev; latency is p95 per query on an RTX 4050, fp32, batch 16.

| Config | Hybrid nDCG | + PhoRanker | + bge-reranker-v2-m3 | MRR: hybrid → bge-reranker | p95 ms (bge-reranker) |
|---|---|---|---|---|---|
| baseline (v1, MiniLM) | 0.605 | 0.588 | 0.606 | 0.635 → 0.706 | 1,493 |
| bgem3_v1 | 0.559 | 0.576 | 0.606 | 0.619 → 0.698 | 1,490 |
| **bgem3_v2ctx** | 0.544 | 0.539 | **0.618** | 0.605 → **0.723** | 1,197 |
| bgem3_v2ctx_w300 | 0.531 | 0.553 | 0.590 | 0.634 → 0.697 | 1,419 |

- **bge-reranker-v2-m3** lifts every configuration's MRR by 0.06–0.12, and bgem3_v2ctx + reranker is the only configuration above the baseline's nDCG@10. The 95% CI of its gain over the baseline is still wide, [−0.126, +0.156], with 30 queries.
- **PhoRanker** (Vietnamese, PhoBERT, 256 tokens) does not help consistently: ±0.02 nDCG.
- **Cost:** ~1.2–1.5 s p95 per query, versus ~50 ms without reranking. Options before serving: rerank 10 instead of 20 candidates, fp16, or a cap of 256 tokens per passage. Each needs measuring.

## Metadata filters

Every chunk carries `doc_type`, `issuer_unit`, `date_posted`, `doc_group` (portal article), `issued_date` (parsed from the regulation header, e.g. "ngày 22 tháng 5 năm 2025") and `newest_in_group`.

```bash
huflit-qa query "chuẩn đầu ra ngoại ngữ" --doc-type quy_che --newest-only
huflit-qa query "lịch thi" --doc-type thong_bao --issuer "Phòng Khảo thí" --year-from 2025
```

`newest_in_group` addresses the version problem: the language and IT exit requirements (group 3209) exist as 2017, 2020, 2024 and 2025 documents. It is a hint, not a validity flag. Some documents in a group are parallel rules for different cohorts, so filtering is opt-in and the answer-generation step will receive issue dates explicitly.

## Qdrant

`HUFLIT_INDEX__VECTOR_STORE=qdrant huflit-qa index` builds an embedded Qdrant collection (no server) instead of FAISS. Checked on bgem3_v2ctx (1,702 chunks):

| Check | Result |
|---|---|
| Dense top-10 vs FAISS, no filter | same order on 157/159 queries (other 2: ties), same set on 159/159 |
| Dense top-10 vs FAISS + filters (3 filter combinations) | same set on 159/159 queries each |
| Full build | 79.7 s (embedding dominates) |
| Rebuild, nothing changed | 0.1 s sync, 0 chunks embedded |
| One chunk's metadata changed | payload rewritten, 0 chunks embedded |
| One chunk's text changed | 1 chunk embedded |

Qdrant is not faster than exact FAISS at this size. It is used because it filters inside the vector search and updates incrementally when the crawler adds documents. bge-m3 sparse vectors in Qdrant (one-store hybrid) are not implemented; BM25 stays separate.
