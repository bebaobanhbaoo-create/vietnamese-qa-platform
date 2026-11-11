# Phase 3: cited answers with a local LLM

> **Interim**, like [2B](EXPERIMENTS_2B.md) and [2C](EXPERIMENTS_2C.md): 30 dev queries, original labels (incomplete, every regulation version marked relevant). Faithfulness is not measured yet: it needs the manual review below.

## Setup

| | |
|---|---|
| Generator | `qwen2.5:7b-instruct` (Q4, 4.7 GB) through Ollama 0.40.16, temperature 0.1, context 4,096 tokens |
| Hardware | RTX 4050 Laptop (6 GB): 25 of 29 layers on GPU, the rest on CPU |
| Context | top 5 retrieved chunks, at most 1,500 words |
| Prompt | answer only from the numbered sources, cite `[n]`, prefer the newest regulation, fixed refusal sentence ([prompts.py](../src/huflit_qa/generation/prompts.py)) |

Reproduce: `huflit-qa eval-answers --split dev --output eval/results/answers_dev_<name>.json` (with `HUFLIT_INDEX_PATH=data/experiments/bgem3_v2ctx` for the bge-m3 rows).

## Results (dev, 30 queries)

| Retrieval | grounded | context hit | supported | citation precision | false refusal | generation p50 / p95 |
|---|---|---|---|---|---|---|
| baseline: MiniLM, chunker v1 | 100% | 76.7% | **63.3%** | **63.8%** | 13.3% | 6.6 / 20.5 s |
| bge-m3, chunker v2 + context | 100% | **83.3%** | 60.0% | 49.7% | **0.0%** | 7.7 / 19.0 s |
| bge-m3, chunker v2 + context, newest regulations only | 96.7% | 66.7% | 50.0% | 48.5% | 6.7% | 8.7 / 28.7 s |

Definitions are in [docs/EVALUATION.md](../docs/EVALUATION.md#answers).

## Observations

1. **Citations are well formed.** Every answer either refuses or cites only sources it was shown (one exception with the newest-only filter). This is the model following the format; it says nothing about whether the cited text supports the claim.
2. **A cited answer can still be wrong.** Asked for the maximum credits per semester, the baseline answered "tối đa 14 tín chỉ [1]". Source [1] is the 2007 regulation and says 14 credits is the *minimum*. Two errors compound: retrieval surfaced a superseded regulation, and the generator turned "tối thiểu" into "tối đa". With bge-m3 and the newest-only filter the answer is correct: "tối đa 3/2 khối lượng trung bình một học kỳ … [1]" (HUFLIT regulation of 2021, Điều 9).
3. **bge-m3 retrieval removes false refusals** (13.3% → 0%) and raises context hit (76.7% → 83.3%). Its lower citation precision has the same likely cause as in 2B: its sources are often relevant but unlabelled.
4. **The labels cannot score the newest-only filter.** They mark every version of a rule as relevant, so dropping superseded versions looks like lost recall. Measuring version correctness needs a label for which version applies; it is part of the re-labelling round.
5. **Latency is dominated by generation** on 6 GB of VRAM: warm prompt processing runs at ~1,600 tokens/s and decoding at ~24 tokens/s, but long answers and partial CPU offload push p95 to 20–30 s.

## Operational notes

- Ollama's CUDA 13 runner crashed on the first model load (`CUDA error: shared object initialization failed`); it fell back to another runner and later loads worked. The client surfaces such failures as HTTP 503 in the API.
- With the model resident, Ollama's `llama-server` holds ~10.5 GB of committed memory. Loading bge-m3 (2.3 GB) in another process then crashed with an access violation on this 16 GB machine. Unloading the model first (`keep_alive: 0`) fixes it. The API loads its retriever at startup, before any LLM call, which avoids the problem unless the model is already resident from another process.

## Faithfulness review (pending)

```bash
python scripts/faithfulness_sheet.py eval/results/answers_dev_baseline.answers.jsonl --n 20
# fill the "faithful" column (1 / 0) in eval/results/answers_dev_baseline.faithfulness.csv, then:
python scripts/faithfulness_sheet.py eval/results/answers_dev_baseline.faithfulness.csv --score
```
