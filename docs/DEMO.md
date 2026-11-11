# Public demo: search in the browser

The public demo ([GitHub Pages](https://bebaobanhbaoo-create.github.io/vietnamese-qa-platform/)) runs the baseline hybrid retriever without a server:

| Part | Python (`huflit-qa`) | Browser ([deploy/static](../deploy/static)) |
|---|---|---|
| Chunks, metadata | `data/index/chunks.json` | exported by `scripts/build_static_demo.py` |
| BM25 | `rank_bm25.BM25Okapi`, lowercase + whitespace tokens | JavaScript port of the same formula (k1 1.5, b 0.75, ε 0.25), stable sort |
| Passage vectors | FAISS `IndexFlatIP` | the same float32 vectors, read back from `faiss.bin` |
| Query vector | `paraphrase-multilingual-MiniLM-L12-v2`, PyTorch | `Xenova/paraphrase-multilingual-MiniLM-L12-v2`, ONNX 8-bit, transformers.js 3.8.0 |
| Fusion | RRF, k = 60, top 20 per retriever | same |

## Parity with the Python retriever (159 test queries, top 10)

| | Same order | Mean overlap | Same top-1 |
|---|---|---|---|
| BM25 | 159 / 159 | 1.000 | 159 / 159 |
| Dense, ONNX 8-bit | 4 / 159 | 0.906 | 139 / 159 |
| Dense, ONNX fp32 (first 60 queries) | 50 / 60 | 0.998 | 58 / 60 |
| Hybrid, ONNX 8-bit | 21 / 159 | 0.933 | 150 / 159 |

The fp32 run shows the dense differences come from 8-bit quantisation of the query encoder, not from the port. Effect on quality, 50 labelled queries:

| | nDCG@10 | Hit@10 |
|---|---|---|
| Python | 0.621 | 0.840 |
| Browser, 8-bit | 0.607 | 0.840 |

The page uses the 8-bit model (118 MB, cached after the first visit) rather than fp32 (470 MB). Until it has loaded, the page searches with BM25 alone and says so.
