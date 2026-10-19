# HUFLIT Regulation QA

Hybrid retrieval over the academic regulations and announcements published on the HUFLIT student portal (Ho Chi Minh City University of Foreign Languages – Information Technology). Ask a question in Vietnamese, get the most relevant regulation articles back with source links.

> Status: work in progress. The retrieval pipeline below works end to end; answer generation, an API and a demo are planned (see [docs/UPGRADE_PLAN.md](docs/UPGRADE_PLAN.md)).

## Pipeline

```
portal.huflit.edu.vn ──► scrape ──► process ──► chunk ──► BM25 + FAISS ──► RRF ──► top-k chunks
   (HTML + PDF)        offline/     offline/    offline/   offline/          online/
                       scraper      scripts     ir_build   ir_build          retrieval
```

| Step | Script | Output |
|---|---|---|
| 1. Scrape announcements, regulations and attachments | `offline/run_scrape.py` | `data/raw/`, `data/attachments/` |
| 2. Classify and flatten into records | `offline/run_process.py` | `data/processed/by_doc_type/*_flat.json` |
| 3. Chunk by article/clause, build BM25 and FAISS indexes | `offline/run_index.py` | `data/processed/chunks.json`, `data/index/` |
| 4. Query | `online/run_query.py` | ranked chunks |
| 5. Evaluate | `online/run_evaluate.py` | P@k, R@k, MRR |

## Data

1,209 chunks from 516 documents (715 regulation chunks, 494 announcement chunks), posted between 07/2020 and 03/2026. Source documents are public pages and attachments from the HUFLIT student portal; all rights remain with the university.

## Results

50 hand-labelled queries, k = 10 ([details](eval/BASELINE.md)):

| Retriever | P@10 | R@10 | MRR |
|---|---|---|---|
| BM25 | 0.152 | 0.640 | 0.499 |
| Dense (multilingual MiniLM) | 0.146 | 0.608 | 0.525 |
| **Hybrid (RRF)** | **0.182** | **0.723** | **0.630** |

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python online/run_query.py "điều kiện xét tốt nghiệp" --top-k 5
python online/run_evaluate.py --mode all
```

The repository ships with a prebuilt index; run `python offline/run_index.py` to rebuild it.

## Tests

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).
