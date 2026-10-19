# Kế hoạch nâng cấp — HUFLIT Regulation QA (RAG)

> Mục tiêu: biến dự án thành portfolio chuẩn cho vị trí **AI/NLP Engineer** và **Data**.
> Nguyên tắc: không code giả, không số liệu bịa — mọi con số trong README/CV phải đo được bằng script trong repo.
> Đơn vị thời gian: 1 ngày ≈ 6 giờ làm việc.

---

## 0. Hiện trạng (đo ngày 2026-10-19)

| Hạng mục | Số liệu |
|---|---|
| Bài viết crawl từ portal | 434 |
| Văn bản sau xử lý (bài + file đính kèm) | 516 |
| Chunk | 1.209 (715 quy chế, 494 thông báo) |
| Khoảng thời gian | 07/2020 → 03/2026 |
| Bộ test retrieval | 50 câu, 129 nhãn chunk (100% khớp chunk thật) |
| Vector index | FAISS `IndexFlatIP`, 384 chiều, `paraphrase-multilingual-MiniLM-L12-v2` |

**Vấn đề đã xác nhận**

| # | Vấn đề | Vị trí |
|---|---|---|
| P1 | Thư mục `config/` che `config.py` → `run_index`, `run_query`, `run_evaluate` lỗi `ImportError` | `config/`, `config.py` |
| P2 | API, models, services là code giả (placeholder, confidence gán cứng 0.85) | `app.py`, `api/`, `models/`, `services/` |
| P3 | Test giả (`pass`, `assert True`) | `tests/test_{app,models,services}.py` |
| P4 | Module trùng lặp | `models/hybrid_retriever.py`, `evaluation/`, `data/chunker.py` |
| P5 | Thiếu `rank_bm25`, không pin phiên bản | `requirements.txt` |
| P6 | BM25 tách từ bằng `.split()` | `offline/ir_build/bm25.py` |
| P7 | >50% chunk bị cắt khi embed (MiniLM tối đa 128 token; chunk trung vị 102 từ, p90 461 từ) | `offline/ir_build/dense.py` |
| P8 | Chunker không giới hạn được độ dài (`CHUNK_SIZE=400` ký tự nhưng có chunk 19.440 ký tự) | `offline/ir_build/chunker.py` |
| P9 | Chunk không mang tên văn bản / Điều → mất ngữ cảnh | `offline/ir_build/chunker.py` |
| P10 | Quy chế nhiều phiên bản (vd. bộ 3209: 2017 → 2024) không phân biệt hiệu lực | dữ liệu |
| P11 | 28 MB PDF + index nhị phân + pickle commit thẳng vào git | `data/` |
| P12 | CI/Docker sơ sài (Python 3.9, dev server, không lint/cache) | `.github/`, `Dockerfile` |
| P13 | README mô tả sai tính năng | `README.md` |

---

## Quy trình làm việc

- Mỗi giai đoạn = 1 nhánh `phase-N/<tên>` + 1 PR vào `main`, merge sau khi bạn duyệt.
- Conventional commits có nội dung cụ thể (`fix(index): ...`, `feat(retrieval): ...`).
- Tác giả commit: `Mai Dong Anh <bebaobanhbaoo@gmail.com>`.
- Mỗi thay đổi ảnh hưởng chất lượng → chạy eval, lưu kết quả vào `eval/results/<ngày>_<tên>.json`.

---

## Giai đoạn 0 — Dọn dẹp & baseline · 1 ngày

Nhánh: `phase-0/cleanup`

| ID | Việc | File | Ước lượng |
|---|---|---|---|
| 0.1 | Xoá thư mục `config/` (không ai import), giữ `config.py` | `config/` | 15' |
| 0.2 | Xoá code giả / trùng lặp (đã grep: code thật không import) | `app.py`, `api/`, `models/`, `services/`, `utils/`, `evaluation/`, `data/{loader,qa_loader,validator,pipeline,preprocessing,chunker,pdf_parser}.py`, `data/sample.json`, `tests/test_{app,models,services}.py`, `Procfile`, `version.py`, `scripts/*.sh` | 1h |
| 0.3 | Chuyển test metrics sang `online.evaluation.metrics` | `tests/test_retrieval.py` | 15' |
| 0.4 | Viết lại `requirements.txt`: chỉ thư viện đang dùng, pin phiên bản, thêm `rank_bm25` | `requirements.txt` | 1h |
| 0.5 | Tạo venv sạch, chạy lại `run_index` → `run_evaluate --mode all` | — | 1,5h |
| 0.6 | Lưu baseline | `eval/results/baseline.json`, `eval/BASELINE.md` | 30' |
| 0.7 | README tạm: mô tả đúng những gì đang có | `README.md` | 30' |

**Xong khi:** `pytest` xanh; eval chạy được; có bảng P@10 / R@10 / MRR cho BM25, Dense, Hybrid.

---

## Giai đoạn 1 — Chuẩn hoá cấu trúc · 1,5 ngày

Nhánh: `phase-1/package`

```
src/huflit_qa/
  config.py              # pydantic-settings, .env
  ingest/                # crawler.py, parser.py, classify.py, storage.py, attachments.py
  processing/            # records.py, flatten.py, sync.py, chunker.py
  indexing/              # tokenizer.py, bm25.py, dense.py, manifest.py
  retrieval/             # hybrid.py, rrf.py
  evaluation/            # metrics.py, retrieval_eval.py
  cli.py                 # typer
tests/unit/  tests/integration/  tests/fixtures/
eval/  docs/
pyproject.toml  Makefile  .pre-commit-config.yaml  .env.example
```

| ID | Việc | Ước lượng |
|---|---|---|
| 1.1 | `pyproject.toml` (deps + extras `[dev]`, `[gpu]`), `pip install -e .` | 1h |
| 1.2 | Di chuyển code `offline/`, `online/` vào `src/huflit_qa/`, bỏ toàn bộ `sys.path.insert` | 3h |
| 1.3 | Gộp `config.py` + `offline/scraper/config.py` → `huflit_qa/config.py` (pydantic-settings) | 1h |
| 1.4 | CLI: `huflit-qa scrape | process | index | query | eval` | 2h |
| 1.5 | `Makefile`: `install`, `data`, `index`, `eval`, `test`, `lint` | 30' |
| 1.6 | `ruff` + `mypy` + pre-commit; sửa lỗi lint; type hints cho phần lõi | 2h |
| 1.7 | Logging chuẩn (thay `print`), xử lý UTF-8 Windows một chỗ | 30' |

**Xong khi:** `make index && make eval` cho **đúng** số baseline của GĐ0 (refactor không đổi kết quả).

---

## Giai đoạn 2 — Chất lượng retrieval · 5–6 ngày *(trọng tâm CV)*

Nhánh: `phase-2a/eval`, `phase-2b/chunk-embed`, `phase-2c/rerank-qdrant` (3 PR)

### 2A. Nâng cấp bộ đánh giá · 1,5 ngày
| ID | Việc | File |
|---|---|---|
| 2.1 | Thêm nDCG@k, Hit@k, Recall@{5,10,20}; đo latency p50/p95 | `evaluation/metrics.py` |
| 2.2 | Sinh câu hỏi ứng viên (LLM) + **pooling** ứng viên từ nhiều hệ thống (BM25, dense, bge-m3) để gán nhãn, tránh thiên vị hệ thống hiện tại | `scripts/build_testset.py` |
| 2.3 | **Bạn duyệt nhãn** (~3–4h): mở rộng lên 150–200 câu, phân loại (tra cứu điều khoản / thủ tục / thời hạn / không có đáp án) | `eval/testset.jsonl` |
| 2.4 | Chia `dev` (tuning) / `test` (chỉ báo cáo) — tránh overfit bộ test | `eval/splits.json` |
| 2.5 | Hướng dẫn gán nhãn + thống kê bộ test | `docs/EVALUATION.md` |
| 2.6 | Báo cáo tự động: JSON + bảng Markdown | `evaluation/report.py` |

### 2B. Chunking + embedding · 2,5 ngày
| ID | Việc | File |
|---|---|---|
| 2.7 | Chunker giới hạn theo **token của model**; fallback tách câu; overlap theo token; test cho chunk 19K ký tự | `processing/chunker.py` |
| 2.8 | Gắn ngữ cảnh: `"{tên văn bản} > {Chương/Điều}\n{nội dung}"` (chỉ dùng khi embed, giữ text gốc để hiển thị) | `processing/chunker.py` |
| 2.9 | Tách từ tiếng Việt cho BM25 (pyvi / underthesea) + bỏ dấu tuỳ chọn; so sánh với `.split()` | `indexing/tokenizer.py`, `indexing/bm25.py` |
| 2.10 | Encoder chung hỗ trợ prefix (`query:`/`passage:`), tách từ cho PhoBERT, batch + fp16 GPU | `indexing/encoders.py` |
| 2.11 | Benchmark embedding: MiniLM (baseline), `multilingual-e5-base`, `bkai-foundation-models/vietnamese-bi-encoder`, `BAAI/bge-m3` | `eval/results/embeddings_*.json` |
| 2.12 | Manifest index (tên model, số chiều, hash dữ liệu, ngày build); báo lỗi khi model không khớp | `indexing/manifest.py` |
| 2.13 | FAISS: `IndexIDMap` + metadata chỉ lưu `chunk_id` (bỏ text trùng lặp) | `indexing/dense.py` |
| 2.14 | Tuning RRF (k, trọng số BM25/dense) trên **dev** | `retrieval/hybrid.py` |

### 2C. Reranker + Qdrant · 1,5–2 ngày
| ID | Việc | File |
|---|---|---|
| 2.15 | Cross-encoder rerank top-20 (`BAAI/bge-reranker-v2-m3`) | `retrieval/reranker.py` |
| 2.16 | Interface `VectorStore` → `FaissStore` + `QdrantStore` (chế độ embedded khi dev, server khi deploy) | `indexing/stores/` |
| 2.17 | Payload: `doc_type`, `issuer_unit`, `date_posted`, `doc_group` (vd. 3209), `is_latest` | `indexing/stores/qdrant.py` |
| 2.18 | Xử lý phiên bản quy chế: đánh dấu bản mới nhất trong cùng nhóm; lọc/ưu tiên bản còn hiệu lực | `processing/versioning.py` |
| 2.19 | Hybrid trong Qdrant (dense + sparse của bge-m3) — so sánh với BM25 + FAISS + RRF | `retrieval/hybrid.py` |
| 2.20 | Upsert tăng dần khi crawl thêm (chỉ embed chunk mới/đổi, theo hash) | `indexing/incremental.py` |
| 2.21 | Bảng **ablation** cuối: từng bước đóng góp bao nhiêu (trên tập test) | `docs/EVALUATION.md` |

### 2D. (Tuỳ chọn, sau khi 2B xong) Fine-tune embedding · 2 ngày
| ID | Việc |
|---|---|
| 2.22 | LLM sinh 2–3K cặp (câu hỏi, chunk) từ dữ liệu; lọc chất lượng; **loại chunk có trong tập test** |
| 2.23 | Fine-tune `MultipleNegativesRankingLoss` + hard negatives từ BM25 trên RTX 4050 |
| 2.24 | So sánh trên tập test; đẩy model lên Hugging Face Hub kèm model card |

**Xong khi:** có bảng ablation đo trên tập test; README/CV chỉ dùng số trong bảng này.

---

## Giai đoạn 3 — Sinh câu trả lời (RAG) · 3–4 ngày

Nhánh: `phase-3/generation`

| ID | Việc | File | Ước lượng |
|---|---|---|---|
| 3.1 | Interface LLM: Claude / OpenAI / Ollama (Qwen2.5-7B local) | `generation/llm.py` | 0,5 ngày |
| 3.2 | Prompt tiếng Việt: chỉ trả lời từ ngữ cảnh, **trích dẫn** `[Điều X — văn bản Y]`, từ chối khi không đủ căn cứ, nêu ngày ban hành | `generation/prompts.py` | 0,5 ngày |
| 3.3 | Pipeline retrieve → rerank → generate, streaming, trả về nguồn | `generation/rag.py` | 0,5 ngày |
| 3.4 | Kiểm tra trích dẫn: mọi `[n]` phải trỏ tới chunk đã đưa vào prompt | `generation/citations.py` | 0,5 ngày |
| 3.5 | Đánh giá câu trả lời: faithfulness, correctness (LLM-as-judge, có đối chiếu tay 20 câu để kiểm tra độ tin cậy của judge), tỉ lệ từ chối đúng | `evaluation/answer_eval.py` | 1 ngày |
| 3.6 | Ghi chi phí/độ trễ mỗi câu hỏi | `generation/rag.py` | 1h |

**Cần bạn:** chọn LLM; tự đặt API key vào `.env` (không gửi qua chat).

---

## Giai đoạn 4 — API & demo · 2–3 ngày

Nhánh: `phase-4/serving`

| ID | Việc | File |
|---|---|---|
| 4.1 | FastAPI: `/health`, `/ready`, `POST /v1/search` (có filter), `POST /v1/ask` (SSE streaming) | `api/main.py`, `api/routers/` |
| 4.2 | Schema Pydantic, lỗi chuẩn, request id, log JSON | `api/schemas.py`, `api/middleware.py` |
| 4.3 | Load model/index một lần (lifespan), dependency injection để test được | `api/deps.py` |
| 4.4 | Cache truy vấn, rate limit, timeout LLM | `api/` |
| 4.5 | `/metrics` Prometheus (latency, số request, cache hit) | `api/metrics.py` |
| 4.6 | Giao diện Gradio: chat, nguồn trích dẫn, link PDF gốc, bộ lọc phòng ban/năm | `ui/app.py` |

---

## Giai đoạn 5 — Test & CI/CD · 2 ngày

Nhánh: `phase-5/ci`

| ID | Việc | File |
|---|---|---|
| 5.1 | Unit test: chunker, tokenizer, rrf, bm25, metrics, versioning, citations, prompt | `tests/unit/` |
| 5.2 | Integration test trên corpus nhỏ ~30 chunk (model nhỏ, chạy CPU < 1 phút) | `tests/integration/`, `tests/fixtures/mini_corpus.json` |
| 5.3 | Test API bằng `TestClient` + retriever/LLM giả | `tests/api/` |
| 5.4 | Coverage ≥ 80% phần lõi | `pyproject.toml` |
| 5.5 | CI: ruff, mypy, pytest (3.11, 3.12), torch CPU, cache pip | `.github/workflows/ci.yml` |
| 5.6 | Eval regression trên corpus nhỏ: fail nếu metric tụt quá ngưỡng | `.github/workflows/eval.yml` |
| 5.7 | Dockerfile multi-stage, non-root, healthcheck, gunicorn + uvicorn workers | `Dockerfile` |
| 5.8 | `docker-compose.yml`: api + qdrant + ui, volume cho index | `docker-compose.yml` |
| 5.9 | Build & push image lên GHCR khi tag | `.github/workflows/release.yml` |

---

## Giai đoạn 6 — Dữ liệu, deploy, tài liệu · 2 ngày

Nhánh: `phase-6/release`

| ID | Việc |
|---|---|
| 6.1 | Đưa PDF, index ra khỏi git → Hugging Face Hub (dataset) hoặc GitHub Release; `make data` để tải |
| 6.2 | Data card: nguồn, thời gian crawl, license/ghi chú bản quyền, thống kê |
| 6.3 | Deploy demo lên Hugging Face Spaces (Docker) + Qdrant Cloud free tier |
| 6.4 | README tiếng Anh: badges, link demo, GIF, sơ đồ kiến trúc (mermaid), bảng kết quả, quickstart 3 lệnh, quyết định thiết kế, hạn chế |
| 6.5 | `docs/ARCHITECTURE.md`, `docs/EVALUATION.md`, `CHANGELOG.md` |
| 6.6 | Tag `v2.0.0`, GitHub Release |
| 6.7 | Viết bullet CV + câu chuyện phỏng vấn (vấn đề → cách đo → giải pháp → kết quả) |

---

## Tổng thời gian

| Giai đoạn | Ước lượng | Tích luỹ |
|---|---|---|
| 0 — Dọn dẹp & baseline | 1 | 1 |
| 1 — Cấu trúc | 1,5 | 2,5 |
| 2 — Retrieval (2A–2C) | 5–6 | 8 |
| 2D — Fine-tune (tuỳ chọn) | 2 | 10 |
| 3 — RAG | 3–4 | 14 |
| 4 — API & demo | 2–3 | 17 |
| 5 — Test & CI/CD | 2 | 19 |
| 6 — Release & docs | 2 | 21 |

**Bản tối thiểu để đưa lên CV** (~10 ngày): GĐ 0 → 1 → 2A/2B/2C → README (6.4) → demo Gradio đơn giản.

## Việc cần bạn

| Khi nào | Việc |
|---|---|
| Trước GĐ0 | Duyệt kế hoạch này |
| GĐ 2A | Duyệt nhãn bộ test (~3–4h) |
| GĐ 3 | Chọn LLM, đặt API key vào `.env` |
| GĐ 6 | Tài khoản Hugging Face (`huggingface-cli login`); quyết định công khai PDF hay chỉ link nguồn |
| Cuối mỗi GĐ | Review & merge PR |

## Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Ổ D còn ~11 GB; model tải về ổ C | Đặt `HF_HOME` sang ổ có chỗ trống; dọn cache model không dùng |
| RTX 4050 6 GB không đủ fine-tune bge-m3 | Fine-tune model base (e5-base / vietnamese-bi-encoder), batch nhỏ + gradient checkpointing |
| Tuning quá tay trên bộ test | Tách dev/test (2.4), chỉ báo cáo trên test |
| LLM-as-judge thiên lệch | Đối chiếu tay 20 câu, báo cáo mức đồng thuận |
| Portal thay đổi cấu trúc HTML | Lưu snapshot HTML trong fixture, test parser trên snapshot |

## Bảng kết quả (điền sau khi đo)

| Cấu hình | nDCG@10 | Recall@10 | MRR | p95 latency |
|---|---|---|---|---|
| Baseline BM25 (`.split()`) | | | | |
| Baseline Hybrid (MiniLM) | | | | |
| + chunker mới + ngữ cảnh | | | | |
| + tách từ tiếng Việt | | | | |
| + embedding tốt nhất | | | | |
| + reranker | | | | |
| + fine-tune (tuỳ chọn) | | | | |
