---
language:
- vi
pretty_name: HUFLIT academic regulations and announcements
size_categories:
- 1K<n<10K
task_categories:
- question-answering
- text-retrieval
---

# HUFLIT academic regulations and announcements

Regulations (quy chế, quy định) and student announcements (thông báo) published on the student portal of Ho Chi Minh City University of Foreign Languages – Information Technology (HUFLIT), crawled for the [vietnamese-qa-platform](https://github.com/bebaobanhbaoo-create/vietnamese-qa-platform) retrieval and RAG project.

**Rights.** The documents belong to HUFLIT. They are public on the portal; this copy is kept for research and reproducibility of the project, not for redistribution. Always check the original page (`source_url` / `doc_url`) before relying on a rule.

## Contents

| Folder | What | Produced by |
|---|---|---|
| `raw/` | Article metadata and page text from the portal (434 articles) | `huflit-qa scrape` |
| `attachments/` | Regulation attachments: PDF/JPG plus extracted `.txt` (some corrected by hand) | download + `huflit-qa extract-pdf` |
| `processed/` | Records split by type; `*_flat.json` = one record per page or attachment (516) | `huflit-qa process` |
| `index/` | Shipped index: 1,209 chunks, BM25, embedded Qdrant (`qdrant/`), `manifest.json` | `huflit-qa index` |

## Statistics

| | |
|---|---|
| Documents (pages + attachments) | 516 |
| Chunks (v1 chunker) | 1,209: 715 regulation, 494 announcement |
| Posting dates | 07/2020 – 03/2026 |
| Issuing units | Chung, Phòng Đào tạo, Phòng CTSV, Phòng Tài chính, Phòng Khảo thí, Bộ phận Y tế |
| Regulation groups with several versions | e.g. language/IT exit requirements: 2017, 2020, 2024, 2025 |

## Known limitations

- Announcements are time-bound; many describe deadlines that have passed.
- Some regulation groups hold successive versions and cohort-specific rules side by side; `issued_date` and `newest_in_group` on chunks help but do not decide which rule applies to a student.
- Scanned attachments (images) have no text.
- The newest page in the crawl was posted on 2026-03-20; the portal changes over time.

## Usage

```bash
huggingface-cli login            # the dataset is private
huflit-qa download-data          # -> data/
```
