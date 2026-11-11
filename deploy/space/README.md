---
title: HUFLIT Regulation Search
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8080
pinned: false
short_description: Hybrid search over HUFLIT academic regulations (Vietnamese)
---

# HUFLIT Regulation Search

Search demo for [vietnamese-qa-platform](https://github.com/bebaobanhbaoo-create/vietnamese-qa-platform): hybrid retrieval (BM25 + multilingual embeddings, Reciprocal Rank Fusion) over 1,209 passages of the academic regulations and announcements published on the HUFLIT student portal, with filters for document type and newest regulation version.

This Space runs on CPU, so answer generation is switched off (`HUFLIT_GENERATION__ENABLED=false`). The full pipeline, with cited answers from a local LLM through Ollama, runs from the GitHub repository.

The documents belong to HUFLIT; check the original page linked under each result before relying on a rule.
