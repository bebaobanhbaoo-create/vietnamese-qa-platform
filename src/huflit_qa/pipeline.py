"""Offline pipeline steps 2 (process) and 3 (chunk + index)."""

import json
import logging
from pathlib import Path

from huflit_qa.config import Settings
from huflit_qa.processing.chunker import chunk_all
from huflit_qa.processing.flatten import flatten_all
from huflit_qa.processing.records import build_flat_records

logger = logging.getLogger(__name__)


def _write_json(data: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def process(settings: Settings) -> None:
    """quy_che.json + attachments -> quy_che_flat.json; thong_bao.json -> thong_bao_flat.json."""
    records = build_flat_records(settings.quy_che_path, settings.quy_che_attachments_dir, settings.root_dir)
    _write_json(records, settings.quy_che_flat_path)
    logger.info("%d quy_che records -> %s", len(records), settings.quy_che_flat_path)

    tb_records = flatten_all(settings.thong_bao_path)
    _write_json(tb_records, settings.thong_bao_flat_path)
    logger.info("%d thong_bao records -> %s", len(tb_records), settings.thong_bao_flat_path)


def build_index(settings: Settings) -> None:
    """Chunk the flat records and build the BM25 and FAISS indexes."""
    # Imported here so `process` does not pay for loading torch/faiss.
    from huflit_qa.indexing.bm25 import build_bm25, save_bm25
    from huflit_qa.indexing.dense import build_dense, save_dense

    chunks = chunk_all(settings.quy_che_flat_path, settings.thong_bao_flat_path, settings.chunking)
    _write_json(chunks, settings.chunks_path)
    logger.info("%d chunks -> %s", len(chunks), settings.chunks_path)

    bm25, chunk_ids = build_bm25(chunks)
    save_bm25(bm25, chunk_ids, settings.bm25_path)
    logger.info("BM25 -> %s", settings.bm25_path)

    logger.info("Encoding chunks with %s", settings.embedding_model)
    faiss_index, meta = build_dense(chunks, settings.embedding_model)
    save_dense(faiss_index, meta, settings.faiss_path, settings.faiss_meta_path)
    logger.info("FAISS -> %s", settings.faiss_path)
