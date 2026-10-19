"""Offline pipeline steps 2 (process) and 3 (chunk + index)."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from huflit_qa.config import Settings
from huflit_qa.processing.chunker import chunk_all, index_text
from huflit_qa.processing.flatten import flatten_all
from huflit_qa.processing.records import build_flat_records

if TYPE_CHECKING:
    from huflit_qa.indexing.encoders import Encoder

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


def build_index(settings: Settings, encoder: "Encoder | None" = None) -> None:
    """Chunk the flat records, build the BM25 and FAISS indexes and write the manifest.

    Pass an already-loaded ``encoder`` to build several indexes with the same model.
    """
    # Imported here so `process` does not pay for loading torch/faiss.
    from huflit_qa.indexing.bm25 import build_bm25, save_bm25
    from huflit_qa.indexing.dense import build_dense, save_dense
    from huflit_qa.indexing.encoders import Encoder
    from huflit_qa.indexing.manifest import build_manifest, save_manifest
    from huflit_qa.indexing.tokenize import get_tokenizer

    chunks = chunk_all(settings.quy_che_flat_path, settings.thong_bao_flat_path, settings.chunking)
    _write_json(chunks, settings.chunks_path)
    logger.info("%d chunks -> %s", len(chunks), settings.chunks_path)

    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [index_text(c, settings.chunking.context_prefix) for c in chunks]

    bm25 = build_bm25(texts, get_tokenizer(settings.bm25.tokenizer))
    save_bm25(bm25, chunk_ids, settings.bm25_path)
    logger.info("BM25 (%s) -> %s", settings.bm25.tokenizer, settings.bm25_path)

    emb = settings.embedding
    if encoder is None:
        encoder = Encoder(emb.model, emb.max_seq_length, emb.batch_size)
    logger.info("Encoding %d chunks with %s", len(chunks), emb.model)
    vectors = encoder.encode_passages(texts)
    save_dense(build_dense(vectors), chunk_ids, settings.faiss_path, settings.faiss_ids_path)
    logger.info("FAISS -> %s", settings.faiss_path)

    save_manifest(build_manifest(settings, len(chunks), encoder.dim), settings.manifest_path)
