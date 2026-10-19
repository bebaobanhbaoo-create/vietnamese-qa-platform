"""Pipeline settings. Every field can be overridden with a ``HUFLIT_`` env var or a ``.env`` file,
e.g. ``HUFLIT_EMBEDDING__MODEL=...`` or ``HUFLIT_CHUNKING__STRATEGY=v2``.

Chunking, BM25 and embedding settings are used when *building* an index and are recorded in its
manifest; at query time the retriever follows the manifest, so an index is always searched the way
it was built.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingSettings(BaseModel):
    # v1: split by Điều/Khoản/paragraph with character limits (the original chunker).
    # v2: same structure, but packed up to a word budget and never cut mid-sentence.
    strategy: Literal["v1", "v2"] = "v1"
    # v1 sizes, in characters. The committed index was built with overlap 0; an 80-char overlap
    # lowered Hybrid R@10 0.723 -> 0.669.
    size: int = 400
    overlap: int = 0
    min_size: int = 100
    # v2 sizes, in words.
    max_words: int = 200
    min_words: int = 40
    # Prepend "document title > chapter > article" to the text that is indexed (BM25 and dense).
    context_prefix: bool = False


class Bm25Settings(BaseModel):
    # whitespace: lowercase + split (original); words: also drops punctuation;
    # pyvi: Vietnamese word segmentation (e.g. "tốt_nghiệp") + punctuation removal.
    tokenizer: Literal["whitespace", "words", "pyvi"] = "whitespace"


class EmbeddingSettings(BaseModel):
    model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    # Truncation length; capped by the model's own limit.
    max_seq_length: int = 512
    batch_size: int = 32
    # The original pipeline lowercased queries (but not passages) before encoding.
    lowercase_query: bool = True


class RetrievalSettings(BaseModel):
    top_k_bm25: int = 20
    top_k_dense: int = 20
    top_k: int = 10
    rrf_k: int = 60


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HUFLIT_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    # Project root; stored attachment paths (``pdf_path``) are relative to it.
    root_dir: Path = Path.cwd()
    # Defaults to data/index; experiments build into their own directories.
    index_path: Path | None = None
    chunking: ChunkingSettings = ChunkingSettings()
    bm25: Bm25Settings = Bm25Settings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def attachments_dir(self) -> Path:
        return self.data_dir / "attachments"

    @property
    def quy_che_attachments_dir(self) -> Path:
        return self.attachments_dir / "quy_ch"

    @property
    def by_doc_type_dir(self) -> Path:
        return self.processed_dir / "by_doc_type"

    @property
    def quy_che_path(self) -> Path:
        return self.by_doc_type_dir / "quy_che.json"

    @property
    def thong_bao_path(self) -> Path:
        return self.by_doc_type_dir / "thong_bao.json"

    @property
    def quy_che_flat_path(self) -> Path:
        return self.by_doc_type_dir / "quy_che_flat.json"

    @property
    def thong_bao_flat_path(self) -> Path:
        return self.by_doc_type_dir / "thong_bao_flat.json"

    @property
    def index_dir(self) -> Path:
        return self.index_path or self.data_dir / "index"

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.json"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    @property
    def bm25_path(self) -> Path:
        return self.index_dir / "bm25.pkl"

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "faiss.bin"

    @property
    def faiss_ids_path(self) -> Path:
        return self.index_dir / "chunk_ids.json"

    @property
    def eval_dir(self) -> Path:
        return self.root_dir / "eval"

    @property
    def testset_path(self) -> Path:
        return self.eval_dir / "testset.jsonl"

    @property
    def label_units_path(self) -> Path:
        return self.eval_dir / "label_units.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
