"""Pipeline settings. Every field can be overridden with a ``HUFLIT_`` env var or a ``.env`` file,
e.g. ``HUFLIT_EMBEDDING_MODEL=...`` or ``HUFLIT_CHUNKING__SIZE=600``."""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingSettings(BaseModel):
    """Chunk sizes are measured in characters."""

    size: int = 400
    # The committed index was built with 0; an 80-char overlap lowered Hybrid R@10 0.723 -> 0.669.
    overlap: int = 0
    min_size: int = 100


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
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    chunking: ChunkingSettings = ChunkingSettings()
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
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.json"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def bm25_path(self) -> Path:
        return self.index_dir / "bm25.pkl"

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "faiss.bin"

    @property
    def faiss_meta_path(self) -> Path:
        return self.index_dir / "faiss_meta.json"

    @property
    def eval_dir(self) -> Path:
        return self.root_dir / "eval"

    @property
    def testset_path(self) -> Path:
        return self.eval_dir / "testset.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
