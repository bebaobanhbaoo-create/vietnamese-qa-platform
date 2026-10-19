"""Index manifest: how an index was built, so it is searched the same way and can be traced."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from huflit_qa import __version__
from huflit_qa.config import Settings


def file_digest(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def build_manifest(settings: Settings, num_chunks: int, dim: int) -> dict:
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "source_digest": file_digest([settings.quy_che_flat_path, settings.thong_bao_flat_path]),
        "num_chunks": num_chunks,
        "chunking": settings.chunking.model_dump(),
        "bm25": settings.bm25.model_dump(),
        "embedding": {**settings.embedding.model_dump(), "dim": dim},
        "index": settings.index.model_dump(),
    }


def save_manifest(manifest: dict, path: Path) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: rebuild the index with `huflit-qa index`.")
    return json.loads(path.read_text(encoding="utf-8"))
