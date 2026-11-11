"""Export the shipped index for the in-browser search demo (GitHub Pages).

    python scripts/build_static_demo.py           # -> deploy/static/data/

Writes the chunk texts and metadata, the dense vectors exactly as stored in Qdrant (float32), and the
settings the page must reproduce: BM25 tokenizer, RRF constant, retrieval depths, embedding model.
The page re-implements BM25Okapi and RRF in JavaScript and embeds the query in the browser.
Needs only numpy and qdrant-client, not torch.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ROOT / "src" / "huflit_qa" / "api" / "static"
SHARED_ASSETS = ("app.css", "ui.js")
FIELDS = (
    "chunk_id", "doc_title", "chunk_title", "doc_type", "source_url",
    "date_posted", "issued_date", "issuer_unit", "newest_in_group", "on_portal", "text",
)  # fmt: skip
# Browser builds of the embedding models (transformers.js / ONNX).
BROWSER_MODELS = {"paraphrase-multilingual-MiniLM-L12-v2": "Xenova/paraphrase-multilingual-MiniLM-L12-v2"}


def read_vectors(qdrant_dir: Path, chunk_ids: list[str]) -> np.ndarray:
    """Vectors of the embedded Qdrant collection, in chunk order."""
    client = QdrantClient(path=str(qdrant_dir))
    by_id: dict = {}
    offset = None
    while True:
        points, offset = client.scroll(
            "chunks", with_payload=["chunk_id"], with_vectors=True, limit=512, offset=offset
        )
        by_id.update({(p.payload or {})["chunk_id"]: p.vector for p in points})
        if offset is None:
            break
    client.close()
    if set(by_id) != set(chunk_ids):
        raise SystemExit("Qdrant collection and chunks.json disagree: rebuild the index")
    return np.asarray([by_id[cid] for cid in chunk_ids], dtype="<f4")


def build(index_dir: Path, out_dir: Path) -> dict:
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["bm25"]["tokenizer"] != "whitespace" or manifest["chunking"].get("context_prefix"):
        raise SystemExit("The browser demo reproduces whitespace BM25 on raw chunk text only.")
    model = manifest["embedding"]["model"]
    if model not in BROWSER_MODELS:
        raise SystemExit(f"No browser build known for {model}")

    chunk_list = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    chunk_ids = [c["chunk_id"] for c in chunk_list]  # the order BM25 was built in
    chunks = {c["chunk_id"]: c for c in chunk_list}
    vectors = read_vectors(index_dir / "qdrant", chunk_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{f: chunks[cid].get(f) for f in FIELDS} for cid in chunk_ids]
    (out_dir / "chunks.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (out_dir / "vectors.f32").write_bytes(np.ascontiguousarray(vectors).tobytes())
    meta = {
        "num_chunks": len(rows),
        "dim": int(vectors.shape[1]),
        "model": BROWSER_MODELS[model],
        "lowercase_query": manifest["embedding"].get("lowercase_query", True),
        "bm25": {"k1": 1.5, "b": 0.75, "epsilon": 0.25},  # rank_bm25.BM25Okapi defaults
        "retrieval": {"top_k_bm25": 20, "top_k_dense": 20, "rrf_k": 60},
        "built_from": {
            "source_digest": manifest.get("source_digest"),
            "created_at": manifest.get("created_at"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # The page shares its stylesheet and UI helpers with the API's chat page.
    for name in SHARED_ASSETS:
        shutil.copy2(SHARED_DIR / name, out_dir.parent / name)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "index")
    parser.add_argument("--out", type=Path, default=ROOT / "deploy" / "static" / "data")
    args = parser.parse_args()
    meta = build(args.index, args.out)
    size = sum(p.stat().st_size for p in args.out.iterdir()) / 1e6
    print(f"{meta['num_chunks']} chunks, dim {meta['dim']}, {size:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
