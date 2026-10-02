"""Hybrid retriever: BM25 + FAISS + RRF."""
import json
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TOP_K_BM25, TOP_K_DENSE, TOP_K_RESULTS, RRF_K, EMBEDDING_MODEL
from offline.ir_build.bm25 import load_bm25, search_bm25
from offline.ir_build.dense import load_dense, search_dense
from online.retrieval.rrf import reciprocal_rank_fusion

BASE_DIR = Path(__file__).parent.parent.parent


class HybridRetriever:
    def __init__(self):
        bm25_path = BASE_DIR / 'data/index/bm25.pkl'
        faiss_path = BASE_DIR / 'data/index/faiss.bin'
        meta_path = BASE_DIR / 'data/index/faiss_meta.json'
        chunks_path = BASE_DIR / 'data/processed/chunks.json'

        self.bm25, self.bm25_ids = load_bm25(bm25_path)
        self.faiss_index, self.faiss_meta = load_dense(faiss_path, meta_path)
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(EMBEDDING_MODEL)

        chunks = json.loads(chunks_path.read_text(encoding='utf-8'))
        self.chunk_map = {}
        for c in chunks:
            self.chunk_map[c['chunk_id']] = c

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> list:
        """Return top-k chunks for query."""
        q = query.lower().strip()
        bm25_results = search_bm25(self.bm25, self.bm25_ids, q, TOP_K_BM25)
        dense_results = search_dense(self.faiss_index, self.faiss_meta, self.model, q, TOP_K_DENSE)
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=RRF_K)

        output = []
        for chunk_id, score in fused[:top_k]:
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            output.append({
                'chunk_id': chunk_id,
                'text': chunk['text'],
                'doc_title': chunk['doc_title'],
                'chunk_title': chunk.get('chunk_title'),
                'source_url': chunk['source_url'],
                'date_posted': chunk.get('date_posted'),
                'score': score,
            })
        return output
