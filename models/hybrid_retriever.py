import numpy as np
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

class HybridRetriever:
    """Hybrid retrieval combining BM25 and dense embeddings with RRF."""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.bm25 = None
        self.faiss_index = None
        self.documents = []
    
    def index(self, documents: List[str]):
        """Index documents for both sparse and dense retrieval."""
        self.documents = documents
        
        # BM25 indexing
        tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        
        # Dense indexing with FAISS
        embeddings = self.encoder.encode(documents, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
    
    def search(self, query: str, k: int = 10, alpha: float = 0.5) -> List[Tuple[int, float]]:
        """Search using Reciprocal Rank Fusion."""
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_ranks = np.argsort(-bm25_scores)
        
        q_emb = self.encoder.encode([query])
        faiss.normalize_L2(q_emb)
        _, dense_ranks = self.faiss_index.search(q_emb, len(self.documents))
        dense_ranks = dense_ranks[0]
        
        # RRF fusion
        rrf_scores = {}
        for rank, idx in enumerate(bm25_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + alpha / (60 + rank)
        for rank, idx in enumerate(dense_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1-alpha) / (60 + rank)
        
        results = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]
        return results
