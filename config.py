"""Shared pipeline configuration constants."""

# Chunking
CHUNK_SIZE = 400
CHUNK_OVERLAP = 0  # committed index was built with 0; 80-char overlap lowered Hybrid R@10 0.723 -> 0.669
MIN_CHUNK_SIZE = 100

# Retrieval
TOP_K_BM25 = 20
TOP_K_DENSE = 20
TOP_K_RESULTS = 10
RRF_K = 60

# Embedding
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
