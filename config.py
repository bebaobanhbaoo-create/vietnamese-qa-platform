"""Shared pipeline configuration constants."""

# Chunking
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
MIN_CHUNK_SIZE = 100

# Retrieval
TOP_K_BM25 = 20
TOP_K_DENSE = 20
TOP_K_RESULTS = 10
RRF_K = 60

# Embedding
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
