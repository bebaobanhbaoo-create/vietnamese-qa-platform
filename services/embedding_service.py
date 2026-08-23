class EmbeddingService:
    def __init__(self, embedder):
        self.embedder = embedder
        self.cache = {}

    def get_embedding(self, text):
        if text in self.cache:
            return self.cache[text]
        emb = self.embedder.encode([text])[0]
        self.cache[text] = emb
        return emb
