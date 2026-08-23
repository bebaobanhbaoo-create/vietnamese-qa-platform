class SearchService:
    def __init__(self, index, embedder):
        self.index = index
        self.embedder = embedder

    def search(self, query, k=5):
        emb = self.embedder.encode([query])[0]
        return self.index.search(emb, k)
