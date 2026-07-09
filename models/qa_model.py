class QAModel:
    def __init__(self, embedder, index):
        self.embedder = embedder
        self.index = index

    def add_documents(self, docs):
        embeddings = self.embedder.encode(docs)
        self.index.add(embeddings)

    def query(self, question, k=5):
        q_emb = self.embedder.encode([question])[0]
        return self.index.search(q_emb, k)

    def answer(self, question, documents, k=5):
        indices, _ = self.query(question, k)
        return [documents[i] for i in indices]
